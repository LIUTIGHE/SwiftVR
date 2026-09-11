"""Stage-3 reconstruction objectives, metrics, and cursor helpers for SwiftVR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .loop import TrainingCursor


def _validate_video_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("prediction and target must be torch tensors")
    if prediction.ndim != 5 or target.ndim != 5:
        raise ValueError(
            "prediction and target must use [B,T,C,H,W], got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}"
        )
    if prediction.shape[2] != 3:
        raise ValueError(f"Expected RGB videos, got C={prediction.shape[2]}")
    return prediction.float(), target.float()


def temporal_difference_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE between consecutive-frame differences used by SwiftVR Stage 3."""

    prediction, target = _validate_video_pair(prediction, target)
    if prediction.shape[1] < 2:
        return prediction.new_zeros(())
    prediction_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    return F.mse_loss(prediction_delta, target_delta)


def stage3_reconstruction_objective(
    output: Mapping[str, object],
    *,
    pixel_weight: float = 1.0,
    temporal_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Build the non-adversarial Stage-3 objective from a training-forward output."""

    if pixel_weight < 0 or temporal_weight < 0:
        raise ValueError("Stage-3 loss weights must be non-negative")
    prediction = output.get("prediction")
    target = output.get("target")
    pixel_l1 = output.get("pixel_l1")
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("Training output must contain tensor prediction and target")
    if not isinstance(pixel_l1, torch.Tensor) or pixel_l1.ndim != 0:
        raise TypeError("Training output must contain scalar tensor pixel_l1")

    temporal_mse = temporal_difference_mse(prediction, target)
    loss = float(pixel_weight) * pixel_l1 + float(temporal_weight) * temporal_mse
    return {
        "loss": loss,
        "pixel_l1": pixel_l1,
        "temporal_mse": temporal_mse,
    }


def advance_cursor_batches(
    cursor: TrainingCursor,
    *,
    consumed_batches: int,
    batches_per_epoch: int,
    optimizer_steps: int = 1,
) -> TrainingCursor:
    """Advance the dataloader cursor and optimizer-step counter independently."""

    if consumed_batches < 0:
        raise ValueError("consumed_batches must be non-negative")
    if optimizer_steps < 0:
        raise ValueError("optimizer_steps must be non-negative")
    if batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch must be positive")
    total = cursor.batch_in_epoch + int(consumed_batches)
    return TrainingCursor(
        global_step=cursor.global_step + int(optimizer_steps),
        epoch=cursor.epoch + total // int(batches_per_epoch),
        batch_in_epoch=total % int(batches_per_epoch),
    )


def _gaussian_kernel(
    channels: int,
    *,
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def video_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Mean frame-wise RGB SSIM for videos in [0, 1]."""

    prediction, target = _validate_video_pair(prediction, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    batch, frames, channels, height, width = prediction.shape
    prediction = prediction.reshape(batch * frames, channels, height, width)
    target = target.reshape(batch * frames, channels, height, width)
    effective_window = min(window_size, height, width)
    if effective_window % 2 == 0:
        effective_window -= 1
    effective_window = max(effective_window, 1)
    kernel = _gaussian_kernel(
        channels,
        window_size=effective_window,
        sigma=sigma,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    padding = effective_window // 2

    mu_x = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(prediction.square(), kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target.square(), kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, kernel, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)).mean()


@dataclass
class VideoMetricAccumulator:
    """Aggregate full-reference metrics over validation batches."""

    sum_abs: float = 0.0
    sum_squared: float = 0.0
    elements: int = 0
    sum_ssim: float = 0.0
    ssim_frames: int = 0
    batches: int = 0

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        clamp: bool = True,
    ) -> None:
        prediction, target = _validate_video_pair(prediction, target)
        if clamp:
            prediction = prediction.clamp(0.0, 1.0)
            target = target.clamp(0.0, 1.0)
        difference = prediction - target
        self.sum_abs += float(difference.abs().sum().item())
        self.sum_squared += float(difference.square().sum().item())
        self.elements += int(difference.numel())
        frames = int(prediction.shape[0] * prediction.shape[1])
        self.sum_ssim += float(video_ssim(prediction, target).item()) * frames
        self.ssim_frames += frames
        self.batches += 1

    def compute(self) -> dict[str, float | int]:
        if self.elements <= 0 or self.ssim_frames <= 0:
            raise RuntimeError("No validation samples were accumulated")
        mae = self.sum_abs / self.elements
        mse = self.sum_squared / self.elements
        rmse = math.sqrt(mse)
        psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "psnr": psnr,
            "ssim": self.sum_ssim / self.ssim_frames,
            "batches": self.batches,
            "frames": self.ssim_frames,
            "elements": self.elements,
        }
