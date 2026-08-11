"""Stage-B tiny-decoder objectives and perceptual-loss adapter."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage3 import temporal_difference_mse


def _validate_pair(name: str, prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.ndim != 5 or target.ndim != 5:
        raise ValueError(f"{name} videos must be [B,T,C,H,W]")
    if prediction.shape != target.shape:
        raise ValueError(
            f"{name} shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if int(prediction.shape[2]) != 3:
        raise ValueError(f"{name} expects RGB videos")


class LPIPSAlexLoss(nn.Module):
    """Differentiable frame-wise LPIPS using the reference ``lpips`` package.

    The dependency is deliberately optional and loaded lazily so architecture/MAC
    gates can run without it. Formal B1-A training should use the default LPIPS
    weight rather than silently falling back to a different perceptual objective.
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            import lpips  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "LPIPS supervision requested but the optional 'lpips' package is "
                "unavailable. Install requirements-stage-b.txt or run an explicit "
                "MSE-only smoke with --lpips-weight 0."
            ) from exc
        self.metric = lpips.LPIPS(net="alex", verbose=False).eval()
        for parameter in self.metric.parameters():
            parameter.requires_grad_(False)

    def forward_video(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        microbatch_frames: int = 16,
    ) -> torch.Tensor:
        _validate_pair("LPIPS", prediction, target)
        microbatch_frames = int(microbatch_frames)
        if microbatch_frames <= 0:
            raise ValueError("microbatch_frames must be positive")

        pred = prediction.float().clamp(0.0, 1.0).flatten(0, 1).mul(2.0).sub(1.0)
        ref = target.detach().float().clamp(0.0, 1.0).flatten(0, 1).mul(2.0).sub(1.0)
        total_frames = int(pred.shape[0])
        if total_frames <= 0:
            raise ValueError("LPIPS received an empty video")

        weighted = pred.new_zeros(())
        for start in range(0, total_frames, microbatch_frames):
            end = min(start + microbatch_frames, total_frames)
            values = self.metric(pred[start:end], ref[start:end])
            weighted = weighted + values.float().mean() * (end - start)
        return weighted / total_frames


def tiny_decoder_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reae_teacher: torch.Tensor,
    *,
    perceptual: LPIPSAlexLoss | None = None,
    gt_l2_weight: float = 1.0,
    teacher_l2_weight: float = 1.0,
    lpips_weight: float = 2.0,
    lpips_microbatch_frames: int = 16,
) -> dict[str, torch.Tensor]:
    """FlashVSR-style dual reconstruction supervision for the tiny decoder.

    The same SR latent is decoded by the frozen ReAE decoder and by the tiny
    decoder. The tiny output is anchored both to GT pixels and to the original
    ReAE rendering behavior:

      L = w_gt * MSE(pred, GT)
        + w_teacher * MSE(pred, ReAE(z_SR))
        + w_lpips * [LPIPS(pred, GT) + LPIPS(pred, ReAE(z_SR))].

    ``lpips_weight=2`` is the initial B1-A recipe. Temporal error is returned as
    a diagnostic only; it is intentionally not another optimization term in the
    first decoder gate.
    """

    _validate_pair("prediction/GT", prediction, target)
    _validate_pair("prediction/ReAE", prediction, reae_teacher)
    weights = {
        "gt_l2_weight": float(gt_l2_weight),
        "teacher_l2_weight": float(teacher_l2_weight),
        "lpips_weight": float(lpips_weight),
    }
    negative = [name for name, value in weights.items() if value < 0.0]
    if negative:
        raise ValueError(f"Tiny-decoder loss weights must be non-negative: {negative}")
    if weights["lpips_weight"] > 0.0 and perceptual is None:
        raise ValueError("Positive lpips_weight requires a perceptual loss module")

    pred_f = prediction.float()
    target_f = target.detach().float()
    teacher_f = reae_teacher.detach().float()
    gt_l2 = F.mse_loss(pred_f, target_f)
    teacher_l2 = F.mse_loss(pred_f, teacher_f)
    zero = gt_l2.new_zeros(())

    gt_lpips = zero
    teacher_lpips = zero
    if weights["lpips_weight"] > 0.0:
        assert perceptual is not None
        gt_lpips = perceptual.forward_video(
            prediction,
            target,
            microbatch_frames=lpips_microbatch_frames,
        )
        teacher_lpips = perceptual.forward_video(
            prediction,
            reae_teacher,
            microbatch_frames=lpips_microbatch_frames,
        )

    loss = (
        weights["gt_l2_weight"] * gt_l2
        + weights["teacher_l2_weight"] * teacher_l2
        + weights["lpips_weight"] * (gt_lpips + teacher_lpips)
    )
    return {
        "loss": loss,
        "gt_l2": gt_l2,
        "teacher_l2": teacher_l2,
        "gt_lpips": gt_lpips,
        "teacher_lpips": teacher_lpips,
        "gt_temporal_mse": temporal_difference_mse(pred_f, target_f),
        "teacher_temporal_mse": temporal_difference_mse(pred_f, teacher_f),
    }
