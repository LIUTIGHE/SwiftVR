from __future__ import annotations

import torch


LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def residual_new_p8_phase_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    *,
    clamp: bool = True,
) -> torch.Tensor:
    """Penalize Tiny-vs-ReAE phase structure that is new at period 8.

    The loss mirrors the Stage-B1 phase diagnostic, but computes phase energy per
    sample/frame before averaging so opposite phase signs across video frames do
    not cancel.  A period-4 parent is removed before the RMS is computed, making
    the loss specific to the new p8 component rather than inherited p2/p4 modes.

    Args:
        prediction: Tiny decoder RGB video, shaped [B, T, 3, H, W].
        teacher: ReAE RGB video with the same shape.
        clamp: Clamp both visible RGB outputs to [0, 1], matching the phase audit.
    """
    if prediction.shape != teacher.shape:
        raise ValueError(
            f"prediction/teacher shape mismatch: {tuple(prediction.shape)} != {tuple(teacher.shape)}"
        )
    if prediction.ndim != 5 or int(prediction.shape[2]) != 3:
        raise ValueError(
            f"prediction/teacher must be [B,T,3,H,W], got {tuple(prediction.shape)}"
        )

    prediction_f = prediction.float()
    teacher_f = teacher.float()
    if clamp:
        prediction_f = prediction_f.clamp(0.0, 1.0)
        teacher_f = teacher_f.clamp(0.0, 1.0)
    residual = prediction_f - teacher_f

    weights = residual.new_tensor(LUMA_WEIGHTS).reshape(1, 1, 3, 1, 1)
    luma = (residual * weights).sum(dim=2)
    height, width = int(luma.shape[-2]), int(luma.shape[-1])
    usable_h = (height // 8) * 8
    usable_w = (width // 8) * 8
    if usable_h <= 0 or usable_w <= 0:
        raise ValueError(f"p8 phase loss requires H,W >= 8, got {height}x{width}")

    luma = luma[..., :usable_h, :usable_w]
    batch, frames = int(luma.shape[0]), int(luma.shape[1])
    nh, nw = usable_h // 8, usable_w // 8
    phase8 = luma.reshape(batch, frames, nh, 8, nw, 8).mean(dim=(2, 4))

    # M4(i,j) = mean M8(a,b) for a mod 4 == i and b mod 4 == j.
    parent4 = phase8.reshape(batch, frames, 2, 4, 2, 4).mean(dim=(2, 4))
    parent4_tiled = parent4.repeat(1, 1, 2, 2)
    novel8 = phase8 - parent4_tiled

    # RMS within each frame first, then average frames/samples. This prevents
    # opposite-signed frame artifacts from cancelling before the penalty.
    per_frame_rms = novel8.square().mean(dim=(-2, -1)).sqrt()
    return per_frame_rms.mean()


def calibrated_phase_weight(
    base_grad_norm: float,
    phase_grad_norm: float,
    *,
    target_ratio: float = 0.1,
) -> float:
    """Return lambda so lambda*phase gradient has the requested norm ratio."""
    if base_grad_norm <= 0:
        raise ValueError("base_grad_norm must be positive")
    if phase_grad_norm <= 0:
        raise ValueError("phase_grad_norm must be positive")
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")
    return float(target_ratio) * float(base_grad_norm) / float(phase_grad_norm)
