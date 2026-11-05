"""M12-2: Slim100 detail-preserving, GT-relative temporal restoration loss.

All video tensors are [B,T,3,H,W] and refer to the SAME M11-B restored latents.
Teacher detail is a visual reference, not the correct brightness/flicker target.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def scene_pair_valid(gt: torch.Tensor, *, cut_hist_threshold: float = 0.65) -> torch.Tensor:
    """Conservative color-histogram scene-cut gate; retain high-motion pairs.

    Threshold operates on the L1 distance of normalized per-channel 16-bin
    color histograms (0..2). It is NOT an optical-flow/motion cut detector.
    """
    if gt.ndim != 5 or gt.shape[2] != 3 or gt.shape[1] < 2:
        raise ValueError("Expected paired GT video [B,T,3,H,W] with T>=2")
    if not 0.0 < cut_hist_threshold <= 2.0:
        raise ValueError("cut_hist_threshold must be in (0,2]")
    b, t, c, h, w = gt.shape
    flat = gt.detach().float().clamp(0, 1).reshape(b * t, c, h, w)
    pooled = F.adaptive_avg_pool2d(flat, (min(h, 32), min(w, 32)))
    bins = (pooled * 15).long()
    # Histogram depends on global color distribution, not positional motion.
    hist = F.one_hot(bins, num_classes=16).float().mean(dim=(2, 3))
    hist = hist.reshape(b, t, c, 16)
    distance = (hist[:, 1:] - hist[:, :-1]).abs().sum(dim=-1).mean(dim=-1)
    return distance <= cut_hist_threshold


def m12_slim_objective(
    prediction: torch.Tensor,
    gt: torch.Tensor,
    original: torch.Tensor,
    *,
    teacher_lpips: nn.Module | None = None,
    teacher_rgb_weight: float = 1.0,
    teacher_hf_weight: float = 0.5,
    teacher_lpips_weight: float = 0.1,
    temporal_weight: float = 0.5,
    lowpass_kernel: int = 7,
    pair_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Keep teacher texture but penalize GT-relative temporal luma error.

    Invalid temporal edges (e.g. detected scene cuts) contribute nothing.
    Static scene and moving scene GT naturally cancel in residual dynamics;
    raw frame-to-frame smoothing is never applied.
    """
    if (prediction.ndim != 5 or prediction.shape != gt.shape
            or prediction.shape != original.shape or prediction.shape[2] != 3
            or prediction.shape[1] < 2):
        raise ValueError("Expected aligned video tensors [B,T,3,H,W], T>=2")
    weights = (teacher_rgb_weight, teacher_hf_weight,
               teacher_lpips_weight, temporal_weight)
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in weights):
        raise ValueError("Loss weights must be finite and non-negative")
    if teacher_rgb_weight + teacher_hf_weight <= 0 or temporal_weight <= 0:
        raise ValueError("Need teacher detail and GT-relative temporal objectives")
    if teacher_lpips_weight and teacher_lpips is None:
        raise ValueError("Positive teacher LPIPS weight requires a local metric")
    if lowpass_kernel < 3 or lowpass_kernel % 2 != 1:
        raise ValueError("lowpass_kernel must be odd and >=3")

    p = prediction.float()
    t = original.detach().float()
    g = gt.detach().float()
    b, frames, channels, h, w = p.shape

    def lowpass(x):
        flat = x.reshape(b * frames, channels, h, w)
        pad = lowpass_kernel // 2
        result = F.avg_pool2d(F.pad(flat, (pad,) * 4, mode="replicate"),
                              kernel_size=lowpass_kernel, stride=1)
        return result.reshape(b, frames, channels, h, w)

    # Same-latent Original ReAE is a texture teacher, not a temporal GT.
    teacher_rgb = torch.sqrt((p - t).square() + 1e-6).mean() - 1e-3
    teacher_hf = ((p - lowpass(p)) - (t - lowpass(t))).abs().mean()
    teacher_perc = p.new_zeros(())
    if teacher_lpips_weight:
        teacher_perc = teacher_lpips(p, t)

    luma_coeff = p.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 1, 3, 1, 1)
    residual_y = ((p - g) * luma_coeff).sum(dim=2, keepdim=True)
    residual_flat = residual_y.reshape(b * frames, 1, h, w)
    pad = lowpass_kernel // 2
    residual_low = F.avg_pool2d(
        F.pad(residual_flat, (pad,) * 4, mode="replicate"),
        kernel_size=lowpass_kernel, stride=1,
    ).reshape(b, frames, 1, h, w)
    differences = (residual_low[:, 1:] - residual_low[:, :-1]).abs()
    if pair_valid is None:
        pair_valid = torch.ones((b, frames - 1), dtype=torch.bool, device=p.device)
    if pair_valid.shape != (b, frames - 1) or pair_valid.dtype != torch.bool:
        raise ValueError("pair_valid must be bool [B,T-1]")
    keep = pair_valid.to(device=p.device, dtype=p.dtype).view(b, frames - 1, 1, 1, 1)
    temporal = (differences * keep).mean(dim=(2, 3, 4)).sum() / keep.sum().clamp_min(1)
    loss = (teacher_rgb_weight * teacher_rgb + teacher_hf_weight * teacher_hf
            + teacher_lpips_weight * teacher_perc + temporal_weight * temporal)
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError("Nonfinite M12 Slim objective")
    return {
        "loss": loss,
        "teacher_rgb_charbonnier": teacher_rgb,
        "teacher_hf_l1": teacher_hf,
        "teacher_lpips": teacher_perc,
        "gt_relative_luma_temporal_l1": temporal,
        "valid_temporal_pair_fraction": pair_valid.float().mean().detach(),
        "gt_rgb_l1": (p - g).abs().mean().detach(),
    }
