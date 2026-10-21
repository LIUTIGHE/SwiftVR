"""M11B-S1-T1B: all-phase Original ReAE distillation objectives.

Variant A: full RGB Original ReAE imitation + GT-relative temporal correction.
Variant B: Original ReAE high-pass imitation + GT low-pass/temporal correction.

Both variants use the SAME frozen M11-B z_SR to render teacher and student.
No phase-conditioned teacher switching or raw-frequency maximization. Teacher
never gets gradients; GT is the source of temporal brightness corrections.
"""
from __future__ import annotations

import math

import torch

from swiftvr.training.m11b_phase_objective import _lowpass, phase_metrics


def t1b_objective(
    prediction: torch.Tensor,
    gt: torch.Tensor,
    original: torch.Tensor,
    *,
    variant: str,
    teacher_weight: float = 1.0,
    lowfreq_gt_weight: float = 1.0,
    temporal_weight: float = 0.5,
    highpass_kernel: int = 7,
    gt_lpips=None,
    gt_lpips_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Full RGB teacher A or frequency-separated B; return identical diagnostics.

    Inputs [B,13,3,H,W]. Uses framewise low/high-frequency spatial separation
    and a GT-relative *temporal derivative of low-frequency luminance error*.
    Natural motion/luminance changes present in GT are thus not suppressed.
    """
    if variant not in ("A", "B"):
        raise ValueError("variant must be A (all-phase RGB) or B (all-phase HF)")
    if (
        prediction.ndim != 5 or prediction.shape != gt.shape
        or prediction.shape != original.shape or prediction.shape[2] != 3
        or prediction.shape[1] != 13
    ):
        raise ValueError("Expected same-geometry RGB video [B,13,3,H,W]")
    weights = (teacher_weight, lowfreq_gt_weight, temporal_weight, gt_lpips_weight)
    if any(not math.isfinite(float(w)) or float(w) < 0 for w in weights):
        raise ValueError("All loss weights must be finite and nonnegative")
    if highpass_kernel < 3 or highpass_kernel % 2 != 1:
        raise ValueError("Highpass kernel must be odd and >=3")
    if gt_lpips_weight > 0 and gt_lpips is None:
        raise ValueError("GT LPIPS requested without locally initialized weights")

    p = prediction.float()
    g = gt.detach().float()
    t = original.detach().float()

    low_p = _lowpass(p, highpass_kernel)
    low_g = _lowpass(g, highpass_kernel)
    low_t = _lowpass(t, highpass_kernel)
    high_p, high_g, high_t = p - low_p, g - low_g, t - low_t

    teacher_rgb_l1 = (p - t).abs().mean()
    teacher_hf_l1 = (high_p - high_t).abs().mean()
    gt_lf_l1 = (low_p - low_g).abs().mean()
    gt_rgb_l1 = (p - g).abs().mean()
    gt_hf_l1 = (high_p - high_g).abs().mean()

    # BT.709 luminance; loss measures consecutive *errors*, not consecutive
    # predicted brightness levels. The teacher's brightness flicker is not GT.
    y_coeff = p.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 1, 3, 1, 1)
    err_low_y = ((low_p - low_g) * y_coeff).sum(dim=2, keepdim=True)
    temporal_gt_lf = (err_low_y[:, 1:] - err_low_y[:, :-1]).abs().mean()

    perceptual_gt = p.new_zeros(())
    if gt_lpips_weight:
        perceptual_gt = gt_lpips(p, g)

    if variant == "A":
        # No separate low-frequency absolute GT anchor; this tests whether
        # ALL-phase teacher RGB imitation alone preserves useful texture.
        loss = teacher_weight * teacher_rgb_l1 + temporal_weight * temporal_gt_lf
    else:
        loss = (
            teacher_weight * teacher_hf_l1
            + lowfreq_gt_weight * gt_lf_l1
            + temporal_weight * temporal_gt_lf
        )
    loss = loss + gt_lpips_weight * perceptual_gt
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("Non-finite T1B loss")

    phases = (torch.arange(p.shape[1], device=p.device) + 3) % 4
    def phase_values(video: torch.Tensor):
        scalar = video.abs().mean(dim=(0, 2, 3, 4))
        return {
            f"phase_{j}_mean": scalar[phases == j].mean()
            for j in range(4)
        }

    pred_energy = phase_values(high_p)
    original_energy = phase_values(high_t)
    hf_disagreement = phase_values(high_p - high_t)
    gt_hf_error = phase_values(high_p - high_g)
    out = {
        "loss": loss,
        "teacher_l1": teacher_rgb_l1,
        "teacher_highfreq_l1": teacher_hf_l1,
        "gt_lowfreq_l1": gt_lf_l1,
        "gt_l1": gt_rgb_l1,
        "gt_highfreq_l1": gt_hf_l1,
        "gt_luma_temporal": temporal_gt_lf,
        "gt_lpips": perceptual_gt,
        **{f"phase_{j}_pred_hf_energy": pred_energy[f"phase_{j}_mean"] for j in range(4)},
        **{f"phase_{j}_original_hf_energy": original_energy[f"phase_{j}_mean"] for j in range(4)},
        **{f"phase_{j}_hf_teacher_l1": hf_disagreement[f"phase_{j}_mean"] for j in range(4)},
        **{f"phase_{j}_hf_gt_l1": gt_hf_error[f"phase_{j}_mean"] for j in range(4)},
        **phase_metrics(p, g, trim=3),
    }
    return out
