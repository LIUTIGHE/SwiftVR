"""Phase-aware, GT-relative decoder objectives for M11B-S1-T1.

No temporal smoothing of raw predictions: reference each video frame to its own
HR target. The teacher image is an auxiliary appearance reference, not a
time-stability reference (Original ReAE itself shows two-phase luminance drift).
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def _validate(pred: torch.Tensor, gt: torch.Tensor, teacher: torch.Tensor) -> None:
    if pred.ndim != 5 or pred.shape != gt.shape or pred.shape != teacher.shape:
        raise ValueError("Expected aligned [B,T,3,H,W] prediction/GT/teacher")
    if pred.shape[2] != 3 or pred.shape[1] < 5:
        raise ValueError("Need RGB video of at least five frames")


def _lowpass(video: torch.Tensor, kernel: int = 7) -> torch.Tensor:
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("Lowpass kernel must be odd and positive")
    b, t, c, h, w = video.shape
    images = video.reshape(b*t, c, h, w)
    if kernel == 1:
        return video
    # Replicate-pad avoids spurious low-frequency errors at ROI/image borders.
    result = F.avg_pool2d(
        F.pad(images, (kernel//2,)*4, mode="replicate"),
        kernel_size=kernel, stride=1,
    )
    return result.reshape(b,t,c,h,w)


def phase_metrics(pred: torch.Tensor, gt: torch.Tensor, *, trim: int = 3) -> dict[str,torch.Tensor]:
    """GT-relative framewise and groupwise measurements, not cross-frame HF equalization."""
    if pred.shape != gt.shape or pred.ndim != 5:
        raise ValueError("pred and GT must have same [B,T,C,H,W] shape")
    diff = (pred.float() - gt.float()).square().mean(dim=(0,2,3,4))
    labels = (torch.arange(pred.shape[1], device=diff.device) + trim) % 4
    return {f"phase_{p}_mse": diff[labels == p].mean() for p in range(4) if bool((labels==p).any())}


def phase_objective(
    pred: torch.Tensor,
    gt: torch.Tensor,
    teacher: torch.Tensor,
    *,
    lpips=None,
    weight_gt: float = 1.0,
    weight_teacher: float = 0.25,
    weight_highfreq: float = 0.1,
    weight_luma_temporal: float = 0.5,
    weight_phase_balance: float = 0.1,
    weight_lpips: float = 0.1,
    lowpass_kernel: int = 7,
    phase_temperature: float = 0.01,
    phase_trim: int = 3,
) -> dict[str,torch.Tensor]:
    _validate(pred,gt,teacher)
    weights=(weight_gt,weight_teacher,weight_highfreq,weight_luma_temporal,weight_phase_balance,weight_lpips)
    if any(not math.isfinite(float(v)) or float(v)<0 for v in weights):
        raise ValueError("All weights must be finite nonnegative")
    if phase_temperature <= 0 or not math.isfinite(phase_temperature):
        raise ValueError("phase_temperature must be finite positive")
    if weight_lpips and lpips is None:
        raise ValueError("Nonzero LPIPS requires locally initialized LPIPS network")

    p, g, t = pred.float(), gt.detach().float(), teacher.detach().float()
    gt_l1 = (p-g).abs().mean()
    teacher_l1 = (p-t).abs().mean()

    # Penalize GT-relative high-pass residual, NOT raw per-phase energy.
    detail = p - _lowpass(p, lowpass_kernel)
    gt_detail = g - _lowpass(g, lowpass_kernel)
    highfreq = (detail-gt_detail).abs().mean()

    # GT-relative low-frequency residual: actual scene motion/exposure is removed.
    luminance = torch.tensor((0.2126,0.7152,0.0722),device=p.device,dtype=p.dtype).view(1,1,3,1,1)
    residual_y = ((p-g)*luminance).sum(dim=2,keepdim=True)
    residual_lf = _lowpass(residual_y,lowpass_kernel)
    # Loss is on residual change, not prediction frame difference.
    luma_temporal = (residual_lf[:,1:] - residual_lf[:,:-1]).abs().mean()

    per_frame_hf_error = (detail-gt_detail).abs().mean(dim=(0,2,3,4))
    phases=(torch.arange(p.shape[1],device=p.device)+phase_trim)%4
    by_phase=[per_frame_hf_error[phases==k].mean() for k in range(4)]
    phase_values=torch.stack(by_phase)
    # Smooth max minus mean, >=0, zero if all phases are equal.
    phase_balance = (phase_temperature*
                     torch.logsumexp(phase_values/phase_temperature,dim=0)
                     -phase_temperature*math.log(4)
                     -phase_values.mean()).clamp_min(0)

    lp = p.new_zeros(())
    if weight_lpips:
        lp=lpips(p,g)
    loss=(weight_gt*gt_l1+weight_teacher*teacher_l1+
          weight_highfreq*highfreq+weight_luma_temporal*luma_temporal+
          weight_phase_balance*phase_balance+weight_lpips*lp)
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("Non-finite phase-aware decoder loss")
    return {"loss":loss,"gt_l1":gt_l1,"teacher_l1":teacher_l1,
            "highfreq_gt_l1":highfreq,"luma_temporal_gt":luma_temporal,
            "phase_highfreq_imbalance":phase_balance,"gt_lpips":lp,
            "phase_0_hf_l1":phase_values[0],"phase_1_hf_l1":phase_values[1],
            "phase_2_hf_l1":phase_values[2],"phase_3_hf_l1":phase_values[3],
            "phase_0_minus_3_hf_l1":phase_values[0]-phase_values[3],
            **phase_metrics(p,g,trim=phase_trim)}
