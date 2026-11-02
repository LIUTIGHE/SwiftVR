"""M10: Transformer-only teacher recovery through a frozen M9-A1 decoder.

The deployment architecture is unchanged. High-pass filtering and its temporal
first difference are training/diagnostic operators, not inference modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

STAGE_A_CACHE_KIND = "swiftvr_b2a_stage_a_teacher_velocity"


@dataclass(frozen=True)
class RecoveryWeights:
    velocity_nmse: float = 0.25
    velocity_cosine_loss: float = 0.25
    rgb_l1: float = 1.0
    hf_l1: float = 1.0
    hf_temporal_l1: float = 0.0
    router_balance: float = 0.01

    def __post_init__(self) -> None:
        values = vars(self)
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("M10 loss weights must be finite and non-negative")
        if not any(v > 0 for k, v in values.items() if k != "router_balance"):
            raise ValueError("At least one teacher-matching loss must be active")


def validate_teacher_metadata(
    train: Mapping[str, object],
    val: Mapping[str, object],
    *,
    reae_sha256: str,
) -> None:
    """Reject TA/M8 endpoint caches and mismatched Stage-A teacher lineages."""
    for label, meta in (("train", train), ("val", val)):
        if meta.get("kind") != STAGE_A_CACHE_KIND:
            raise ValueError(
                f"{label}: M10 requires a Stage-A D3072 velocity cache, "
                f"not {meta.get('kind')!r}; do not use the D1536 TA or M8-A z_SR cache"
            )
        if int(meta.get("teacher_delta_step", -1)) != 200000:
            raise ValueError(f"{label}: expected the locked Stage-A step200000 teacher")
        if meta.get("reae_sha256") != reae_sha256:
            raise ValueError(f"{label}: cached teacher used different ReAE weights")
        if not meta.get("teacher_delta_weights_sha256"):
            raise ValueError(f"{label}: missing Stage-A teacher weight fingerprint")
    if train["teacher_delta_weights_sha256"] != val["teacher_delta_weights_sha256"]:
        raise ValueError("Train/validation Stage-A teacher weight fingerprints differ")


class GaussianHighPass(nn.Module):
    """Per-frame, per-channel 5x5 Gaussian residual; never mixes time/batch."""

    def __init__(self) -> None:
        super().__init__()
        coordinates = torch.arange(-2, 3, dtype=torch.float32)
        kernel = torch.exp(-0.5 * coordinates.square())  # sigma = 1 pixel
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", (kernel[:, None] * kernel[None, :])[None, None])

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or min(video.shape[-2:]) < 3:
            raise ValueError("High-pass input must be [B,T,C,H,W] with H,W >= 3")
        b, t, c, h, w = video.shape
        # Keep differences/reductions in FP32, including under BF16 autocast.
        with torch.autocast(device_type=video.device.type, enabled=False):
            pixels = video.float().reshape(b * t, c, h, w)
            kernel = self.kernel.to(device=pixels.device, dtype=torch.float32).expand(c, 1, 5, 5)
            low = F.conv2d(F.pad(pixels, (2, 2, 2, 2), mode="reflect"), kernel, groups=c)
            return (pixels - low).reshape(b, t, c, h, w)


def high_frequency_terms(
    student: torch.Tensor, teacher: torch.Tensor, high_pass: GaussianHighPass
) -> dict[str, torch.Tensor]:
    """RGB, HF, and teacher-relative HF first-difference errors.

    This matches temporal changes; it does NOT minimize the student's temporal
    change towards zero. No optical-flow/correspondence claim is made.
    """
    if student.shape != teacher.shape or student.ndim != 5 or student.shape[2] != 3:
        raise ValueError("Student/teacher must have identical [B,T,3,H,W] shapes")
    student_f, teacher_f = student.float(), teacher.detach().float()
    hs, ht = high_pass(student_f), high_pass(teacher_f)
    error = hs - ht
    temporal = (
        (error[:, 1:] - error[:, :-1]).abs().mean()
        if student.shape[1] > 1 else error.sum() * 0.0
    )
    return {
        "rgb_l1": F.l1_loss(student_f, teacher_f),
        "hf_l1": error.abs().mean(),
        "hf_temporal_l1": temporal,
    }


def recovery_objective(
    output: Mapping[str, torch.Tensor],
    teacher_velocity: torch.Tensor,
    *,
    weights: RecoveryWeights,
    high_pass: GaussianHighPass,
) -> dict[str, torch.Tensor]:
    """Teacher-only objective. GT is intentionally not accepted by this API."""
    sv, tv = output["velocity"].float(), teacher_velocity.detach().float()
    if sv.shape != tv.shape:
        raise ValueError(f"Velocity shape mismatch: {sv.shape} vs {tv.shape}")
    nmse = F.mse_loss(sv, tv) / tv.square().mean().clamp_min(1e-8)
    cosine = 1.0 - F.cosine_similarity(sv.flatten(1), tv.flatten(1), dim=1, eps=1e-8).mean()
    terms = {
        "velocity_nmse": nmse,
        "velocity_cosine_loss": cosine,
        **high_frequency_terms(output["prediction"], output["teacher_prediction"], high_pass),
        "router_balance": output["router_balance_loss"],
    }
    weighted = {f"weighted_{key}": value * getattr(weights, key) for key, value in terms.items()}
    # Keep the selection score identical for spatial and spatiotemporal gates.
    # It is a reproducible ranking aid, NOT a visual-quality pass/fail test.
    score = terms["rgb_l1"] + terms["hf_l1"] + terms["hf_temporal_l1"]
    return {**terms, **weighted, "loss": sum(weighted.values()), "teacher_selection_score": score}


def decode_student(
    decoder: nn.Module,
    z_lq: torch.Tensor,
    velocity: torch.Tensor,
    *,
    output_frames: int,
    checkpointing: bool = True,
) -> torch.Tensor:
    """Keep dRGB/dz through the frozen decoder; z/v use [B,C,T,H,W]."""
    if z_lq.ndim != 5 or z_lq.shape != velocity.shape:
        raise ValueError("z_lq and velocity must have identical [B,C,T,H,W] shapes")
    endpoint = (z_lq - velocity).permute(0, 2, 1, 3, 4).contiguous()

    def decode(z):
        return decoder(z, output_frames=output_frames, clamp=False)

    if checkpointing and torch.is_grad_enabled() and endpoint.requires_grad:
        return checkpoint(decode, endpoint, use_reentrant=False)
    return decode(endpoint)


class M10BackboneRecoveryForward(nn.Module):
    """Frozen E + trainable M8-A + frozen A1; frozen Original D teaches RGB."""

    def __init__(self, reae: nn.Module, transformer: nn.Module, decoder: nn.Module,
                 *, attention_backend: str = "sdpa") -> None:
        super().__init__()
        from .forward import prepare_prompt_free_no_time_transformer_for_training

        self.reae, self.transformer, self.decoder = reae, transformer, decoder
        self.reae.requires_grad_(False).eval()
        self.decoder.requires_grad_(False).eval()
        self.transformer.requires_grad_(True)
        prepare_prompt_free_no_time_transformer_for_training(
            self.transformer, attention_backend=attention_backend
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.reae.eval()
        self.decoder.eval()
        return self

    def forward(self, batch, teacher_velocity: torch.Tensor, *, include_original: bool = False):
        from .forward import decode_reae_clip, encode_reae_clip, prepare_training_batch
        from .b2b_moe_training import forward_moe_transformer_training

        prepared = prepare_training_batch(batch)
        target = prepared["target"]
        frames = int(target.shape[1])
        with torch.no_grad():
            z = encode_reae_clip(self.reae, prepared["lq_input"], require_4k_plus_1=True)
            z_lq = z.permute(0, 2, 1, 3, 4).contiguous()
            if teacher_velocity.shape != z_lq.shape:
                raise ValueError("Stage-A cache shape does not match this encoded view")
            teacher_rgb = decode_reae_clip(
                self.reae, (z_lq - teacher_velocity).permute(0, 2, 1, 3, 4).contiguous(),
                output_frames=frames, clamp=False,
            )
        velocity, balance = forward_moe_transformer_training(
            self.transformer, z_lq, gradient_checkpointing=True
        )
        prediction = decode_student(self.decoder, z_lq, velocity, output_frames=frames)
        result = {
            "velocity": velocity, "router_balance_loss": balance,
            "prediction": prediction, "teacher_prediction": teacher_rgb,
            "target": target.detach(), "lq_input": prepared["lq_input"].detach(),
        }
        if include_original:
            with torch.no_grad():
                result["student_original_prediction"] = decode_reae_clip(
                    self.reae, (z_lq - velocity).permute(0, 2, 1, 3, 4).contiguous(),
                    output_frames=frames, clamp=False,
                )
        return result
