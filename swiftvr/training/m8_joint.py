"""Decoder-aware joint co-adaptation for the M8 SwiftVR system.

M8 couples a D1024/L20 sparse-MoE Transformer with the structured
(128,96,64,64) ReAE Decoder76. The frozen ReAE encoder supplies z_LQ. Stage-A
D3072 velocity remains the final-behavior representation anchor; the original
frozen ReAE decoder renders both the current student latent and the Stage-A
latent for decoder-aware Transformer recovery.

The compact decoder is deliberately optimized on a same-latent teacher task:
Decoder76(z_student.detach()) matches OriginalDecoder(z_student.detach()). This
prevents the compact decoder from being forced to compensate for the remaining
Transformer latent gap. RGB system losses use the frozen original decoder and
therefore update the Transformer through z_student without updating Decoder76.

GT is never part of the optimization objective in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .b2b_moe_training import forward_moe_transformer_training
from .forward import encode_reae_clip, prepare_prompt_free_no_time_transformer_for_training, prepare_training_batch
from .tiny_decoder import LPIPSAlexLoss
from .stage3 import temporal_difference_mse


M8_DECODER_CHANNELS = (128, 96, 64, 64)


@dataclass(frozen=True)
class M8JointLossWeights:
    """Legacy coupled M8-C objective weights retained for compatibility/tests."""

    velocity_nmse: float = 0.25
    velocity_cosine: float = 0.25
    latent_spatial: float = 0.5
    latent_temporal: float = 0.5
    teacher_rgb_l1: float = 1.0
    teacher_lpips: float = 0.1
    teacher_rgb_temporal: float = 1.0
    router_balance: float = 0.01


@dataclass(frozen=True)
class M8JointDecoupledLossWeights:
    """Formal M8-C V2 weights with explicit Transformer/decoder responsibilities."""

    velocity_nmse: float = 0.25
    velocity_cosine: float = 0.25
    latent_spatial: float = 0.5
    latent_temporal: float = 0.5

    # Frozen-original-decoder system loss: gradients go to Transformer via latent.
    system_rgb_l1: float = 1.0
    system_lpips: float = 0.1
    system_rgb_temporal: float = 1.0

    # Same-latent compact-decoder recovery: gradients go to Decoder76 only.
    decoder_teacher_l2: float = 10.0
    decoder_teacher_lpips: float = 0.1
    decoder_teacher_temporal: float = 1.0

    router_balance: float = 0.01


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    pred = prediction.float()
    ref = target.detach().float()
    mse = (pred - ref).square().mean()
    power = ref.square().mean().clamp_min(float(epsilon))
    return mse / power


def cosine_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    pred = prediction.float().flatten(1)
    ref = target.detach().float().flatten(1)
    cosine = F.cosine_similarity(pred, ref, dim=1, eps=float(epsilon)).mean()
    return 1.0 - cosine


def latent_spatial_detail_loss(student: torch.Tensor, teacher: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalized finite-difference loss on latent H/W gradients."""
    s = student.float()
    t = teacher.detach().float()
    s_dx = s[..., :, 1:] - s[..., :, :-1]
    t_dx = t[..., :, 1:] - t[..., :, :-1]
    s_dy = s[..., 1:, :] - s[..., :-1, :]
    t_dy = t[..., 1:, :] - t[..., :-1, :]
    numerator = (s_dx - t_dx).square().mean() + (s_dy - t_dy).square().mean()
    denominator = t_dx.square().mean() + t_dy.square().mean()
    return numerator / denominator.clamp_min(float(epsilon))


def latent_temporal_detail_loss(student: torch.Tensor, teacher: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalized finite-difference loss on latent frame changes [B,C,F,H,W]."""
    if student.shape[2] < 2:
        return student.float().new_zeros(())
    s = student.float()
    t = teacher.detach().float()
    s_dt = s[:, :, 1:] - s[:, :, :-1]
    t_dt = t[:, :, 1:] - t[:, :, :-1]
    return (s_dt - t_dt).square().mean() / t_dt.square().mean().clamp_min(float(epsilon))


class M8JointForward(nn.Module):
    """Frozen ReAE encoder + M8 sparse-MoE Transformer + Decoder76.

    Decoder76 receives a detached student latent during M8-C V2. This is
    intentional: compact-decoder losses update Decoder76 only. Transformer RGB
    gradients are supplied separately through the frozen original ReAE decoder in
    the trainer.
    """

    def __init__(
        self,
        reae: nn.Module,
        transformer: nn.Module,
        decoder: nn.Module,
        *,
        attention_backend: str = "sdpa",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.reae = reae
        self.transformer = transformer
        self.decoder = decoder
        self.gradient_checkpointing = bool(gradient_checkpointing)
        for parameter in self.reae.parameters():
            parameter.requires_grad_(False)
        self.reae.eval()
        prepare_prompt_free_no_time_transformer_for_training(
            self.transformer,
            attention_backend=attention_backend,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.reae.eval()
        return self

    def forward(self, batch) -> dict[str, torch.Tensor]:
        prepared = prepare_training_batch(batch)
        lq_input = prepared["lq_input"]
        target = prepared["target"]
        if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("Prepared batch is missing lq_input/target tensors")
        with torch.no_grad():
            z_lq_ntchw = encode_reae_clip(self.reae, lq_input, require_4k_plus_1=True)
        z_lq = z_lq_ntchw.permute(0, 2, 1, 3, 4).contiguous().detach()
        velocity, router_balance = forward_moe_transformer_training(
            self.transformer,
            z_lq,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        if velocity.shape != z_lq.shape:
            raise ValueError(f"M8 velocity {tuple(velocity.shape)} != z_lq {tuple(z_lq.shape)}")
        z_student = z_lq - velocity

        # Decoder76 recovery is intentionally same-latent and decoder-only. The
        # trainer separately decodes non-detached z_student with the frozen full
        # ReAE decoder to provide RGB gradients to the Transformer.
        prediction = self.decoder(
            z_student.detach().permute(0, 2, 1, 3, 4).contiguous(),
            output_frames=int(target.shape[1]),
            clamp=False,
        )
        if prediction.shape != target.shape:
            raise ValueError(f"M8 prediction {tuple(prediction.shape)} != target {tuple(target.shape)}")
        return {
            "velocity": velocity,
            "router_balance_loss": router_balance,
            "z_lq": z_lq,
            "z_student": z_student,
            "prediction": prediction,
            "target": target,
            "lq_input": lq_input,
        }


def m8_joint_objective(
    *,
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    z_lq: torch.Tensor,
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    router_balance_loss: torch.Tensor,
    perceptual: LPIPSAlexLoss | None,
    weights: M8JointLossWeights,
    lpips_microbatch_frames: int = 16,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Legacy coupled M8-C objective retained for backwards compatibility."""
    teacher_velocity = teacher_velocity.detach()
    teacher_prediction = teacher_prediction.detach()
    z_teacher = (z_lq.detach() - teacher_velocity).detach()
    z_student = z_lq - student_velocity

    velocity_nmse = normalized_mse(student_velocity, teacher_velocity, epsilon)
    velocity_cosine_loss = cosine_loss(student_velocity, teacher_velocity, epsilon)
    latent_spatial = latent_spatial_detail_loss(z_student, z_teacher, epsilon)
    latent_temporal = latent_temporal_detail_loss(z_student, z_teacher, epsilon)
    rgb_l1 = F.l1_loss(student_prediction.float(), teacher_prediction.float())
    rgb_temporal = temporal_difference_mse(student_prediction.float(), teacher_prediction.float())
    lpips = rgb_l1.new_zeros(())
    if float(weights.teacher_lpips) > 0:
        if perceptual is None:
            raise ValueError("positive teacher_lpips weight requires LPIPSAlexLoss")
        lpips = perceptual.forward_video(
            student_prediction,
            teacher_prediction,
            microbatch_frames=int(lpips_microbatch_frames),
        )

    loss = (
        float(weights.velocity_nmse) * velocity_nmse
        + float(weights.velocity_cosine) * velocity_cosine_loss
        + float(weights.latent_spatial) * latent_spatial
        + float(weights.latent_temporal) * latent_temporal
        + float(weights.teacher_rgb_l1) * rgb_l1
        + float(weights.teacher_lpips) * lpips
        + float(weights.teacher_rgb_temporal) * rgb_temporal
        + float(weights.router_balance) * router_balance_loss
    )
    return {
        "loss": loss,
        "velocity_nmse": velocity_nmse,
        "velocity_cosine_loss": velocity_cosine_loss,
        "latent_spatial_detail": latent_spatial,
        "latent_temporal_detail": latent_temporal,
        "teacher_rgb_l1": rgb_l1,
        "teacher_lpips": lpips,
        "teacher_rgb_temporal_mse": rgb_temporal,
        "router_balance_loss": router_balance_loss,
    }


def m8_joint_decoupled_objective(
    *,
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    z_lq: torch.Tensor,
    compact_prediction: torch.Tensor,
    full_student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    router_balance_loss: torch.Tensor,
    perceptual: LPIPSAlexLoss | None,
    weights: M8JointDecoupledLossWeights,
    lpips_microbatch_frames: int = 16,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """M8-C V2 objective with explicit gradient-role separation.

    ``full_student_prediction`` must be OriginalDecoder(z_student) with frozen
    original decoder parameters and a non-detached z_student, so the system RGB
    terms update only the Transformer. ``compact_prediction`` is produced from
    z_student.detach() by M8JointForward, so the same-latent decoder terms update
    only Decoder76.
    """

    teacher_velocity = teacher_velocity.detach()
    teacher_prediction = teacher_prediction.detach()
    decoder_teacher = full_student_prediction.detach()
    z_teacher = (z_lq.detach() - teacher_velocity).detach()
    z_student = z_lq - student_velocity

    velocity_nmse = normalized_mse(student_velocity, teacher_velocity, epsilon)
    velocity_cosine_loss = cosine_loss(student_velocity, teacher_velocity, epsilon)
    latent_spatial = latent_spatial_detail_loss(z_student, z_teacher, epsilon)
    latent_temporal = latent_temporal_detail_loss(z_student, z_teacher, epsilon)

    system_rgb_l1 = F.l1_loss(full_student_prediction.float(), teacher_prediction.float())
    system_rgb_temporal = temporal_difference_mse(
        full_student_prediction.float(), teacher_prediction.float()
    )
    system_lpips = system_rgb_l1.new_zeros(())
    if float(weights.system_lpips) > 0:
        if perceptual is None:
            raise ValueError("positive system_lpips weight requires LPIPSAlexLoss")
        system_lpips = perceptual.forward_video(
            full_student_prediction,
            teacher_prediction,
            microbatch_frames=int(lpips_microbatch_frames),
        )

    decoder_teacher_l2 = F.mse_loss(compact_prediction.float(), decoder_teacher.float())
    decoder_teacher_temporal = temporal_difference_mse(
        compact_prediction.float(), decoder_teacher.float()
    )
    decoder_teacher_lpips = decoder_teacher_l2.new_zeros(())
    if float(weights.decoder_teacher_lpips) > 0:
        if perceptual is None:
            raise ValueError("positive decoder_teacher_lpips weight requires LPIPSAlexLoss")
        decoder_teacher_lpips = perceptual.forward_video(
            compact_prediction,
            decoder_teacher,
            microbatch_frames=int(lpips_microbatch_frames),
        )

    loss = (
        float(weights.velocity_nmse) * velocity_nmse
        + float(weights.velocity_cosine) * velocity_cosine_loss
        + float(weights.latent_spatial) * latent_spatial
        + float(weights.latent_temporal) * latent_temporal
        + float(weights.system_rgb_l1) * system_rgb_l1
        + float(weights.system_lpips) * system_lpips
        + float(weights.system_rgb_temporal) * system_rgb_temporal
        + float(weights.decoder_teacher_l2) * decoder_teacher_l2
        + float(weights.decoder_teacher_lpips) * decoder_teacher_lpips
        + float(weights.decoder_teacher_temporal) * decoder_teacher_temporal
        + float(weights.router_balance) * router_balance_loss
    )

    return {
        "loss": loss,
        "velocity_nmse": velocity_nmse,
        "velocity_cosine_loss": velocity_cosine_loss,
        "latent_spatial_detail": latent_spatial,
        "latent_temporal_detail": latent_temporal,
        "system_rgb_l1": system_rgb_l1,
        "system_lpips": system_lpips,
        "system_rgb_temporal_mse": system_rgb_temporal,
        "decoder_teacher_l2": decoder_teacher_l2,
        "decoder_teacher_lpips": decoder_teacher_lpips,
        "decoder_teacher_temporal_mse": decoder_teacher_temporal,
        "router_balance_loss": router_balance_loss,
    }
