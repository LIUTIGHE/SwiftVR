"""Trust-region M8-C co-adaptation with explicit responsibility separation.

The compressed Transformer is responsible for approaching the Stage-A latent/
RGB behaviour. Decoder76 is responsible for remaining a faithful decoder for the
*current* M8 latent distribution. This prevents the small decoder from absorbing
representation error that belongs to the Transformer.

Gradient routing:
  * representation + Stage-A RGB losses -> Transformer;
  * local full-ReAE decoder imitation -> Decoder76;
  * Stage-A RGB is rendered through a frozen Decoder76 anchor copy;
  * Decoder76 local imitation sees z_student.detach().

GT is diagnostic only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .b2b_moe_training import forward_moe_transformer_training
from .forward import (
    encode_reae_clip,
    prepare_prompt_free_no_time_transformer_for_training,
    prepare_training_batch,
)
from .m8_joint import (
    cosine_loss,
    latent_spatial_detail_loss,
    latent_temporal_detail_loss,
    normalized_mse,
)
from .stage3 import temporal_difference_mse
from .tiny_decoder import LPIPSAlexLoss


@dataclass(frozen=True)
class M8TrustLossWeights:
    velocity_nmse: float = 0.25
    velocity_cosine: float = 0.25
    latent_spatial: float = 0.5
    latent_temporal: float = 0.5
    stagea_rgb_l1: float = 1.0
    stagea_lpips: float = 0.1
    stagea_rgb_temporal: float = 1.0
    decoder_l2: float = 10.0
    decoder_lpips: float = 0.1
    decoder_temporal: float = 1.0
    router_balance: float = 0.01


class M8TrustRegionForward(nn.Module):
    """Frozen ReAE + M8 Transformer + trainable Decoder76 + frozen anchor Decoder76."""

    def __init__(
        self,
        reae: nn.Module,
        transformer: nn.Module,
        decoder: nn.Module,
        anchor_decoder: nn.Module,
        *,
        attention_backend: str = "sdpa",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.reae = reae
        self.transformer = transformer
        self.decoder = decoder
        self.anchor_decoder = anchor_decoder
        self.gradient_checkpointing = bool(gradient_checkpointing)
        for parameter in self.reae.parameters():
            parameter.requires_grad_(False)
        for parameter in self.anchor_decoder.parameters():
            parameter.requires_grad_(False)
        self.reae.eval()
        self.anchor_decoder.eval()
        prepare_prompt_free_no_time_transformer_for_training(
            self.transformer,
            attention_backend=attention_backend,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.reae.eval()
        self.anchor_decoder.eval()
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
        decoder_input = z_student.permute(0, 2, 1, 3, 4).contiguous()

        # Decoder-local path: do not let decoder imitation alter Transformer.
        prediction = self.decoder(
            decoder_input.detach(),
            output_frames=int(target.shape[1]),
            clamp=False,
        )

        # Transformer final-RGB path: frozen anchor decoder transmits gradients
        # only to z_student/Transformer, never to Decoder76 parameters.
        transformer_prediction = self.anchor_decoder(
            decoder_input,
            output_frames=int(target.shape[1]),
            clamp=False,
        )

        if prediction.shape != target.shape or transformer_prediction.shape != target.shape:
            raise ValueError(
                "M8 trust-region prediction shape mismatch: "
                f"decoder={tuple(prediction.shape)} anchor={tuple(transformer_prediction.shape)} "
                f"target={tuple(target.shape)}"
            )
        return {
            "velocity": velocity,
            "router_balance_loss": router_balance,
            "z_lq": z_lq,
            "z_student": z_student,
            "prediction": prediction,
            "transformer_prediction": transformer_prediction,
            "target": target,
            "lq_input": lq_input,
        }


def m8_trust_objective(
    *,
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    z_lq: torch.Tensor,
    decoder_prediction: torch.Tensor,
    decoder_teacher_prediction: torch.Tensor,
    transformer_prediction: torch.Tensor,
    stagea_teacher_prediction: torch.Tensor,
    router_balance_loss: torch.Tensor,
    perceptual: LPIPSAlexLoss | None,
    weights: M8TrustLossWeights,
    lpips_microbatch_frames: int = 16,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    teacher_velocity = teacher_velocity.detach()
    stagea_teacher_prediction = stagea_teacher_prediction.detach()
    decoder_teacher_prediction = decoder_teacher_prediction.detach()
    z_teacher = (z_lq.detach() - teacher_velocity).detach()
    z_student = z_lq - student_velocity

    velocity_nmse = normalized_mse(student_velocity, teacher_velocity, epsilon)
    velocity_cosine_loss = cosine_loss(student_velocity, teacher_velocity, epsilon)
    latent_spatial = latent_spatial_detail_loss(z_student, z_teacher, epsilon)
    latent_temporal = latent_temporal_detail_loss(z_student, z_teacher, epsilon)

    stagea_rgb_l1 = F.l1_loss(
        transformer_prediction.float(), stagea_teacher_prediction.float()
    )
    stagea_rgb_temporal = temporal_difference_mse(
        transformer_prediction.float(), stagea_teacher_prediction.float()
    )
    decoder_l2 = F.mse_loss(
        decoder_prediction.float(), decoder_teacher_prediction.float()
    )
    decoder_temporal = temporal_difference_mse(
        decoder_prediction.float(), decoder_teacher_prediction.float()
    )

    zero = stagea_rgb_l1.new_zeros(())
    stagea_lpips = zero
    decoder_lpips = zero
    if float(weights.stagea_lpips) > 0 or float(weights.decoder_lpips) > 0:
        if perceptual is None:
            raise ValueError("positive LPIPS weights require LPIPSAlexLoss")
        if float(weights.stagea_lpips) > 0:
            stagea_lpips = perceptual.forward_video(
                transformer_prediction,
                stagea_teacher_prediction,
                microbatch_frames=int(lpips_microbatch_frames),
            )
        if float(weights.decoder_lpips) > 0:
            decoder_lpips = perceptual.forward_video(
                decoder_prediction,
                decoder_teacher_prediction,
                microbatch_frames=int(lpips_microbatch_frames),
            )

    loss = (
        float(weights.velocity_nmse) * velocity_nmse
        + float(weights.velocity_cosine) * velocity_cosine_loss
        + float(weights.latent_spatial) * latent_spatial
        + float(weights.latent_temporal) * latent_temporal
        + float(weights.stagea_rgb_l1) * stagea_rgb_l1
        + float(weights.stagea_lpips) * stagea_lpips
        + float(weights.stagea_rgb_temporal) * stagea_rgb_temporal
        + float(weights.decoder_l2) * decoder_l2
        + float(weights.decoder_lpips) * decoder_lpips
        + float(weights.decoder_temporal) * decoder_temporal
        + float(weights.router_balance) * router_balance_loss
    )
    return {
        "loss": loss,
        "velocity_nmse": velocity_nmse,
        "velocity_cosine_loss": velocity_cosine_loss,
        "latent_spatial_detail": latent_spatial,
        "latent_temporal_detail": latent_temporal,
        "stagea_rgb_l1": stagea_rgb_l1,
        "stagea_lpips": stagea_lpips,
        "stagea_rgb_temporal_mse": stagea_rgb_temporal,
        "decoder_teacher_l2": decoder_l2,
        "decoder_teacher_lpips": decoder_lpips,
        "decoder_teacher_temporal_mse": decoder_temporal,
        "router_balance_loss": router_balance_loss,
    }
