"""Joint representation/rendering adaptation for the extreme B2B SwiftVR branch.

B2B keeps the frozen Stage-A ReAE encoder and trains two aggressively compressed
modules together:

    LQ -> frozen ReAE encoder -> tiny prompt-free/no-time DiT -> adapted latent
       -> extreme SlimReAE decoder -> RGB

The first B2B target deliberately preserves all 30 DiT blocks and 128-D attention
heads while shrinking width hard enough to fit the 210-GMAC DiT+decoder budget:

    hidden=768, heads=6x128, FFN=4080, depth=30, adapter=128
    decoder=(96,48,24,16)

The Stage-A velocity remains a *weak representation anchor* rather than a hard
latent target.  Teacher-RGB and GT-RGB terms are therefore first-class losses so
that the tiny DiT may learn a decoder-friendly latent representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .b2a_width import B2AWidthSpec, forward_b2a_compact_transformer_training
from .distillation import velocity_distillation_objective
from .forward import (
    encode_reae_clip,
    prepare_prompt_free_no_time_transformer_for_training,
    prepare_training_batch,
)


B2B_TINY_SPEC = B2AWidthSpec(
    hidden_dim=768,
    num_heads=6,
    ffn_dim=4080,
    head_dim=128,
    num_layers=30,
    adapter_dim=128,
)
B2B_EXTREME_DECODER_CHANNELS = (96, 48, 24, 16)
B2B_EXTREME_DECODER_GMAC_PER_FRAME = 13.35785472

# Canonical 1920x1088 streaming-middle reference decomposition for the already
# validated D=1536/F=8960 B2-A Transformer.  Scaling follows the exact operation
# dimensions: patch/proj, attention matmuls and adapters are linear in D; QKV/out
# projections are quadratic in D; FFNs are proportional to D*F.
_REF_D = 1536
_REF_F = 8960
_REF_ADAPTER = 128
_REF_PATCH_AND_OUT_GMAC = 0.30081024
_REF_QKV_AND_ATTN_OUT_GMAC = 144.3889152
_REF_QK_AND_AV_GMAC = 122.30590464
_REF_ADAPTER_GMAC = 6.0162048
_REF_FFN_GMAC = 421.134336


def estimate_b2b_dit_gmac_per_output_frame(spec: B2AWidthSpec = B2B_TINY_SPEC) -> float:
    """Analytical canonical streaming-middle DiT GMAC/output-frame."""

    d = float(spec.hidden_dim)
    f = float(spec.ffn_dim)
    adapter = float(spec.adapter_dim)
    depth_scale = float(spec.num_layers) / 30.0
    d_scale = d / _REF_D
    return depth_scale * (
        _REF_PATCH_AND_OUT_GMAC * d_scale
        + _REF_QKV_AND_ATTN_OUT_GMAC * d_scale * d_scale
        + _REF_QK_AND_AV_GMAC * d_scale
        + _REF_ADAPTER_GMAC * d_scale * (adapter / _REF_ADAPTER)
        + _REF_FFN_GMAC * d_scale * (f / _REF_F)
    )


def b2b_compute_budget(spec: B2AWidthSpec = B2B_TINY_SPEC) -> dict[str, float]:
    dit = estimate_b2b_dit_gmac_per_output_frame(spec)
    decoder = float(B2B_EXTREME_DECODER_GMAC_PER_FRAME)
    combined = dit + decoder
    return {
        "dit_gmac_per_frame": dit,
        "decoder_gmac_per_frame": decoder,
        "dit_plus_decoder_gmac_per_frame": combined,
        "headroom_to_210_gmac": 210.0 - combined,
    }


class B2BJointForward(nn.Module):
    """Frozen ReAE encoder + trainable tiny DiT + trainable extreme decoder."""

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
        # The encoder is permanently frozen/eval even while the two compact
        # student modules train jointly.
        self.reae.eval()
        return self

    def forward(self, batch) -> dict[str, torch.Tensor]:
        prepared = prepare_training_batch(batch)
        lq_input = prepared["lq_input"]
        target = prepared["target"]
        if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("Prepared batch is missing lq_input/target tensors")

        # No gradients are needed through the frozen encoder.  Detaching here is
        # intentional: B2B representation adaptation starts *after* z_LQ.
        with torch.no_grad():
            z_lq_ntchw = encode_reae_clip(
                self.reae,
                lq_input,
                require_4k_plus_1=True,
            )
        z_lq = z_lq_ntchw.permute(0, 2, 1, 3, 4).contiguous().detach()

        velocity = forward_b2a_compact_transformer_training(
            self.transformer,
            z_lq,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        if velocity.shape != z_lq.shape:
            raise ValueError(
                f"B2B velocity shape {tuple(velocity.shape)} does not match "
                f"z_LQ shape {tuple(z_lq.shape)}"
            )

        z_student = z_lq - velocity
        z_student_ntchw = z_student.permute(0, 2, 1, 3, 4).contiguous()
        prediction = self.decoder(
            z_student_ntchw,
            output_frames=int(target.shape[1]),
            clamp=False,
        )
        if prediction.shape != target.shape:
            raise ValueError(
                f"B2B RGB shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )
        return {
            "velocity": velocity,
            "z_lq": z_lq,
            "z_student": z_student,
            "prediction": prediction,
            "target": target,
            "lq_input": lq_input,
        }


def b2b_joint_objective(
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    representation_mse_weight: float = 0.05,
    representation_cosine_weight: float = 0.05,
    teacher_rgb_l1_weight: float = 1.0,
    gt_rgb_l1_weight: float = 0.5,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Minimal B2B-v1 loss: weak representation anchor + teacher/GT RGB.

    The representation weights are deliberately small.  A strong velocity loss
    would force the student back toward the fixed Stage-A latent that B2B-0C
    showed is difficult for the 13.36-GMAC decoder to render globally.
    """

    return velocity_distillation_objective(
        student_velocity,
        teacher_velocity,
        student_prediction=student_prediction,
        teacher_prediction=teacher_prediction,
        target=target,
        velocity_mse_weight=representation_mse_weight,
        velocity_cosine_weight=representation_cosine_weight,
        output_l1_weight=teacher_rgb_l1_weight,
        output_temporal_weight=0.0,
        gt_loss_mode="direct" if gt_rgb_l1_weight > 0 else "none",
        gt_pixel_weight=gt_rgb_l1_weight,
        gt_temporal_weight=0.0,
        epsilon=epsilon,
    )
