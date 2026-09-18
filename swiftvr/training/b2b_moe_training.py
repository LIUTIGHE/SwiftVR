"""Training-only helpers for the D1024 sparse-MoE SwiftVR student.

This module keeps inference/checkpoint semantics untouched.  It mirrors the
validated prompt-free/no-time B2-A forward, but calls ``SparseMoEFFN.forward_with_aux``
so the differentiable router load-balance loss is explicit in the batch output.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .b2a_width import _base_block
from .forward import (
    _transformer_patch_size,
    encode_reae_clip,
    prepare_prompt_free_no_time_transformer_for_training,
    prepare_training_batch,
)
from ..models.transformer_prompt_free_no_time_moe import SparseMoEFFN


def _forward_moe_block_training(block: nn.Module, hidden_states: torch.Tensor, rotary_emb):
    hidden_dtype = hidden_states.dtype
    mods = block.scale_shift_table.to(dtype=hidden_dtype)
    shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = mods.chunk(6, dim=1)

    attention_input = block.norm1(hidden_states)
    attention_input = attention_input * (1.0 + scale_msa) + shift_msa
    attention_output = block.attn1(attention_input, None, None, rotary_emb)
    hidden_states = hidden_states + attention_output * gate_msa

    hidden_states = hidden_states + block.prompt_free_adapter(hidden_states)

    ffn_input = block.norm3(hidden_states)
    ffn_input = ffn_input * (1.0 + scale_ffn) + shift_ffn
    if not isinstance(block.ffn, SparseMoEFFN):
        raise TypeError(f"Expected SparseMoEFFN, got {type(block.ffn).__name__}")
    ffn_output, balance_loss = block.ffn.forward_with_aux(ffn_input)
    return hidden_states + ffn_output * gate_ffn, balance_loss


def forward_moe_transformer_training(
    transformer: nn.Module,
    hidden_states: torch.Tensor,
    *,
    gradient_checkpointing: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the MoE DiT and return ``(velocity, mean_router_balance_loss)``."""
    if hidden_states.ndim != 5:
        raise ValueError(f"Expected [B,C,F,H,W], got {tuple(hidden_states.shape)}")

    batch, _, frames, height, width = hidden_states.shape
    patch_t, patch_h, patch_w = _transformer_patch_size(transformer)
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"Latent size {frames}x{height}x{width} is not divisible by "
            f"patch_size={(patch_t, patch_h, patch_w)}"
        )

    patched_frames = frames // patch_t
    patched_height = height // patch_h
    patched_width = width // patch_w
    rotary_emb = transformer.rope(hidden_states)
    tokens = transformer.patch_embedding(hidden_states).flatten(2).transpose(1, 2).contiguous()
    thw = (patched_frames, patched_height, patched_width)
    balance_total = tokens.new_zeros(())

    for index, raw_block in enumerate(transformer.blocks):
        block = _base_block(raw_block)
        if not isinstance(block.ffn, SparseMoEFFN):
            raise TypeError(f"Block {index} does not contain SparseMoEFFN")
        block.attn1._thw = thw

        if gradient_checkpointing and torch.is_grad_enabled():
            def block_forward(value, *, _block=block):
                return _forward_moe_block_training(_block, value, rotary_emb)

            tokens, block_balance = activation_checkpoint(
                block_forward, tokens, use_reentrant=False
            )
        else:
            tokens, block_balance = _forward_moe_block_training(block, tokens, rotary_emb)
        balance_total = balance_total + block_balance

    hidden_dtype = tokens.dtype
    shift, scale = transformer.scale_shift_table.to(hidden_dtype).chunk(2, dim=1)
    tokens = transformer.norm_out(tokens)
    tokens = tokens * (1.0 + scale) + shift
    tokens = transformer.proj_out(tokens)
    tokens = tokens.reshape(
        batch,
        patched_frames,
        patched_height,
        patched_width,
        patch_t,
        patch_h,
        patch_w,
        -1,
    )
    tokens = tokens.permute(0, 7, 1, 4, 2, 5, 3, 6)
    velocity = tokens.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return velocity, balance_total / max(len(transformer.blocks), 1)


class B2BMoEVelocityDistillationForward(nn.Module):
    """Frozen ReAE encoder + trainable D1024 sparse-MoE DiT; no decoder."""

    def __init__(
        self,
        reae: nn.Module,
        transformer: nn.Module,
        *,
        attention_backend: str = "sdpa",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.reae = reae
        self.transformer = transformer
        self.gradient_checkpointing = bool(gradient_checkpointing)
        prepare_prompt_free_no_time_transformer_for_training(
            self.transformer, attention_backend=attention_backend
        )

    def forward(self, batch) -> dict[str, torch.Tensor]:
        prepared = prepare_training_batch(batch)
        lq_input = prepared["lq_input"]
        target = prepared["target"]
        if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("Prepared batch is missing lq_input/target tensors")
        z_lq_ntchw = encode_reae_clip(self.reae, lq_input, require_4k_plus_1=True)
        z_lq = z_lq_ntchw.permute(0, 2, 1, 3, 4).contiguous()
        velocity, balance_loss = forward_moe_transformer_training(
            self.transformer,
            z_lq,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        if velocity.shape != z_lq.shape:
            raise ValueError(f"Velocity shape {tuple(velocity.shape)} != latent {tuple(z_lq.shape)}")
        return {
            "velocity": velocity,
            "router_balance_loss": balance_loss,
            "z_lq": z_lq,
            "target": target,
            "lq_input": lq_input,
        }


def router_summary(transformer: nn.Module) -> dict[str, object]:
    """Summarize the latest routing decisions across all Transformer blocks."""
    stats = transformer.router_stats()
    valid = [item for item in stats if item is not None]
    if not valid:
        raise RuntimeError("Router statistics are unavailable; run a forward first")
    num_experts = len(valid[0].expert_counts)
    counts = [0] * num_experts
    tokens = 0
    assignments = 0
    entropy = 0.0
    balance = 0.0
    for item in valid:
        if len(item.expert_counts) != num_experts:
            raise RuntimeError("Router expert count changed between blocks")
        tokens += int(item.token_count)
        assignments += int(item.assignment_count)
        entropy += float(item.probability_entropy)
        balance += float(item.balance_loss)
        for index, value in enumerate(item.expert_counts):
            counts[index] += int(value)
    fractions = [value / max(assignments, 1) for value in counts]
    mean_fraction = sum(fractions) / max(len(fractions), 1)
    variance = sum((value - mean_fraction) ** 2 for value in fractions) / max(len(fractions), 1)
    cv = math.sqrt(variance) / max(mean_fraction, 1e-12)
    return {
        "blocks": len(valid),
        "tokens": tokens,
        "assignments": assignments,
        "expert_counts": counts,
        "expert_fractions": fractions,
        "mean_entropy": entropy / len(valid),
        "normalized_entropy": (entropy / len(valid)) / math.log(num_experts),
        "mean_balance_loss": balance / len(valid),
        "min_fraction": min(fractions),
        "max_fraction": max(fractions),
        "load_cv": cv,
    }
