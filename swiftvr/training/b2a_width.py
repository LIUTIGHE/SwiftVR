"""Structured Wan-family width compression helpers for SwiftVR B2-A.

B2-A keeps the Stage-A prompt-free/no-time contract, 30-block depth, patch
geometry, ReAE latent interface, and 128-D attention heads.  It only shrinks
Transformer width toward the mature Wan2.1-1.3B shape:

    hidden: 3072 -> 1536
    heads:     24 -> 12  (head_dim stays 128)
    FFN:    14336 -> 8960

The helpers are intentionally model-structure aware but dataset agnostic.  They
collect activation RMS importance on a frozen Stage-A teacher and then extract a
coherent compact subnetwork by slicing residual channels, whole attention heads,
and per-block FFN neurons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class B2AWidthSpec:
    hidden_dim: int = 1536
    num_heads: int = 12
    ffn_dim: int = 8960
    head_dim: int = 128
    num_layers: int = 30
    adapter_dim: int = 128


def _base_block(block: nn.Module) -> nn.Module:
    return getattr(block, "_orig_mod", block)


def _linear_pair(ffn: nn.Module, hidden_dim: int, ffn_dim: int) -> tuple[nn.Linear, nn.Linear]:
    up = [
        module
        for module in ffn.modules()
        if isinstance(module, nn.Linear)
        and module.in_features == hidden_dim
        and module.out_features == ffn_dim
    ]
    down = [
        module
        for module in ffn.modules()
        if isinstance(module, nn.Linear)
        and module.in_features == ffn_dim
        and module.out_features == hidden_dim
    ]
    if len(up) != 1 or len(down) != 1:
        raise RuntimeError(
            "Expected one FFN up/down Linear pair, got "
            f"up={[(m.in_features, m.out_features) for m in up]}, "
            f"down={[(m.in_features, m.out_features) for m in down]}"
        )
    return up[0], down[0]


def transformer_width_shape(transformer: nn.Module) -> dict[str, int]:
    blocks = list(getattr(transformer, "blocks", []))
    if not blocks:
        raise ValueError("Transformer has no blocks")
    first = _base_block(blocks[0])
    hidden = int(first.attn1.to_q.in_features)
    heads = int(first.attn1.heads)
    inner = int(first.attn1.inner_dim)
    if hidden != inner:
        raise ValueError(f"B2-A expects self-attention inner_dim==hidden_dim, got {inner} vs {hidden}")
    if hidden % heads:
        raise ValueError(f"hidden_dim={hidden} is not divisible by heads={heads}")
    ffn_candidates = sorted(
        {
            int(module.out_features)
            for module in first.ffn.modules()
            if isinstance(module, nn.Linear)
            and int(module.in_features) == hidden
            and int(module.out_features) != hidden
        }
    )
    if len(ffn_candidates) != 1:
        raise RuntimeError(f"Cannot infer unique FFN width: {ffn_candidates}")
    return {
        "hidden_dim": hidden,
        "num_heads": heads,
        "head_dim": hidden // heads,
        "ffn_dim": int(ffn_candidates[0]),
        "num_layers": len(blocks),
        "adapter_dim": int(first.prompt_free_adapter.down.out_features),
    }


def validate_b2a_teacher_shape(transformer: nn.Module, spec: B2AWidthSpec) -> dict[str, int]:
    source = transformer_width_shape(transformer)
    if source["num_layers"] != spec.num_layers:
        raise ValueError(
            f"B2-A keeps depth fixed at {spec.num_layers}, teacher has {source['num_layers']}"
        )
    if source["head_dim"] != spec.head_dim:
        raise ValueError(
            f"B2-A keeps head_dim={spec.head_dim}, teacher has {source['head_dim']}"
        )
    if source["adapter_dim"] != spec.adapter_dim:
        raise ValueError(
            f"B2-A keeps adapter_dim={spec.adapter_dim}, teacher has {source['adapter_dim']}"
        )
    if spec.hidden_dim >= source["hidden_dim"]:
        raise ValueError("Compact hidden_dim must be smaller than teacher hidden_dim")
    if spec.num_heads >= source["num_heads"]:
        raise ValueError("Compact num_heads must be smaller than teacher num_heads")
    if spec.ffn_dim >= source["ffn_dim"]:
        raise ValueError("Compact ffn_dim must be smaller than teacher ffn_dim")
    if spec.hidden_dim != spec.num_heads * spec.head_dim:
        raise ValueError(
            "Compact hidden dimension must equal num_heads * head_dim: "
            f"{spec.hidden_dim} != {spec.num_heads} * {spec.head_dim}"
        )
    return source


def _topk_sorted(score: torch.Tensor, k: int) -> torch.Tensor:
    score = score.detach().float().cpu().reshape(-1)
    if k <= 0 or k > score.numel():
        raise ValueError(f"Invalid top-k={k} for {score.numel()} values")
    if not torch.isfinite(score).all():
        raise ValueError("Importance scores contain non-finite values")
    selected = torch.topk(score, k=k, largest=True, sorted=False).indices
    return torch.sort(selected).values.to(dtype=torch.long)


def expand_head_indices(head_indices: torch.Tensor, head_dim: int) -> torch.Tensor:
    heads = head_indices.detach().cpu().to(dtype=torch.long).reshape(-1)
    if heads.numel() == 0:
        raise ValueError("At least one attention head must be selected")
    offsets = torch.arange(int(head_dim), dtype=torch.long)
    return (heads[:, None] * int(head_dim) + offsets[None, :]).reshape(-1)


class ActivationImportanceCollector:
    """Collect scale-normalized activation RMS importance from a frozen teacher.

    Hidden-channel importance is aggregated across every block input using one
    shared channel basis.  Attention-head and FFN-neuron scores remain block
    local, which avoids imposing artificial alignment between different blocks.
    """

    def __init__(self, transformer: nn.Module) -> None:
        shape = transformer_width_shape(transformer)
        self.hidden_dim = shape["hidden_dim"]
        self.num_heads = shape["num_heads"]
        self.head_dim = shape["head_dim"]
        self.ffn_dim = shape["ffn_dim"]
        self.num_layers = shape["num_layers"]

        self.hidden_sum_sq = torch.zeros(self.num_layers, self.hidden_dim, dtype=torch.float64)
        self.hidden_count = torch.zeros(self.num_layers, dtype=torch.float64)
        self.head_sum_sq = torch.zeros(self.num_layers, self.num_heads, dtype=torch.float64)
        self.head_count = torch.zeros(self.num_layers, dtype=torch.float64)
        self.ffn_sum_sq = torch.zeros(self.num_layers, self.ffn_dim, dtype=torch.float64)
        self.ffn_count = torch.zeros(self.num_layers, dtype=torch.float64)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for index, raw_block in enumerate(transformer.blocks):
            block = _base_block(raw_block)
            _, ffn_down = _linear_pair(block.ffn, self.hidden_dim, self.ffn_dim)
            self._handles.append(block.register_forward_pre_hook(self._hidden_hook(index)))
            self._handles.append(block.attn1.to_out[0].register_forward_pre_hook(self._head_hook(index)))
            self._handles.append(ffn_down.register_forward_pre_hook(self._ffn_hook(index)))

    @staticmethod
    def _tensor_from_inputs(inputs) -> torch.Tensor:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("Importance hook expected a tensor as its first input")
        return inputs[0].detach()

    def _hidden_hook(self, index: int):
        def hook(_module, inputs):
            value = self._tensor_from_inputs(inputs)
            if value.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"Block {index} hidden width changed: {value.shape[-1]} != {self.hidden_dim}"
                )
            flat = value.float().reshape(-1, self.hidden_dim)
            self.hidden_sum_sq[index] += flat.square().sum(dim=0).double().cpu()
            self.hidden_count[index] += flat.shape[0]
        return hook

    def _head_hook(self, index: int):
        def hook(_module, inputs):
            value = self._tensor_from_inputs(inputs)
            if value.shape[-1] != self.num_heads * self.head_dim:
                raise ValueError(f"Block {index} attention output width mismatch")
            flat = value.float().reshape(-1, self.num_heads, self.head_dim)
            self.head_sum_sq[index] += flat.square().sum(dim=(0, 2)).double().cpu()
            self.head_count[index] += flat.shape[0] * self.head_dim
        return hook

    def _ffn_hook(self, index: int):
        def hook(_module, inputs):
            value = self._tensor_from_inputs(inputs)
            if value.shape[-1] != self.ffn_dim:
                raise ValueError(
                    f"Block {index} FFN width changed: {value.shape[-1]} != {self.ffn_dim}"
                )
            flat = value.float().reshape(-1, self.ffn_dim)
            self.ffn_sum_sq[index] += flat.square().sum(dim=0).double().cpu()
            self.ffn_count[index] += flat.shape[0]
        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def scores(self) -> dict[str, torch.Tensor]:
        if (self.hidden_count <= 0).any() or (self.head_count <= 0).any() or (self.ffn_count <= 0).any():
            raise RuntimeError("Importance calibration did not execute every block/component")

        hidden_rms = torch.sqrt(self.hidden_sum_sq / self.hidden_count[:, None]).float()
        hidden_normalized = hidden_rms / hidden_rms.mean(dim=1, keepdim=True).clamp_min(1e-12)
        hidden_global = hidden_normalized.mean(dim=0)
        head_rms = torch.sqrt(self.head_sum_sq / self.head_count[:, None]).float()
        ffn_rms = torch.sqrt(self.ffn_sum_sq / self.ffn_count[:, None]).float()
        return {
            "hidden_global": hidden_global,
            "hidden_by_block": hidden_rms,
            "head_by_block": head_rms,
            "ffn_by_block": ffn_rms,
        }

    def select(self, spec: B2AWidthSpec) -> dict[str, object]:
        scores = self.scores()
        return {
            "hidden": _topk_sorted(scores["hidden_global"], spec.hidden_dim),
            "heads": [
                _topk_sorted(scores["head_by_block"][index], spec.num_heads)
                for index in range(self.num_layers)
            ],
            "ffn": [
                _topk_sorted(scores["ffn_by_block"][index], spec.ffn_dim)
                for index in range(self.num_layers)
            ],
        }


def _copy_parameter_(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.shape != source.shape:
        raise ValueError(f"Structured transfer shape mismatch: {tuple(target.shape)} != {tuple(source.shape)}")
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def _slice(source: torch.Tensor, *selections: tuple[int, torch.Tensor]) -> torch.Tensor:
    result = source.detach()
    for dim, indices in selections:
        result = torch.index_select(result, int(dim), indices.to(device=result.device, dtype=torch.long))
    return result


@torch.no_grad()
def transfer_structured_width(
    teacher: nn.Module,
    student: nn.Module,
    *,
    hidden_indices: torch.Tensor,
    head_indices_by_block: Sequence[torch.Tensor],
    ffn_indices_by_block: Sequence[torch.Tensor],
    spec: B2AWidthSpec = B2AWidthSpec(),
) -> dict[str, object]:
    """Extract a compact student from a Stage-A teacher without changing depth."""

    source = validate_b2a_teacher_shape(teacher, spec)
    target = transformer_width_shape(student)
    expected_target = {
        "hidden_dim": spec.hidden_dim,
        "num_heads": spec.num_heads,
        "head_dim": spec.head_dim,
        "ffn_dim": spec.ffn_dim,
        "num_layers": spec.num_layers,
        "adapter_dim": spec.adapter_dim,
    }
    if target != expected_target:
        raise ValueError(f"Student shape is not B2-A target: {target} != {expected_target}")

    hidden = hidden_indices.detach().cpu().to(dtype=torch.long).reshape(-1)
    if hidden.numel() != spec.hidden_dim or torch.unique(hidden).numel() != hidden.numel():
        raise ValueError("hidden_indices must contain unique compact hidden channels")
    if hidden.min() < 0 or hidden.max() >= source["hidden_dim"]:
        raise ValueError("hidden_indices are out of teacher range")
    if len(head_indices_by_block) != spec.num_layers or len(ffn_indices_by_block) != spec.num_layers:
        raise ValueError("Need one head/FFN selection per transformer block")

    _copy_parameter_(student.patch_embedding.weight, _slice(teacher.patch_embedding.weight, (0, hidden)))
    if student.patch_embedding.bias is not None:
        _copy_parameter_(student.patch_embedding.bias, _slice(teacher.patch_embedding.bias, (0, hidden)))

    for index, (teacher_raw, student_raw) in enumerate(zip(teacher.blocks, student.blocks)):
        tb = _base_block(teacher_raw)
        sb = _base_block(student_raw)
        heads = head_indices_by_block[index].detach().cpu().to(dtype=torch.long).reshape(-1)
        if heads.numel() != spec.num_heads or torch.unique(heads).numel() != heads.numel():
            raise ValueError(f"Block {index}: invalid attention-head selection")
        if heads.min() < 0 or heads.max() >= source["num_heads"]:
            raise ValueError(f"Block {index}: attention-head selection out of range")
        qkv = expand_head_indices(heads, spec.head_dim)

        ffn = ffn_indices_by_block[index].detach().cpu().to(dtype=torch.long).reshape(-1)
        if ffn.numel() != spec.ffn_dim or torch.unique(ffn).numel() != ffn.numel():
            raise ValueError(f"Block {index}: invalid FFN-neuron selection")
        if ffn.min() < 0 or ffn.max() >= source["ffn_dim"]:
            raise ValueError(f"Block {index}: FFN-neuron selection out of range")

        _copy_parameter_(sb.scale_shift_table, _slice(tb.scale_shift_table, (2, hidden)))

        for name in ("to_q", "to_k", "to_v"):
            teacher_linear = getattr(tb.attn1, name)
            student_linear = getattr(sb.attn1, name)
            _copy_parameter_(student_linear.weight, _slice(teacher_linear.weight, (0, qkv), (1, hidden)))
            if student_linear.bias is not None:
                _copy_parameter_(student_linear.bias, _slice(teacher_linear.bias, (0, qkv)))

        _copy_parameter_(sb.attn1.norm_q.weight, _slice(tb.attn1.norm_q.weight, (0, qkv)))
        _copy_parameter_(sb.attn1.norm_k.weight, _slice(tb.attn1.norm_k.weight, (0, qkv)))
        _copy_parameter_(
            sb.attn1.to_out[0].weight,
            _slice(tb.attn1.to_out[0].weight, (0, hidden), (1, qkv)),
        )
        if sb.attn1.to_out[0].bias is not None:
            _copy_parameter_(sb.attn1.to_out[0].bias, _slice(tb.attn1.to_out[0].bias, (0, hidden)))

        _copy_parameter_(sb.prompt_free_adapter.norm.weight, _slice(tb.prompt_free_adapter.norm.weight, (0, hidden)))
        _copy_parameter_(sb.prompt_free_adapter.norm.bias, _slice(tb.prompt_free_adapter.norm.bias, (0, hidden)))
        _copy_parameter_(
            sb.prompt_free_adapter.down.weight,
            _slice(tb.prompt_free_adapter.down.weight, (1, hidden)),
        )
        _copy_parameter_(sb.prompt_free_adapter.down.bias, tb.prompt_free_adapter.down.bias.detach())
        _copy_parameter_(
            sb.prompt_free_adapter.up.weight,
            _slice(tb.prompt_free_adapter.up.weight, (0, hidden)),
        )
        _copy_parameter_(sb.prompt_free_adapter.up.bias, _slice(tb.prompt_free_adapter.up.bias, (0, hidden)))

        teacher_up, teacher_down = _linear_pair(tb.ffn, source["hidden_dim"], source["ffn_dim"])
        student_up, student_down = _linear_pair(sb.ffn, spec.hidden_dim, spec.ffn_dim)
        _copy_parameter_(student_up.weight, _slice(teacher_up.weight, (0, ffn), (1, hidden)))
        if student_up.bias is not None:
            _copy_parameter_(student_up.bias, _slice(teacher_up.bias, (0, ffn)))
        _copy_parameter_(student_down.weight, _slice(teacher_down.weight, (0, hidden), (1, ffn)))
        if student_down.bias is not None:
            _copy_parameter_(student_down.bias, _slice(teacher_down.bias, (0, hidden)))

    _copy_parameter_(student.scale_shift_table, _slice(teacher.scale_shift_table, (2, hidden)))
    _copy_parameter_(student.proj_out.weight, _slice(teacher.proj_out.weight, (1, hidden)))
    if student.proj_out.bias is not None:
        _copy_parameter_(student.proj_out.bias, teacher.proj_out.bias.detach())

    return {
        "teacher": source,
        "student": target,
        "hidden_indices": hidden.tolist(),
        "head_indices_by_block": [value.detach().cpu().long().tolist() for value in head_indices_by_block],
        "ffn_indices_by_block": [value.detach().cpu().long().tolist() for value in ffn_indices_by_block],
    }


def build_compact_transformer_from_teacher(
    teacher: nn.Module,
    spec: B2AWidthSpec = B2AWidthSpec(),
) -> nn.Module:
    """Instantiate the Wan-1.3B-shaped prompt-free/no-time student."""

    validate_b2a_teacher_shape(teacher, spec)
    from swiftvr.models.transformer_prompt_free_no_time import (
        WanTransformer3DModelPromptFreeNoTime,
    )

    cfg = teacher.config
    return WanTransformer3DModelPromptFreeNoTime(
        patch_size=tuple(cfg.patch_size),
        num_attention_heads=spec.num_heads,
        attention_head_dim=spec.head_dim,
        in_channels=int(cfg.in_channels),
        out_channels=int(cfg.out_channels),
        text_dim=int(getattr(cfg, "text_dim", 4096)),
        freq_dim=int(getattr(cfg, "freq_dim", 256)),
        ffn_dim=spec.ffn_dim,
        num_layers=spec.num_layers,
        cross_attn_norm=bool(getattr(cfg, "cross_attn_norm", True)),
        qk_norm=str(getattr(cfg, "qk_norm", "rms_norm_across_heads")),
        eps=float(getattr(cfg, "eps", 1e-6)),
        image_dim=getattr(cfg, "image_dim", None),
        added_kv_proj_dim=getattr(cfg, "added_kv_proj_dim", None),
        rope_max_seq_len=int(getattr(cfg, "rope_max_seq_len", 1024)),
        pos_embed_seq_len=getattr(cfg, "pos_embed_seq_len", None),
        enable_swa=bool(getattr(cfg, "enable_swa", True)),
        self_attn_window_hw=tuple(getattr(cfg, "self_attn_window_hw", (16, 16))),
        use_torch_compile=False,
        compile_mode="default",
        adapter_dim=spec.adapter_dim,
        folded_timestep=float(getattr(cfg, "folded_timestep", 1000.0)),
        time_condition_folded=True,
    )
