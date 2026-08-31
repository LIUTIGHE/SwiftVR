"""D1536 teaching-assistant -> D1024 sparse-MoE construction helpers.

The target keeps 30 Transformer blocks and 128-D attention heads but widens the
D768 residual stream to 1024 while replacing each dense FFN with a 1S12E2A MoE.
Initialization is teacher-centric:

* residual channels and whole heads are selected by activation RMS;
* the most important dense-TA FFN neurons initialize the shared expert;
* the next most important neurons are distributed round-robin across routed
  experts so every expert inherits useful dense weights;
* router weights start from a small deterministic near-uniform projection.

No decoder or GT target participates in this transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .b2a_width import (
    _base_block,
    _copy_parameter_,
    _linear_pair,
    _slice,
    expand_head_indices,
    transformer_width_shape,
)
from ..models.transformer_prompt_free_no_time_moe import (
    SparseMoEFFN,
    WanTransformer3DModelPromptFreeNoTimeMoE,
)


@dataclass(frozen=True)
class B2BMoESpec:
    hidden_dim: int = 1024
    num_heads: int = 8
    head_dim: int = 128
    num_layers: int = 30
    adapter_dim: int = 128
    shared_expert_dim: int = 1024
    normal_expert_dim: int = 256
    num_experts: int = 12
    top_k: int = 2

    @property
    def total_ffn_dim(self) -> int:
        return self.shared_expert_dim + self.num_experts * self.normal_expert_dim

    @property
    def active_ffn_dim(self) -> int:
        return self.shared_expert_dim + self.top_k * self.normal_expert_dim


EXPECTED_D1536_TA = {
    "hidden_dim": 1536,
    "num_heads": 12,
    "head_dim": 128,
    "ffn_dim": 8960,
    "num_layers": 30,
    "adapter_dim": 128,
}


def transformer_moe_shape(transformer: nn.Module) -> dict[str, int | float]:
    blocks = list(getattr(transformer, "blocks", []))
    if not blocks:
        raise ValueError("Transformer has no blocks")
    first = _base_block(blocks[0])
    hidden = int(first.attn1.to_q.in_features)
    heads = int(first.attn1.heads)
    if hidden % heads:
        raise ValueError("MoE hidden width is not divisible by attention heads")
    ffn = first.ffn
    if not isinstance(ffn, SparseMoEFFN):
        raise TypeError(f"Expected SparseMoEFFN, got {type(ffn).__name__}")
    return {
        "hidden_dim": hidden,
        "num_heads": heads,
        "head_dim": hidden // heads,
        "num_layers": len(blocks),
        "adapter_dim": int(first.prompt_free_adapter.down.out_features),
        "shared_expert_dim": int(ffn.shared_expert_dim),
        "normal_expert_dim": int(ffn.normal_expert_dim),
        "num_experts": int(ffn.num_experts),
        "top_k": int(ffn.top_k),
        "total_ffn_dim": int(ffn.shared_expert_dim + ffn.num_experts * ffn.normal_expert_dim),
        "active_ffn_dim": int(ffn.shared_expert_dim + ffn.top_k * ffn.normal_expert_dim),
        "total_expansion": float(ffn.total_expansion),
        "active_expansion": float(ffn.active_expansion),
    }


def expected_moe_shape(spec: B2BMoESpec = B2BMoESpec()) -> dict[str, int | float]:
    return {
        "hidden_dim": spec.hidden_dim,
        "num_heads": spec.num_heads,
        "head_dim": spec.head_dim,
        "num_layers": spec.num_layers,
        "adapter_dim": spec.adapter_dim,
        "shared_expert_dim": spec.shared_expert_dim,
        "normal_expert_dim": spec.normal_expert_dim,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "total_ffn_dim": spec.total_ffn_dim,
        "active_ffn_dim": spec.active_ffn_dim,
        "total_expansion": spec.total_ffn_dim / spec.hidden_dim,
        "active_expansion": spec.active_ffn_dim / spec.hidden_dim,
    }


def validate_d1536_ta(transformer: nn.Module) -> dict[str, int]:
    shape = transformer_width_shape(transformer)
    if shape != EXPECTED_D1536_TA:
        raise ValueError(f"Expected the locked B2-A D1536 TA, got {shape}")
    return shape


def validate_moe_spec(spec: B2BMoESpec) -> None:
    if spec.hidden_dim != spec.num_heads * spec.head_dim:
        raise ValueError("hidden_dim must equal num_heads * head_dim")
    if spec.num_layers <= 0 or spec.adapter_dim <= 0:
        raise ValueError("num_layers/adapter_dim must be positive")
    if spec.shared_expert_dim <= 0 or spec.normal_expert_dim <= 0:
        raise ValueError("expert dimensions must be positive")
    if not 1 <= spec.top_k <= spec.num_experts:
        raise ValueError("top_k must be in [1,num_experts]")


def build_moe_transformer_from_teacher(
    teacher: nn.Module,
    spec: B2BMoESpec = B2BMoESpec(),
) -> WanTransformer3DModelPromptFreeNoTimeMoE:
    validate_d1536_ta(teacher)
    validate_moe_spec(spec)
    cfg = teacher.config
    return WanTransformer3DModelPromptFreeNoTimeMoE(
        patch_size=tuple(cfg.patch_size),
        num_attention_heads=spec.num_heads,
        attention_head_dim=spec.head_dim,
        in_channels=int(cfg.in_channels),
        out_channels=int(cfg.out_channels),
        text_dim=int(getattr(cfg, "text_dim", 4096)),
        freq_dim=int(getattr(cfg, "freq_dim", 256)),
        ffn_dim=spec.active_ffn_dim,
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
        shared_expert_dim=spec.shared_expert_dim,
        normal_expert_dim=spec.normal_expert_dim,
        num_experts=spec.num_experts,
        top_k=spec.top_k,
    )


def _rank_indices(score: torch.Tensor, count: int) -> torch.Tensor:
    value = score.detach().float().cpu().reshape(-1)
    if count <= 0 or count > int(value.numel()):
        raise ValueError(f"Invalid selection count {count} for {value.numel()} values")
    if not torch.isfinite(value).all():
        raise ValueError("Importance score contains non-finite values")
    return torch.topk(value, k=count, largest=True, sorted=True).indices.long()


def partition_ffn_neurons(
    score: torch.Tensor,
    spec: B2BMoESpec = B2BMoESpec(),
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Allocate dense-TA FFN neurons to shared and normal experts.

    The shared expert receives the highest-ranked neurons.  The next
    ``num_experts*normal_expert_dim`` neurons are assigned round-robin by rank,
    avoiding a systematic importance advantage for expert 0 over expert N.
    """
    total = spec.shared_expert_dim + spec.num_experts * spec.normal_expert_dim
    ranked = _rank_indices(score, total)
    shared = torch.sort(ranked[: spec.shared_expert_dim]).values
    normal_ranked = ranked[spec.shared_expert_dim :]
    experts: list[torch.Tensor] = []
    for expert_id in range(spec.num_experts):
        values = normal_ranked[expert_id :: spec.num_experts]
        if int(values.numel()) != spec.normal_expert_dim:
            raise RuntimeError(
                f"Expert {expert_id} received {values.numel()} neurons, "
                f"expected {spec.normal_expert_dim}"
            )
        experts.append(torch.sort(values).values)
    return shared, experts


def _copy_expert_from_dense(
    *,
    teacher_up: nn.Linear,
    teacher_down: nn.Linear,
    expert,
    hidden_indices: torch.Tensor,
    neuron_indices: torch.Tensor,
    copy_down_bias: bool,
) -> None:
    _copy_parameter_(
        expert.up.weight,
        _slice(teacher_up.weight, (0, neuron_indices), (1, hidden_indices)),
    )
    if expert.up.bias is not None:
        _copy_parameter_(expert.up.bias, _slice(teacher_up.bias, (0, neuron_indices)))
    _copy_parameter_(
        expert.down.weight,
        _slice(teacher_down.weight, (0, hidden_indices), (1, neuron_indices)),
    )
    if expert.down.bias is not None:
        if copy_down_bias:
            _copy_parameter_(expert.down.bias, _slice(teacher_down.bias, (0, hidden_indices)))
        else:
            expert.down.bias.zero_()


@torch.no_grad()
def transfer_d1536_to_moe(
    teacher: nn.Module,
    student: nn.Module,
    *,
    hidden_indices: torch.Tensor,
    head_indices_by_block: Sequence[torch.Tensor],
    ffn_scores_by_block: torch.Tensor,
    spec: B2BMoESpec = B2BMoESpec(),
    router_seed: int = 20260831,
    router_init_std: float = 1e-3,
) -> dict[str, object]:
    source = validate_d1536_ta(teacher)
    validate_moe_spec(spec)
    target = transformer_moe_shape(student)
    expected = expected_moe_shape(spec)
    if target != expected:
        raise ValueError(f"Student MoE shape mismatch: {target} != {expected}")
    if router_init_std < 0:
        raise ValueError("router_init_std must be non-negative")

    hidden = hidden_indices.detach().cpu().long().reshape(-1)
    if int(hidden.numel()) != spec.hidden_dim or int(torch.unique(hidden).numel()) != spec.hidden_dim:
        raise ValueError("hidden_indices must contain unique target hidden channels")
    if int(hidden.min()) < 0 or int(hidden.max()) >= source["hidden_dim"]:
        raise ValueError("hidden_indices out of D1536 range")
    if len(head_indices_by_block) != spec.num_layers:
        raise ValueError("Need one head selection per block")
    if tuple(ffn_scores_by_block.shape) != (spec.num_layers, source["ffn_dim"]):
        raise ValueError(
            f"ffn_scores_by_block must be {(spec.num_layers, source['ffn_dim'])}, "
            f"got {tuple(ffn_scores_by_block.shape)}"
        )

    _copy_parameter_(
        student.patch_embedding.weight,
        _slice(teacher.patch_embedding.weight, (0, hidden)),
    )
    if student.patch_embedding.bias is not None:
        _copy_parameter_(
            student.patch_embedding.bias,
            _slice(teacher.patch_embedding.bias, (0, hidden)),
        )

    expert_allocations: list[dict[str, object]] = []
    for index, (teacher_raw, student_raw) in enumerate(zip(teacher.blocks, student.blocks)):
        tb = _base_block(teacher_raw)
        sb = _base_block(student_raw)
        heads = head_indices_by_block[index].detach().cpu().long().reshape(-1)
        if int(heads.numel()) != spec.num_heads or int(torch.unique(heads).numel()) != spec.num_heads:
            raise ValueError(f"Block {index}: invalid head selection")
        if int(heads.min()) < 0 or int(heads.max()) >= source["num_heads"]:
            raise ValueError(f"Block {index}: head selection out of range")
        qkv = expand_head_indices(heads, spec.head_dim)

        _copy_parameter_(sb.scale_shift_table, _slice(tb.scale_shift_table, (2, hidden)))
        for name in ("to_q", "to_k", "to_v"):
            src = getattr(tb.attn1, name)
            dst = getattr(sb.attn1, name)
            _copy_parameter_(dst.weight, _slice(src.weight, (0, qkv), (1, hidden)))
            if dst.bias is not None:
                _copy_parameter_(dst.bias, _slice(src.bias, (0, qkv)))

        _copy_parameter_(sb.attn1.norm_q.weight, _slice(tb.attn1.norm_q.weight, (0, qkv)))
        _copy_parameter_(sb.attn1.norm_k.weight, _slice(tb.attn1.norm_k.weight, (0, qkv)))
        _copy_parameter_(
            sb.attn1.to_out[0].weight,
            _slice(tb.attn1.to_out[0].weight, (0, hidden), (1, qkv)),
        )
        if sb.attn1.to_out[0].bias is not None:
            _copy_parameter_(
                sb.attn1.to_out[0].bias,
                _slice(tb.attn1.to_out[0].bias, (0, hidden)),
            )

        _copy_parameter_(
            sb.prompt_free_adapter.norm.weight,
            _slice(tb.prompt_free_adapter.norm.weight, (0, hidden)),
        )
        _copy_parameter_(
            sb.prompt_free_adapter.norm.bias,
            _slice(tb.prompt_free_adapter.norm.bias, (0, hidden)),
        )
        _copy_parameter_(
            sb.prompt_free_adapter.down.weight,
            _slice(tb.prompt_free_adapter.down.weight, (1, hidden)),
        )
        _copy_parameter_(
            sb.prompt_free_adapter.down.bias,
            tb.prompt_free_adapter.down.bias.detach(),
        )
        _copy_parameter_(
            sb.prompt_free_adapter.up.weight,
            _slice(tb.prompt_free_adapter.up.weight, (0, hidden)),
        )
        _copy_parameter_(
            sb.prompt_free_adapter.up.bias,
            _slice(tb.prompt_free_adapter.up.bias, (0, hidden)),
        )

        teacher_up, teacher_down = _linear_pair(
            tb.ffn, source["hidden_dim"], source["ffn_dim"]
        )
        shared_indices, normal_indices = partition_ffn_neurons(
            ffn_scores_by_block[index], spec
        )
        _copy_expert_from_dense(
            teacher_up=teacher_up,
            teacher_down=teacher_down,
            expert=sb.ffn.shared_expert,
            hidden_indices=hidden,
            neuron_indices=shared_indices,
            copy_down_bias=True,
        )
        for expert, indices in zip(sb.ffn.experts, normal_indices):
            _copy_expert_from_dense(
                teacher_up=teacher_up,
                teacher_down=teacher_down,
                expert=expert,
                hidden_indices=hidden,
                neuron_indices=indices,
                copy_down_bias=False,
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(router_seed) + index)
        initial_router = torch.randn(
            tuple(sb.ffn.router.weight.shape),
            generator=generator,
            dtype=torch.float32,
        ) * float(router_init_std)
        _copy_parameter_(sb.ffn.router.weight, initial_router)

        expert_allocations.append(
            {
                "block": index,
                "shared": shared_indices.tolist(),
                "normal": [value.tolist() for value in normal_indices],
            }
        )

    _copy_parameter_(
        student.scale_shift_table,
        _slice(teacher.scale_shift_table, (2, hidden)),
    )
    _copy_parameter_(
        student.proj_out.weight,
        _slice(teacher.proj_out.weight, (1, hidden)),
    )
    if student.proj_out.bias is not None:
        _copy_parameter_(student.proj_out.bias, teacher.proj_out.bias.detach())

    return {
        "teacher_shape": source,
        "student_shape": target,
        "hidden_indices": hidden.tolist(),
        "head_indices_by_block": [
            value.detach().cpu().long().tolist() for value in head_indices_by_block
        ],
        "expert_allocations": expert_allocations,
        "router_seed": int(router_seed),
        "router_init_std": float(router_init_std),
    }


def parameter_accounting(
    transformer: nn.Module,
) -> dict[str, int | float]:
    """Report total and exactly activated parameters for fixed top-k routing."""
    shape = transformer_moe_shape(transformer)
    total = sum(int(parameter.numel()) for parameter in transformer.parameters())
    all_expert_params = 0
    active_expert_params = 0

    for raw_block in transformer.blocks:
        block = _base_block(raw_block)
        ffn = block.ffn
        shared = sum(int(p.numel()) for p in ffn.shared_expert.parameters())
        normal_each = [
            sum(int(p.numel()) for p in expert.parameters())
            for expert in ffn.experts
        ]
        if len(set(normal_each)) != 1:
            raise ValueError("Normal experts do not have equal parameter counts")
        all_expert_params += shared + sum(normal_each)
        active_expert_params += shared + ffn.top_k * normal_each[0]

    active = total - all_expert_params + active_expert_params
    return {
        "total_parameters": total,
        "activated_parameters": active,
        "inactive_routed_parameters": total - active,
        "activated_fraction": active / total,
        "total_ffn_dim": int(shape["total_ffn_dim"]),
        "active_ffn_dim": int(shape["active_ffn_dim"]),
    }
