"""Sparse-MoE prompt-free/no-time SwiftVR DiT.

This module keeps the Stage-A one-step DiT contract unchanged while replacing
each dense FFN with a Dense2MoE-style shared+routed expert layer.  The first
deployment candidate is intentionally conservative:

    hidden=1024, heads=8, head_dim=128, layers=30
    shared expert width=1024
    12 routed experts, width=256 each
    top-k=2 routed experts per token

Total FFN capacity is 4x hidden width, while every token activates only 1.5x
hidden width (shared 1.0x + routed 2*0.25x).  No token or Transformer block is
skipped in this first architecture gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from diffusers.configuration_utils import register_to_config

from .transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
    WanTransformerBlockPromptFreeNoTime,
)


@dataclass(frozen=True)
class SparseMoERouterStats:
    token_count: int
    assignment_count: int
    expert_counts: tuple[int, ...]
    probability_entropy: float
    balance_loss: float


class SparseMoEExpert(nn.Module):
    """Two-linear GELU FFN expert matching SwiftVR's dense FFN semantics."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("Expert dimensions must be positive")
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.up = nn.Linear(self.dim, self.hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.down = nn.Linear(self.hidden_dim, self.dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(hidden_states)))


class SparseMoEFFN(nn.Module):
    """Shared expert plus token-wise top-k routed experts.

    The implementation performs real sparse dispatch: routed experts receive only
    the tokens assigned to them.  There is no capacity dropping.  Top-k gate
    probabilities are renormalized over the selected experts before combination.
    """

    def __init__(
        self,
        dim: int,
        *,
        shared_expert_dim: int,
        normal_expert_dim: int,
        num_experts: int = 12,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.shared_expert_dim = int(shared_expert_dim)
        self.normal_expert_dim = int(normal_expert_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        if self.dim <= 0 or self.shared_expert_dim <= 0 or self.normal_expert_dim <= 0:
            raise ValueError("MoE dimensions must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")

        self.shared_expert = SparseMoEExpert(self.dim, self.shared_expert_dim)
        self.experts = nn.ModuleList(
            SparseMoEExpert(self.dim, self.normal_expert_dim)
            for _ in range(self.num_experts)
        )
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)
        self._last_stats: SparseMoERouterStats | None = None

    @property
    def total_expansion(self) -> float:
        return (
            self.shared_expert_dim + self.num_experts * self.normal_expert_dim
        ) / self.dim

    @property
    def active_expansion(self) -> float:
        return (
            self.shared_expert_dim + self.top_k * self.normal_expert_dim
        ) / self.dim

    def _route(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(hidden_states)
        probabilities = torch.softmax(logits.float(), dim=-1)
        top_prob, top_index = torch.topk(
            probabilities, k=self.top_k, dim=-1, largest=True, sorted=True
        )
        top_prob = top_prob / top_prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        flat_index = top_index.reshape(-1, self.top_k)
        assignment_fraction = torch.nn.functional.one_hot(
            flat_index, num_classes=self.num_experts
        ).float().sum(dim=(0, 1))
        assignment_fraction = assignment_fraction / max(
            int(flat_index.shape[0]) * self.top_k, 1
        )
        mean_probability = probabilities.reshape(-1, self.num_experts).mean(dim=0)

        # Dense2MoE / Switch-style balancing term.  A perfectly balanced router
        # has value 1.0; minimizing the term discourages expert collapse.
        balance_loss = self.num_experts * torch.sum(
            assignment_fraction * mean_probability
        )
        return top_index, top_prob, probabilities, balance_loss

    def forward_with_aux(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        if hidden_states.ndim < 2 or int(original_shape[-1]) != self.dim:
            raise ValueError(
                f"Expected [...,{self.dim}] MoE input, got {tuple(original_shape)}"
            )

        flat = hidden_states.reshape(-1, self.dim)
        shared = self.shared_expert(flat)
        top_index, top_prob, probabilities, balance_loss = self._route(hidden_states)
        route_index = top_index.reshape(-1, self.top_k)
        route_prob = top_prob.reshape(-1, self.top_k)

        routed = torch.zeros_like(shared)
        counts: list[int] = []
        for expert_id, expert in enumerate(self.experts):
            positions, slots = torch.where(route_index == expert_id)
            count = int(positions.numel())
            counts.append(count)
            if count == 0:
                continue
            expert_input = flat.index_select(0, positions)
            expert_output = expert(expert_input)
            weight = route_prob[positions, slots].to(
                device=expert_output.device,
                dtype=expert_output.dtype,
            )
            contribution = expert_output * weight.unsqueeze(-1)
            routed = routed.index_add(0, positions, contribution)

        with torch.no_grad():
            probs = probabilities.reshape(-1, self.num_experts)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
            self._last_stats = SparseMoERouterStats(
                token_count=int(flat.shape[0]),
                assignment_count=int(flat.shape[0]) * self.top_k,
                expert_counts=tuple(counts),
                probability_entropy=float(entropy.item()),
                balance_loss=float(balance_loss.detach().item()),
            )

        return (shared + routed).reshape(original_shape), balance_loss

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_aux(hidden_states)
        return output

    def last_router_stats(self) -> SparseMoERouterStats | None:
        return self._last_stats


class WanTransformerBlockPromptFreeNoTimeMoE(
    WanTransformerBlockPromptFreeNoTime
):
    """Prompt-free/no-time block with Dense2MoE-style sparse FFN."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        adapter_dim: int = 128,
        shared_expert_dim: int,
        normal_expert_dim: int,
        num_experts: int = 12,
        top_k: int = 2,
        qk_norm: str = "rms_norm_across_heads",
        eps: float = 1e-6,
    ) -> None:
        active_ffn_dim = int(shared_expert_dim) + int(top_k) * int(normal_expert_dim)
        super().__init__(
            dim=dim,
            ffn_dim=active_ffn_dim,
            num_heads=num_heads,
            adapter_dim=adapter_dim,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.ffn = SparseMoEFFN(
            dim,
            shared_expert_dim=shared_expert_dim,
            normal_expert_dim=normal_expert_dim,
            num_experts=num_experts,
            top_k=top_k,
        )


class WanTransformer3DModelPromptFreeNoTimeMoE(
    WanTransformer3DModelPromptFreeNoTime
):
    """Time-folded SwiftVR DiT with sparse MoE FFNs."""

    _no_split_modules = ["WanTransformerBlockPromptFreeNoTimeMoE"]
    _repeated_blocks = ["WanTransformerBlockPromptFreeNoTimeMoE"]

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads=8,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=1536,
        num_layers=30,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        image_dim=None,
        added_kv_proj_dim=None,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        enable_swa=True,
        self_attn_window_hw=(16, 16),
        use_torch_compile=False,
        compile_mode="default",
        adapter_dim=128,
        folded_timestep=1000.0,
        time_condition_folded=True,
        shared_expert_dim=1024,
        normal_expert_dim=256,
        num_experts=12,
        top_k=2,
    ) -> None:
        expected_active = int(shared_expert_dim) + int(top_k) * int(normal_expert_dim)
        if int(ffn_dim) != expected_active:
            raise ValueError(
                "For the MoE model, ffn_dim records activated FFN width and must "
                f"equal shared_expert_dim + top_k*normal_expert_dim: "
                f"{ffn_dim} != {expected_active}"
            )
        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
            enable_swa=enable_swa,
            self_attn_window_hw=self_attn_window_hw,
            use_torch_compile=use_torch_compile,
            compile_mode=compile_mode,
            adapter_dim=adapter_dim,
            folded_timestep=folded_timestep,
            time_condition_folded=time_condition_folded,
        )

        dim = int(num_attention_heads) * int(attention_head_dim)
        self.blocks = nn.ModuleList(
            WanTransformerBlockPromptFreeNoTimeMoE(
                dim,
                num_heads=int(num_attention_heads),
                adapter_dim=int(adapter_dim),
                shared_expert_dim=int(shared_expert_dim),
                normal_expert_dim=int(normal_expert_dim),
                num_experts=int(num_experts),
                top_k=int(top_k),
                qk_norm=qk_norm,
                eps=float(eps),
            )
            for _ in range(int(num_layers))
        )

    def router_stats(self) -> list[SparseMoERouterStats | None]:
        result: list[SparseMoERouterStats | None] = []
        for raw_block in self.blocks:
            block = getattr(raw_block, "_orig_mod", raw_block)
            result.append(block.ffn.last_router_stats())
        return result
