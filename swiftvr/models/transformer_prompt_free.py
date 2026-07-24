"""Prompt-free SwiftVR diffusion transformer.

This module keeps SwiftVR's self-attention, timestep modulation, feed-forward
layers, RoPE, and output head unchanged while replacing each empty-prompt
cross-attention layer with a lightweight content-dependent residual adapter.

The adapter is a per-token low-rank MLP. This is a natural surrogate for the
original inference-time cross-attention because the text keys and values come
from a fixed empty-prompt embedding, while the visual query remains
content-dependent. The final projection is zero-initialized, so a newly created
student starts as the exact hard-removal baseline rather than adding a random
residual.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

from .transformer import (
    WanAttention,
    WanAttnProcessor,
    WanRotaryPosEmbed,
    _WindowIndexCache,
    _WindowRuntimeMetaCache,
    enable_shifted_window_self_attention,
    list_available_attention_backends,
    set_attention_backend,
)


logger = logging.get_logger(__name__)


class PromptFreeResidualAdapter(nn.Module):
    """Predict a content-dependent residual in place of text cross-attention.

    The adapter acts independently on each visual token. ``up`` is initialized
    to zero so the initial residual is exactly zero for every input.
    """

    def __init__(self, dim: int, bottleneck_dim: int = 128, eps: float = 1e-6):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")

        self.norm = FP32LayerNorm(dim, eps, elementwise_affine=True)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.SiLU()
        self.up = nn.Linear(bottleneck_dim, dim)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.norm(hidden_states)
        residual = self.down(residual)
        residual = self.act(residual)
        return self.up(residual)


class WanTimeEmbedding(nn.Module):
    """Timestep-only counterpart of SwiftVR's time-text-image embedder."""

    def __init__(self, dim: int, time_freq_dim: int, time_proj_dim: int):
        super().__init__()
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim,
            time_embed_dim=dim,
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        *,
        timestep_seq_len: Optional[int] = None,
        output_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        parameter_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != parameter_dtype and parameter_dtype != torch.int8:
            timestep = timestep.to(parameter_dtype)

        temb = self.time_embedder(timestep)
        timestep_proj = self.time_proj(self.act_fn(temb))

        if output_dtype is not None:
            temb = temb.to(output_dtype)
            timestep_proj = timestep_proj.to(output_dtype)
        return temb, timestep_proj


@maybe_allow_in_graph
class WanTransformerBlockPromptFree(nn.Module):
    """SwiftVR transformer block with a residual adapter instead of CA."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        adapter_dim: int = 128,
        qk_norm: str = "rms_norm_across_heads",
        eps: float = 1e-6,
    ):
        super().__init__()
        del qk_norm  # Kept in the signature for checkpoint/config compatibility.

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )
        self.prompt_free_adapter = PromptFreeResidualAdapter(
            dim=dim,
            bottleneck_dim=adapter_dim,
            eps=eps,
        )
        self.ffn = FeedForward(
            dim,
            inner_dim=ffn_dim,
            activation_fn="gelu-approximate",
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb,
    ) -> torch.Tensor:
        h_dtype = hidden_states.dtype

        if temb.ndim == 4:
            mods = (self.scale_shift_table.unsqueeze(0) + temb.float()).to(h_dtype)
            (
                shift_msa,
                scale_msa,
                gate_msa,
                c_shift_msa,
                c_scale_msa,
                c_gate_msa,
            ) = mods.chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            mods = (self.scale_shift_table + temb.float()).to(h_dtype)
            (
                shift_msa,
                scale_msa,
                gate_msa,
                c_shift_msa,
                c_scale_msa,
                c_gate_msa,
            ) = mods.chunk(6, dim=1)

        attn_output = self.attn1(
            self.norm1(hidden_states).mul_(1.0 + scale_msa).add_(shift_msa),
            None,
            None,
            rotary_emb,
        )
        hidden_states.addcmul_(attn_output, gate_msa)
        del attn_output

        adapter_output = self.prompt_free_adapter(hidden_states)
        hidden_states.add_(adapter_output)
        del adapter_output

        ff_output = self.ffn(
            self.norm3(hidden_states).mul_(1.0 + c_scale_msa).add_(c_shift_msa)
        )
        hidden_states.addcmul_(ff_output, c_gate_msa)
        del ff_output

        return hidden_states


def compile_prompt_free_transformer_blocks(model: nn.Module, mode: str = "default") -> None:
    """Compile prompt-free blocks without changing the original helper."""

    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available (requires PyTorch 2.0+). Skipping.")
        return
    if mode == "reduce-overhead":
        logger.warning(
            "compile_mode='reduce-overhead' is incompatible with the in-place "
            "residuals; falling back to 'default'."
        )
        mode = "default"

    for i, block in enumerate(getattr(model, "blocks", [])):
        if isinstance(block, WanTransformerBlockPromptFree):
            model.blocks[i] = torch.compile(block, mode=mode, fullgraph=False)


class WanTransformer3DModelPromptFree(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    CacheMixin,
    AttentionMixin,
):
    """Prompt-free SwiftVR DiT with low-rank CA-surrogate adapters."""

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlockPromptFree"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm3",
        "prompt_free_adapter.norm",
    ]
    _repeated_blocks = ["WanTransformerBlockPromptFree"]

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads=40,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=13824,
        num_layers=40,
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
    ):
        super().__init__()

        # Retain source config fields so an original SwiftVR config can be reused.
        # They intentionally do not instantiate text/image conditioning modules.
        del text_dim, cross_attn_norm, image_dim, added_kv_proj_dim, pos_embed_seq_len

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.rope = WanRotaryPosEmbed(
            attention_head_dim,
            patch_size,
            rope_max_seq_len,
        )
        self.patch_embedding = nn.Conv3d(
            in_channels,
            inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.condition_embedder = WanTimeEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
        )
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlockPromptFree(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    adapter_dim=adapter_dim,
                    qk_norm=qk_norm,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False
        self._enable_swa = enable_swa
        self._self_attn_window_hw = self_attn_window_hw
        self._use_torch_compile = use_torch_compile
        self._compile_mode = compile_mode

    def prepare_for_inference(
        self,
        attention_backend: str = "auto",
        use_torch_compile: bool = False,
        compile_mode: str = "default",
    ) -> None:
        backend = set_attention_backend(attention_backend)
        logger.info(
            f"Using attention backend: {backend} "
            f"(available: {list_available_attention_backends()})"
        )
        enable_shifted_window_self_attention(
            self,
            window_hw=self._self_attn_window_hw,
        )
        if use_torch_compile:
            compile_prompt_free_transformer_blocks(self, mode=compile_mode)
        _WindowIndexCache.clear()
        _WindowRuntimeMetaCache.clear()
        self.eval()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        return_dict: bool = True,
        attention_kwargs=None,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, _, frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = self.config.patch_size
        patches_f = frames // patch_t
        patches_h = height // patch_h
        patches_w = width // patch_w

        rotary_emb = self.rope(hidden_states)
        hidden_states = (
            self.patch_embedding(hidden_states)
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        timestep_seq_len = None
        if timestep.ndim == 2:
            timestep_seq_len = timestep.shape[1]
            timestep = timestep.flatten()

        temb, timestep_proj = self.condition_embedder(
            timestep,
            timestep_seq_len=timestep_seq_len,
            output_dtype=hidden_states.dtype,
        )
        timestep_proj = (
            timestep_proj.unflatten(2, (6, -1))
            if timestep_seq_len is not None
            else timestep_proj.unflatten(1, (6, -1))
        )

        thw_global = (patches_f, patches_h, patches_w)
        cfg_h, cfg_w = self._self_attn_window_hw
        device = hidden_states.device
        _WindowRuntimeMetaCache.get(
            patches_f,
            patches_h,
            patches_w,
            min(cfg_h, patches_h),
            min(cfg_w, patches_w),
            do_shift=False,
            prefer_front=True,
            device=device,
        )
        _WindowRuntimeMetaCache.get(
            patches_f,
            patches_h,
            patches_w,
            min(cfg_h, patches_h),
            min(cfg_w, patches_w),
            do_shift=True,
            prefer_front=False,
            device=device,
        )

        for block in self.blocks:
            underlying = getattr(block, "_orig_mod", block)
            underlying.attn1._thw = thw_global

        for block in self.blocks:
            hidden_states = block(hidden_states, timestep_proj, rotary_emb)

        h_dtype = hidden_states.dtype
        if temb.ndim == 3:
            mods = (
                self.scale_shift_table.unsqueeze(0).to(temb.device)
                + temb.unsqueeze(2)
            ).to(h_dtype)
            shift, scale = mods.chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            mods = (
                self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
            ).to(h_dtype)
            shift, scale = mods.chunk(2, dim=1)

        normed = self.norm_out(hidden_states)
        normed.mul_(1.0 + scale).add_(shift)
        hidden_states = self.proj_out(normed)
        del normed

        hidden_states = hidden_states.reshape(
            batch_size,
            patches_f,
            patches_h,
            patches_w,
            patch_t,
            patch_h,
            patch_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
