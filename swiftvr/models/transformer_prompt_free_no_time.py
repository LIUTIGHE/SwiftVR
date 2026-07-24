"""Prompt-free SwiftVR DiT with the fixed timestep condition folded away.

SwiftVR performs one-step inference at a constant timestep. The corresponding
per-block scale/shift/gate modulation and final output modulation can therefore
be precomputed once and added to the learned ``scale_shift_table`` tensors.
This module consumes such a folded checkpoint and contains no timestep
embedding or timestep input at runtime.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
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
from .transformer_prompt_free import PromptFreeResidualAdapter


logger = logging.get_logger(__name__)


@maybe_allow_in_graph
class WanTransformerBlockPromptFreeNoTime(nn.Module):
    """Prompt-free block whose fixed timestep modulation is already folded."""

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
        del qk_norm

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

        # In a folded checkpoint this table already contains the constant
        # timestep projection in addition to the original learned table.
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, hidden_states: torch.Tensor, rotary_emb) -> torch.Tensor:
        hidden_dtype = hidden_states.dtype
        mods = self.scale_shift_table.to(dtype=hidden_dtype)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_ffn,
            scale_ffn,
            gate_ffn,
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

        ffn_output = self.ffn(
            self.norm3(hidden_states).mul_(1.0 + scale_ffn).add_(shift_ffn)
        )
        hidden_states.addcmul_(ffn_output, gate_ffn)
        del ffn_output
        return hidden_states


def compile_prompt_free_no_time_blocks(model: nn.Module, mode: str = "default") -> None:
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available (requires PyTorch 2.0+). Skipping.")
        return
    if mode == "reduce-overhead":
        logger.warning(
            "compile_mode='reduce-overhead' is incompatible with the in-place "
            "residuals; falling back to 'default'."
        )
        mode = "default"

    for index, block in enumerate(getattr(model, "blocks", [])):
        if isinstance(block, WanTransformerBlockPromptFreeNoTime):
            model.blocks[index] = torch.compile(block, mode=mode, fullgraph=False)


class WanTransformer3DModelPromptFreeNoTime(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    CacheMixin,
    AttentionMixin,
):
    """Prompt-free SwiftVR DiT with no runtime timestep modules or input."""

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "norm"]
    _no_split_modules = ["WanTransformerBlockPromptFreeNoTime"]
    _keep_in_fp32_modules = [
        "scale_shift_table",
        "norm1",
        "norm3",
        "prompt_free_adapter.norm",
    ]
    _repeated_blocks = ["WanTransformerBlockPromptFreeNoTime"]

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
        folded_timestep=1000.0,
        time_condition_folded=True,
    ):
        super().__init__()

        # Keep the source configuration fields load-compatible while avoiding
        # construction of any text/image/time conditioning modules.
        del (
            text_dim,
            freq_dim,
            cross_attn_norm,
            image_dim,
            added_kv_proj_dim,
            pos_embed_seq_len,
            folded_timestep,
        )
        if not time_condition_folded:
            raise ValueError(
                "WanTransformer3DModelPromptFreeNoTime requires a checkpoint "
                "with time_condition_folded=true"
            )

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
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlockPromptFreeNoTime(
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

        # In a folded checkpoint both rows already include the fixed temb.
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
            compile_prompt_free_no_time_blocks(self, mode=compile_mode)
        _WindowIndexCache.clear()
        _WindowRuntimeMetaCache.clear()
        self.eval()

    def forward(
        self,
        hidden_states: torch.Tensor,
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
            hidden_states = block(hidden_states, rotary_emb)

        hidden_dtype = hidden_states.dtype
        shift, scale = self.scale_shift_table.to(hidden_dtype).chunk(2, dim=1)
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
