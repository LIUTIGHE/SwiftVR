"""Differentiable whole-clip training forward for SwiftVR.

The public streaming wrappers are intentionally inference-only: they use
``inference_mode`` and detach cross-chunk boundary states. This module provides a
separate training path that preserves gradients through ReAE, the prompt-free
fixed-time DiT, and the decoded pixel-space prediction.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.reae import MemBlock, TGrow, TPool


_INTERP_NEEDS_ALIGN = {"linear", "bilinear", "bicubic", "trilinear"}


def _validate_video(name: str, video: torch.Tensor) -> None:
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(video).__name__}")
    if video.ndim != 5:
        raise ValueError(
            f"{name} must have shape [B,T,C,H,W], got {tuple(video.shape)}"
        )
    if video.shape[2] != 3:
        raise ValueError(f"{name} must have 3 RGB channels, got C={video.shape[2]}")
    if not video.is_floating_point():
        raise TypeError(f"{name} must be floating point in [0,1], got {video.dtype}")


def _resize_btchw(
    video: torch.Tensor,
    size: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    if tuple(video.shape[-2:]) == tuple(size):
        return video

    batch, frames, channels, height, width = video.shape
    flat = video.reshape(batch * frames, channels, height, width)
    if mode in _INTERP_NEEDS_ALIGN:
        flat = F.interpolate(flat, size=size, mode=mode, align_corners=False)
    else:
        flat = F.interpolate(flat, size=size, mode=mode)
    return flat.reshape(batch, frames, channels, *size)


def prepare_training_batch(
    batch: Mapping[str, torch.Tensor],
    *,
    input_key: str = "lr",
    target_key: str = "hr",
    auxiliary_key: Optional[str] = "hq",
    upscale_mode: str = "bilinear",
) -> dict[str, Optional[torch.Tensor]]:
    """Align an LR/HQ/HR batch to deployment-time output geometry.

    SwiftVR inference resizes the low-quality input to the requested output
    resolution before ReAE encoding. Training mirrors that behavior: the LR
    input and optional HQ reference are resized to the spatial size of the
    selected target while temporal alignment is preserved.

    The default target is ``hr``. The clean 720p ``hq`` sequence is retained as
    an aligned auxiliary reference but is not included in the base loss.
    """

    if input_key not in batch:
        raise KeyError(f"Missing input tensor {input_key!r}")
    if target_key not in batch:
        raise KeyError(f"Missing target tensor {target_key!r}")

    lq = batch[input_key]
    target = batch[target_key]
    _validate_video(input_key, lq)
    _validate_video(target_key, target)

    if lq.shape[:3] != target.shape[:3]:
        raise ValueError(
            f"{input_key} and {target_key} must share [B,T,C], got "
            f"{tuple(lq.shape[:3])} vs {tuple(target.shape[:3])}"
        )

    target_size = (int(target.shape[-2]), int(target.shape[-1]))
    lq_input = _resize_btchw(lq, target_size, upscale_mode)

    hq_reference = None
    if auxiliary_key is not None and auxiliary_key in batch:
        hq_reference = batch[auxiliary_key]
        _validate_video(auxiliary_key, hq_reference)
        if hq_reference.shape[:3] != target.shape[:3]:
            raise ValueError(
                f"{auxiliary_key} and {target_key} must share [B,T,C], got "
                f"{tuple(hq_reference.shape[:3])} vs {tuple(target.shape[:3])}"
            )
        hq_reference = _resize_btchw(hq_reference, target_size, upscale_mode)

    return {
        "lq_input": lq_input,
        "target": target,
        "hq_reference": hq_reference,
    }


def _apply_reae_whole_clip(
    model: nn.Sequential,
    video: torch.Tensor,
) -> torch.Tensor:
    """Apply a ReAE stack to a whole clip without detaching temporal history."""

    batch, frames, channels, height, width = video.shape
    hidden = video.reshape(batch * frames, channels, height, width)

    for index, layer in enumerate(model):
        if isinstance(layer, MemBlock):
            _, channels, height, width = hidden.shape
            current = hidden.reshape(batch, frames, channels, height, width)
            past = torch.cat(
                [torch.zeros_like(current[:, :1]), current[:, :-1]],
                dim=1,
            )
            hidden = layer(
                hidden,
                past.reshape(batch * frames, channels, height, width),
            )
        elif isinstance(layer, TPool):
            if frames % layer.stride != 0:
                raise ValueError(
                    f"ReAE TPool at layer {index} requires T divisible by "
                    f"{layer.stride}, got T={frames}"
                )
            hidden = layer(hidden)
        elif isinstance(layer, TGrow):
            hidden = layer(hidden)
        else:
            hidden = layer(hidden)

        if hidden.shape[0] % batch != 0:
            raise RuntimeError(
                f"ReAE layer {index} produced invalid leading dimension "
                f"{hidden.shape[0]} for batch={batch}"
            )
        frames = hidden.shape[0] // batch

    _, channels, height, width = hidden.shape
    return hidden.reshape(batch, frames, channels, height, width)


def encode_reae_clip(
    reae: nn.Module,
    pixels: torch.Tensor,
    *,
    require_4k_plus_1: bool = True,
) -> torch.Tensor:
    """Encode ``[B,T,3,H,W]`` to differentiable ``[B,F,C,h,w]`` latents.

    For the deployment-compatible ``T=4k+1`` protocol, three copies of the last
    frame are appended before the two stride-2 temporal pooling layers. A
    17-frame pixel clip therefore becomes a 5-frame latent clip.
    """

    _validate_video("pixels", pixels)
    batch, frames, channels, height, width = pixels.shape

    if require_4k_plus_1 and frames % 4 != 1:
        raise ValueError(f"SwiftVR training clips must satisfy T=4k+1, got T={frames}")

    patch_size = int(getattr(reae, "patch_size", 1))
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Pixel size {height}x{width} must be divisible by "
            f"ReAE patch_size={patch_size}"
        )

    if patch_size > 1:
        flat = F.pixel_unshuffle(
            pixels.reshape(batch * frames, channels, height, width),
            patch_size,
        )
        pixels = flat.reshape(batch, frames, *flat.shape[1:])

    temporal_pad = (-frames) % 4
    if temporal_pad:
        pixels = torch.cat(
            [
                pixels,
                pixels[:, -1:].expand(-1, temporal_pad, -1, -1, -1),
            ],
            dim=1,
        )

    return _apply_reae_whole_clip(reae.encoder, pixels)


def decode_reae_clip(
    reae: nn.Module,
    latents: torch.Tensor,
    *,
    output_frames: Optional[int] = None,
    clamp: bool = False,
) -> torch.Tensor:
    """Decode ``[B,F,C,h,w]`` while preserving causal warm-up semantics."""

    if latents.ndim != 5:
        raise ValueError(
            f"latents must have shape [B,F,C,H,W], got {tuple(latents.shape)}"
        )

    pixels = _apply_reae_whole_clip(reae.decoder, latents)

    patch_size = int(getattr(reae, "patch_size", 1))
    if patch_size > 1:
        batch, frames, channels, height, width = pixels.shape
        flat = F.pixel_shuffle(
            pixels.reshape(batch * frames, channels, height, width),
            patch_size,
        )
        pixels = flat.reshape(batch, frames, *flat.shape[1:])

    trim = int(getattr(reae, "frames_to_trim", 0))
    if trim:
        pixels = pixels[:, trim:]

    if output_frames is not None:
        if pixels.shape[1] < output_frames:
            raise RuntimeError(
                f"ReAE decoder emitted {pixels.shape[1]} valid frames; "
                f"requested {output_frames}"
            )
        pixels = pixels[:, :output_frames]

    return pixels.clamp(0, 1) if clamp else pixels


def _config_value(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, Mapping):
        return config.get(key, default)
    return default


def _transformer_patch_size(transformer: nn.Module) -> tuple[int, int, int]:
    patch_size = _config_value(
        getattr(transformer, "config", None),
        "patch_size",
        (1, 1, 1),
    )
    if isinstance(patch_size, int):
        return int(patch_size), int(patch_size), int(patch_size)
    if len(patch_size) != 3:
        raise ValueError(
            f"Transformer patch_size must have 3 values, got {patch_size!r}"
        )
    return tuple(int(value) for value in patch_size)


def required_pixel_multiple(
    reae: nn.Module,
    transformer: nn.Module,
) -> tuple[int, int]:
    """Return the pixel-space multiple required by ReAE and DiT patching."""

    # ReAE pixel-unshuffle contributes ``patch_size`` and the encoder contains
    # three stride-2 spatial convolutions.
    reae_factor = int(getattr(reae, "patch_size", 1)) * 8
    _, patch_h, patch_w = _transformer_patch_size(transformer)
    return reae_factor * patch_h, reae_factor * patch_w


class WanShiftWindow2DTrainProcessor:
    """Autograd-safe mask-free shifted-window attention for training.

    The inference processor aggressively reuses in-place RoPE and may release
    CUDA input storage. This processor keeps the same window geometry but uses
    out-of-place RoPE, never frees tensors needed by backward, and deliberately
    leaves Q/K/V projections unfused so saved checkpoints retain their canonical
    parameter layout.
    """

    def __init__(self, window_hw: tuple[int, int] = (16, 16)):
        from ..models import transformer as transformer_ops

        wh, ww = window_hw
        if wh <= 0 or ww <= 0:
            raise ValueError(f"window_hw must be positive, got {window_hw!r}")
        self.window_hw = (int(wh), int(ww))
        self._ops = transformer_ops

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ):
        if encoder_hidden_states is not None or getattr(
            attn, "is_cross_attention", False
        ):
            raise RuntimeError(
                "WanShiftWindow2DTrainProcessor only supports self-attention"
            )
        if attention_mask is not None:
            raise RuntimeError("External attention_mask is not supported")
        if not hasattr(attn, "_thw") or attn._thw is None:
            raise RuntimeError("attn._thw=(T,H,W) must be set before forward")

        tg, hg, wg = attn._thw
        batch, tokens, _ = hidden_states.shape
        frames, height, width = self._ops._infer_local_thw(
            (tg, hg, wg),
            tokens,
        )
        if tokens != frames * height * width:
            raise RuntimeError(
                f"Token mismatch: K={tokens}, inferred "
                f"T*H*W={frames * height * width}"
            )

        cfg_h, cfg_w = self.window_hw
        win_h = min(cfg_h, height)
        win_w = min(cfg_w, width)
        do_shift = bool(getattr(attn, "_do_shift", False))
        prefer_front = not do_shift

        meta = self._ops._WindowRuntimeMetaCache.get(
            frames,
            height,
            width,
            win_h,
            win_w,
            do_shift=do_shift,
            prefer_front=prefer_front,
            device=hidden_states.device,
        )
        num_windows, window_tokens = meta.Nw, meta.Lw
        heads = attn.heads
        head_dim = attn.inner_dim // heads

        query, key, value = self._ops._get_qkv_projections(
            attn,
            hidden_states,
            None,
        )
        query = attn.norm_q(query).unflatten(2, (heads, head_dim))
        key = attn.norm_k(key).unflatten(2, (heads, head_dim))
        value = value.unflatten(2, (heads, head_dim))

        if rotary_emb is not None:
            query = self._ops._apply_rotary_emb(query, *rotary_emb)
            key = self._ops._apply_rotary_emb(key, *rotary_emb)

        query = torch.index_select(query, 1, meta.lin_flat).view(
            batch * num_windows,
            window_tokens,
            heads,
            head_dim,
        )
        key = torch.index_select(key, 1, meta.lin_flat).view(
            batch * num_windows,
            window_tokens,
            heads,
            head_dim,
        )
        value = torch.index_select(value, 1, meta.lin_flat).view(
            batch * num_windows,
            window_tokens,
            heads,
            head_dim,
        )

        output_windows = self._ops._dense_attn(query, key, value)
        output_flat = output_windows.reshape(
            batch,
            num_windows * window_tokens,
            heads,
            head_dim,
        )
        output = torch.index_select(output_flat, 1, meta.owner_pos)
        output = output.reshape(batch, tokens, heads * head_dim)
        output = attn.to_out[0](output)
        if (
            attn.training
            and isinstance(attn.to_out[1], nn.Dropout)
            and attn.to_out[1].p > 0.0
        ):
            output = attn.to_out[1](output)
        return output


def prepare_prompt_free_no_time_transformer_for_training(
    transformer: nn.Module,
    *,
    attention_backend: str = "sdpa",
    window_hw: Optional[tuple[int, int]] = None,
) -> str:
    """Install training-safe MFSWA without changing checkpoint parameter layout."""

    from ..models import transformer as transformer_ops

    backend = transformer_ops.set_attention_backend(attention_backend)
    if window_hw is None:
        window_hw = tuple(
            int(value)
            for value in getattr(
                transformer,
                "_self_attn_window_hw",
                (16, 16),
            )
        )

    processor = WanShiftWindow2DTrainProcessor(window_hw=window_hw)
    for index, block in enumerate(getattr(transformer, "blocks", [])):
        underlying = getattr(block, "_orig_mod", block)
        if not hasattr(underlying, "attn1"):
            raise TypeError(
                f"Transformer block {index} does not expose prompt-free self-attention"
            )
        attention = underlying.attn1
        attention._do_shift = bool(index % 2 == 1)
        if getattr(attention, "fused_projections", False):
            attention.unfuse_projections()
        attention.set_processor(processor)

    transformer_ops._WindowIndexCache.clear()
    transformer_ops._WindowRuntimeMetaCache.clear()
    transformer.train()
    return backend


def _forward_prompt_free_no_time_block_training(
    block: nn.Module,
    hidden_states: torch.Tensor,
    rotary_emb,
) -> torch.Tensor:
    hidden_dtype = hidden_states.dtype
    mods = block.scale_shift_table.to(dtype=hidden_dtype)
    (
        shift_msa,
        scale_msa,
        gate_msa,
        shift_ffn,
        scale_ffn,
        gate_ffn,
    ) = mods.chunk(6, dim=1)

    attention_input = block.norm1(hidden_states)
    attention_input = attention_input * (1.0 + scale_msa) + shift_msa
    attention_output = block.attn1(
        attention_input,
        None,
        None,
        rotary_emb,
    )
    hidden_states = hidden_states + attention_output * gate_msa

    adapter_output = block.prompt_free_adapter(hidden_states)
    hidden_states = hidden_states + adapter_output

    ffn_input = block.norm3(hidden_states)
    ffn_input = ffn_input * (1.0 + scale_ffn) + shift_ffn
    ffn_output = block.ffn(ffn_input)
    return hidden_states + ffn_output * gate_ffn


def forward_prompt_free_no_time_training(
    transformer: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Run the folded prompt-free DiT with an autograd-safe functional forward."""

    if hidden_states.ndim != 5:
        raise ValueError(
            "Transformer input must have shape [B,C,F,H,W], got "
            f"{tuple(hidden_states.shape)}"
        )

    batch, _, frames, height, width = hidden_states.shape
    patch_t, patch_h, patch_w = _transformer_patch_size(transformer)
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"Latent size {frames}x{height}x{width} must be divisible by "
            f"transformer patch_size={(patch_t, patch_h, patch_w)}"
        )

    patched_frames = frames // patch_t
    patched_height = height // patch_h
    patched_width = width // patch_w
    rotary_emb = transformer.rope(hidden_states)

    tokens = (
        transformer.patch_embedding(hidden_states)
        .flatten(2)
        .transpose(1, 2)
        .contiguous()
    )
    thw = (patched_frames, patched_height, patched_width)

    for index, block in enumerate(transformer.blocks):
        underlying = getattr(block, "_orig_mod", block)
        if not all(
            hasattr(underlying, name)
            for name in (
                "attn1",
                "norm1",
                "norm3",
                "prompt_free_adapter",
                "ffn",
                "scale_shift_table",
            )
        ):
            raise TypeError(
                f"Block {index} is not compatible with the prompt-free no-time "
                "training forward"
            )
        underlying.attn1._thw = thw
        tokens = _forward_prompt_free_no_time_block_training(
            underlying,
            tokens,
            rotary_emb,
        )

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
    return tokens.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def _extract_transformer_sample(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, Sequence) and output:
        return output[0]
    raise TypeError(
        "Transformer output must be a tensor, expose .sample, or be a "
        "non-empty sequence"
    )


class SwiftVRTrainingForward(nn.Module):
    """Minimal fixed-time, prompt-free SwiftVR training closure.

    This implements the deployment endpoint used by Stage 3:

    ``z_lq = E(x_lq)``, ``velocity = v(z_lq, 1)``,
    ``z_prediction = z_lq - velocity``, ``prediction = D(z_prediction)``.

    Pixel L1 is the default loss. An optional endpoint latent-velocity MSE can
    be enabled, but it is not the paper's uniformly sampled Stage-1 objective
    because the folded student deliberately has no runtime timestep input.
    """

    def __init__(
        self,
        reae: nn.Module,
        transformer: nn.Module,
        *,
        input_key: str = "lr",
        target_key: str = "hr",
        auxiliary_key: Optional[str] = "hq",
        upscale_mode: str = "bilinear",
        pixel_loss_weight: float = 1.0,
        latent_loss_weight: float = 0.0,
        detach_velocity_target: bool = True,
        clamp_prediction_for_loss: bool = False,
        require_4k_plus_1: bool = True,
        validate_spatial_multiple: bool = True,
        training_safe_transformer: bool = True,
        prepare_transformer: bool = True,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        self.reae = reae
        self.transformer = transformer
        self.input_key = input_key
        self.target_key = target_key
        self.auxiliary_key = auxiliary_key
        self.upscale_mode = upscale_mode
        self.pixel_loss_weight = float(pixel_loss_weight)
        self.latent_loss_weight = float(latent_loss_weight)
        self.detach_velocity_target = bool(detach_velocity_target)
        self.clamp_prediction_for_loss = bool(clamp_prediction_for_loss)
        self.require_4k_plus_1 = bool(require_4k_plus_1)
        self.validate_spatial_multiple = bool(validate_spatial_multiple)
        self.training_safe_transformer = bool(training_safe_transformer)

        if self.training_safe_transformer and prepare_transformer:
            prepare_prompt_free_no_time_transformer_for_training(
                self.transformer,
                attention_backend=attention_backend,
            )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, object]:
        prepared = prepare_training_batch(
            batch,
            input_key=self.input_key,
            target_key=self.target_key,
            auxiliary_key=self.auxiliary_key,
            upscale_mode=self.upscale_mode,
        )
        lq_input = prepared["lq_input"]
        target = prepared["target"]
        if not isinstance(lq_input, torch.Tensor) or not isinstance(
            target, torch.Tensor
        ):
            raise RuntimeError("Prepared training batch did not contain tensors")

        if self.validate_spatial_multiple:
            multiple_h, multiple_w = required_pixel_multiple(
                self.reae,
                self.transformer,
            )
            height, width = target.shape[-2:]
            if height % multiple_h or width % multiple_w:
                raise ValueError(
                    f"Target size {height}x{width} must be divisible by "
                    f"{multiple_h}x{multiple_w} for ReAE + DiT patching"
                )

        z_lq_ntchw = encode_reae_clip(
            self.reae,
            lq_input,
            require_4k_plus_1=self.require_4k_plus_1,
        )
        z_lq = z_lq_ntchw.permute(0, 2, 1, 3, 4).contiguous()

        expected_channels = _config_value(
            getattr(self.transformer, "config", None),
            "in_channels",
            None,
        )
        if expected_channels is not None and z_lq.shape[1] != int(
            expected_channels
        ):
            raise ValueError(
                f"ReAE emitted {z_lq.shape[1]} latent channels but transformer "
                f"expects {expected_channels}"
            )

        if self.training_safe_transformer:
            velocity = forward_prompt_free_no_time_training(
                self.transformer,
                z_lq,
            )
        else:
            velocity = _extract_transformer_sample(self.transformer(z_lq))

        if velocity.shape != z_lq.shape:
            raise ValueError(
                f"Transformer velocity shape {tuple(velocity.shape)} does not "
                f"match input latent shape {tuple(z_lq.shape)}"
            )

        z_prediction = z_lq - velocity
        prediction_raw = decode_reae_clip(
            self.reae,
            z_prediction.permute(0, 2, 1, 3, 4).contiguous(),
            output_frames=target.shape[1],
            clamp=False,
        )
        prediction_clamped = prediction_raw.clamp(0, 1)
        prediction_for_loss = (
            prediction_clamped
            if self.clamp_prediction_for_loss
            else prediction_raw
        )
        pixel_l1 = F.l1_loss(prediction_for_loss, target)

        z_target = None
        velocity_target = None
        latent_velocity_mse = pixel_l1.new_zeros(())
        if self.latent_loss_weight != 0.0:
            z_target_ntchw = encode_reae_clip(
                self.reae,
                target,
                require_4k_plus_1=self.require_4k_plus_1,
            )
            z_target = z_target_ntchw.permute(0, 2, 1, 3, 4).contiguous()
            velocity_target = z_lq - z_target
            if self.detach_velocity_target:
                velocity_target = velocity_target.detach()
            latent_velocity_mse = F.mse_loss(velocity, velocity_target)

        loss = (
            self.pixel_loss_weight * pixel_l1
            + self.latent_loss_weight * latent_velocity_mse
        )

        return {
            "loss": loss,
            "pixel_l1": pixel_l1,
            "latent_velocity_mse": latent_velocity_mse,
            "prediction": prediction_raw,
            "prediction_clamped": prediction_clamped,
            "target": target,
            "lq_input": lq_input,
            "hq_reference": prepared["hq_reference"],
            "velocity": velocity,
            "velocity_target": velocity_target,
            "z_lq": z_lq,
            "z_target": z_target,
            "z_prediction": z_prediction,
        }
