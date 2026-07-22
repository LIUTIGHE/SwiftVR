"""Streaming one-step DiT for the prompt-free SwiftVR student.

This module mirrors :mod:`swiftvr.streaming.dit` while removing all prompt
embedding and text-conditioning inputs. Chunk boundaries, temporal RoPE
offsets, overlap blending, and last-chunk padding remain unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch

from .chunk import ChunkSpec
from .dit import INFERENCE_TIMESTEP, _rope_with_offset


def _precompute_time_condition(transformer, batch_size, timestep, output_dtype):
    """Compute the fixed timestep modulation used by every prompt-free block."""

    ts = timestep.clone()
    timestep_seq_len = None
    if ts.ndim == 2:
        timestep_seq_len = ts.shape[1]
        ts = ts.flatten()

    temb, timestep_proj = transformer.condition_embedder(
        ts,
        timestep_seq_len=timestep_seq_len,
        output_dtype=output_dtype,
    )
    timestep_proj = timestep_proj.unflatten(
        2 if timestep_seq_len is not None else 1,
        (6, -1),
    )

    if temb.shape[0] != batch_size:
        raise RuntimeError(
            f"Timestep embedding batch mismatch: expected {batch_size}, "
            f"got {temb.shape[0]}"
        )
    return temb, timestep_proj


@torch.inference_mode()
def _dit_forward_chunk_prompt_free(
    transformer,
    chunk,
    temb,
    timestep_proj,
    t_off=0,
):
    """Run one prompt-free DiT pass and return degradation velocity."""

    patch_t, patch_h, patch_w = transformer.config.patch_size
    batch_size, _, frames, height, width = chunk.shape
    patches_f = frames // patch_t
    patches_h = height // patch_h
    patches_w = width // patch_w

    rotary_emb = _rope_with_offset(
        transformer.rope,
        patches_f,
        patches_h,
        patches_w,
        t_off=t_off,
    )
    hidden_states = transformer.patch_embedding(chunk).flatten(2).transpose(1, 2)

    thw_global = (patches_f, patches_h, patches_w)
    for block in transformer.blocks:
        underlying = getattr(block, "_orig_mod", block)
        if hasattr(underlying, "attn1"):
            underlying.attn1._thw = thw_global

    for block in transformer.blocks:
        hidden_states = block(hidden_states, timestep_proj, rotary_emb)

    if temb.ndim == 3:
        shift, scale = (
            transformer.scale_shift_table.unsqueeze(0).to(temb.device)
            + temb.unsqueeze(2)
        ).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        shift, scale = (
            transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
        ).chunk(2, dim=1)

    hidden_states = (
        transformer.norm_out(hidden_states.float())
        * (1 + scale.to(hidden_states.device))
        + shift.to(hidden_states.device)
    ).type_as(hidden_states)
    hidden_states = transformer.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(
        batch_size,
        patches_f,
        patches_h,
        patches_w,
        patch_t,
        patch_h,
        patch_w,
        -1,
    ).permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


class StreamingDiTPromptFree:
    """Prompt-free one-step DiT with temporal overlap blending."""

    def __init__(self, transformer, overlap=0):
        self.transformer = transformer
        self.overlap = overlap
        self._prev_lq = None
        self._prev_out = None
        self._g_off = 0
        self._cond_cache_key = None
        self._cond_cache = None

    def reset(self):
        self._prev_lq = None
        self._prev_out = None
        self._g_off = 0

    def _get_cached_condition(self, batch_size, device, dtype):
        cache_key = (int(batch_size), device.type, device.index, str(dtype))
        if self._cond_cache_key != cache_key or self._cond_cache is None:
            timestep = torch.full(
                (batch_size,),
                INFERENCE_TIMESTEP,
                device=device,
                dtype=torch.float32,
            )
            self._cond_cache = _precompute_time_condition(
                self.transformer,
                batch_size,
                timestep,
                output_dtype=dtype,
            )
            self._cond_cache_key = cache_key
        return self._cond_cache

    @torch.inference_mode()
    def denoise(self, lq):
        """Restore one latent chunk with shape ``[B, C, F, H, W]``."""

        device, dtype = lq.device, lq.dtype
        batch_size, _, frames_cur, _, _ = lq.shape

        overlap_len = 0
        if self._prev_lq is not None and self.overlap > 0:
            overlap_len = self._prev_lq.shape[2]
            lq_ext = torch.cat([self._prev_lq.to(device), lq], dim=2)
            t_rope = self._g_off - overlap_len
        else:
            lq_ext = lq
            t_rope = self._g_off

        temb, timestep_proj = self._get_cached_condition(
            batch_size,
            device,
            dtype,
        )
        pred = _dit_forward_chunk_prompt_free(
            self.transformer,
            lq_ext,
            temb,
            timestep_proj,
            t_off=t_rope,
        )
        den_ext = lq_ext - pred

        if overlap_len > 0 and self._prev_out is not None:
            ramp = torch.linspace(
                0,
                1,
                overlap_len,
                device=device,
                dtype=dtype,
            ).view(1, 1, overlap_len, 1, 1)
            den_ext[:, :, :overlap_len] = (
                self._prev_out.to(device) * (1 - ramp)
                + den_ext[:, :, :overlap_len] * ramp
            )
            den_out = den_ext[:, :, overlap_len:]
        else:
            den_out = den_ext

        keep = min(self.overlap, frames_cur)
        if keep > 0:
            self._prev_lq = lq[:, :, -keep:].detach().cpu().clone()
            self._prev_out = den_out[:, :, -keep:].detach().cpu().clone()
        else:
            self._prev_lq = None
            self._prev_out = None

        self._g_off += frames_cur
        return den_out

    @torch.inference_mode()
    def denoise_last_chunk(
        self,
        z_new_ntchw,
        spec: ChunkSpec,
        prev_dit_out_cpu: Optional[torch.Tensor],
        n_lat: int,
        device,
        dtype,
    ):
        """Restore the final fixed-protocol chunk without prompt conditioning."""

        lat_count = spec.b + 1
        pad_count = (n_lat + 1) - lat_count

        z_bcfhw = z_new_ntchw.permute(0, 2, 1, 3, 4).contiguous()
        if pad_count > 0:
            if prev_dit_out_cpu is not None:
                pad_z = prev_dit_out_cpu[:, :, -pad_count:].to(
                    device=device,
                    dtype=dtype,
                )
            else:
                pad_z = torch.zeros(
                    z_bcfhw.shape[0],
                    z_bcfhw.shape[1],
                    pad_count,
                    z_bcfhw.shape[3],
                    z_bcfhw.shape[4],
                    device=device,
                    dtype=dtype,
                )
            z_bcfhw = torch.cat([pad_z, z_bcfhw], dim=2)

        t_off = max(0, self._g_off - pad_count)
        temb, timestep_proj = self._get_cached_condition(
            z_bcfhw.shape[0],
            device,
            dtype,
        )
        pred = _dit_forward_chunk_prompt_free(
            self.transformer,
            z_bcfhw,
            temb,
            timestep_proj,
            t_off=t_off,
        )
        z_den = (z_bcfhw - pred)[:, :, -lat_count:].contiguous()

        self._g_off += lat_count
        return z_den.permute(0, 2, 1, 3, 4).contiguous()
