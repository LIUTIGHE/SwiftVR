"""Streaming DiT for the time-folded prompt-free SwiftVR student.

The fixed one-step timestep modulation is stored in the transformer's
``scale_shift_table`` tensors. Runtime therefore needs no timestep tensor,
condition embedder, or condition cache.
"""

from __future__ import annotations

from typing import Optional

import torch

from .chunk import ChunkSpec
from .dit import _rope_with_offset


@torch.inference_mode()
def _dit_forward_chunk_prompt_free_no_time(
    transformer,
    chunk,
    t_off=0,
):
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
        hidden_states = block(hidden_states, rotary_emb)

    hidden_dtype = hidden_states.dtype
    shift, scale = transformer.scale_shift_table.to(hidden_dtype).chunk(2, dim=1)
    hidden_states = transformer.norm_out(hidden_states)
    hidden_states.mul_(1.0 + scale).add_(shift)
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


class StreamingDiTPromptFreeNoTime:
    """Prompt-free one-step DiT with fixed time modulation folded away."""

    def __init__(self, transformer, overlap=0):
        self.transformer = transformer
        self.overlap = overlap
        self._prev_lq = None
        self._prev_out = None
        self._g_off = 0

    def reset(self):
        self._prev_lq = None
        self._prev_out = None
        self._g_off = 0

    @torch.inference_mode()
    def denoise(self, lq):
        device, dtype = lq.device, lq.dtype
        _, _, frames_cur, _, _ = lq.shape

        overlap_len = 0
        if self._prev_lq is not None and self.overlap > 0:
            overlap_len = self._prev_lq.shape[2]
            lq_ext = torch.cat([self._prev_lq.to(device), lq], dim=2)
            t_rope = self._g_off - overlap_len
        else:
            lq_ext = lq
            t_rope = self._g_off

        prediction = _dit_forward_chunk_prompt_free_no_time(
            self.transformer,
            lq_ext,
            t_off=t_rope,
        )
        denoised_ext = lq_ext - prediction

        if overlap_len > 0 and self._prev_out is not None:
            ramp = torch.linspace(
                0,
                1,
                overlap_len,
                device=device,
                dtype=dtype,
            ).view(1, 1, overlap_len, 1, 1)
            denoised_ext[:, :, :overlap_len] = (
                self._prev_out.to(device) * (1 - ramp)
                + denoised_ext[:, :, :overlap_len] * ramp
            )
            denoised_out = denoised_ext[:, :, overlap_len:]
        else:
            denoised_out = denoised_ext

        keep = min(self.overlap, frames_cur)
        if keep > 0:
            self._prev_lq = lq[:, :, -keep:].detach().cpu().clone()
            self._prev_out = denoised_out[:, :, -keep:].detach().cpu().clone()
        else:
            self._prev_lq = None
            self._prev_out = None

        self._g_off += frames_cur
        return denoised_out

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
        prediction = _dit_forward_chunk_prompt_free_no_time(
            self.transformer,
            z_bcfhw,
            t_off=t_off,
        )
        z_den = (z_bcfhw - prediction)[:, :, -lat_count:].contiguous()

        self._g_off += lat_count
        return z_den.permute(0, 2, 1, 3, 4).contiguous()
