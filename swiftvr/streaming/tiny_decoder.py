"""Streaming wrapper for :class:`TinyConditionalDecoder`.

The wrapper mirrors the ReAE decoder boundary-state protocol: decoder MemBlocks and
TGrow layers carry causal state across chunks, the first decoded chunk discards the
three warm-up frames, and middle chunks are emitted directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.tiny_conditional_decoder import TinyConditionalDecoder
from .tae import apply_parallel_with_boundary


class StreamingTinyConditionalDecoder:
    def __init__(self, model: TinyConditionalDecoder) -> None:
        self.model = model
        self._state = None
        self._first_decode = True

    def reset(self) -> None:
        self._state = None
        self._first_decode = True

    @torch.inference_mode()
    def decode_chunk(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one aligned chunk.

        ``latents`` use ``[B,F,C,h,w]`` and ``condition`` uses
        ``[B,T,3,H,W]``. Fixed SwiftVR middle chunks have ``T=4F``. The first
        chunk may also use ``T=4F`` but emits three fewer frames because causal
        decoder warm-up is removed.
        """

        if latents.ndim != 5 or condition.ndim != 5:
            raise ValueError("latents/condition must both be five-dimensional")
        if int(latents.shape[2]) != self.model.latent_channels:
            raise ValueError(
                f"Expected latent C={self.model.latent_channels}, got {latents.shape[2]}"
            )
        if int(condition.shape[2]) != 3:
            raise ValueError("condition must be RGB")
        if int(latents.shape[0]) != int(condition.shape[0]):
            raise ValueError("latents and condition must share batch size")

        projected = self.model.project_condition(condition)
        if tuple(projected.shape[:2]) != tuple(latents.shape[:2]) or tuple(
            projected.shape[-2:]
        ) != tuple(latents.shape[-2:]):
            raise ValueError(
                "Streaming condition/latent grid mismatch: "
                f"condition={tuple(projected.shape)}, latent={tuple(latents.shape)}"
            )
        hidden = torch.cat([latents, projected], dim=2)
        pixels, self._state = apply_parallel_with_boundary(
            self.model.decoder,
            hidden,
            self._state,
        )
        if pixels is None:
            raise RuntimeError("Tiny decoder unexpectedly buffered a latent chunk")

        batch, frames, channels, height, width = pixels.shape
        flat = F.pixel_shuffle(
            pixels.reshape(batch * frames, channels, height, width),
            self.model.patch_size,
        )
        pixels = flat.reshape(batch, frames, *flat.shape[1:]).clamp(0.0, 1.0)
        if self._first_decode and self.model.frames_to_trim:
            pixels = pixels[:, self.model.frames_to_trim :]
            self._first_decode = False
        return pixels
