"""High-level pipeline for the time-folded prompt-free SwiftVR student."""

from __future__ import annotations

from pathlib import Path

import torch

from .models import ReAE
from .models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from .pipeline import SwiftVRPipeline, _as_dtype
from .streaming.dit_prompt_free_no_time import StreamingDiTPromptFreeNoTime


class _RunnerCompatiblePromptFreeNoTimeDiT(StreamingDiTPromptFreeNoTime):
    """Ignore the legacy runner's prompt argument without changing runner.py."""

    @torch.inference_mode()
    def denoise(self, lq, prompt_emb=None):
        del prompt_emb
        return super().denoise(lq)

    @torch.inference_mode()
    def denoise_last_chunk(
        self,
        z_new_ntchw,
        spec,
        prompt_emb,
        prev_dit_out_cpu,
        n_lat,
        device,
        dtype,
    ):
        del prompt_emb
        return super().denoise_last_chunk(
            z_new_ntchw,
            spec,
            prev_dit_out_cpu,
            n_lat,
            device,
            dtype,
        )


class SwiftVRPromptFreeNoTimePipeline(SwiftVRPipeline):
    """SwiftVR pipeline with both text and runtime timestep modules removed."""

    def __init__(self, reae, transformer, upscale_mode: str = "bilinear"):
        super().__init__(
            reae=reae,
            transformer=transformer,
            prompt_emb=None,
            upscale_mode=upscale_mode,
        )
        self.dit_stream = _RunnerCompatiblePromptFreeNoTimeDiT(
            transformer,
            overlap=0,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir,
        *,
        reae_filename: str = "reae.safetensors",
        transformer_subfolder: str = "transformer",
        upscale_mode: str = "bilinear",
        device=None,
        dtype=None,
    ) -> "SwiftVRPromptFreeNoTimePipeline":
        root = Path(checkpoint_dir)
        reae = ReAE(str(root / reae_filename))

        load_kwargs = {"subfolder": transformer_subfolder}
        if dtype is not None:
            load_kwargs["torch_dtype"] = _as_dtype(dtype)
        transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
            str(root),
            **load_kwargs,
        )

        pipe = cls(reae, transformer, upscale_mode=upscale_mode)
        if device is not None or dtype is not None:
            pipe.to(device or "cpu", dtype=dtype or "float32")
        return pipe
