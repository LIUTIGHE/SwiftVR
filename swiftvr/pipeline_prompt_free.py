"""High-level pipeline for the prompt-free SwiftVR student.

The pipeline preserves SwiftVR's ReAE encoder/decoder, frame preprocessing,
causal chunk protocol, and threaded runner. Only the DiT implementation is
replaced. The converted checkpoint therefore does not need
``prompt_embedding.safetensors``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .models import ReAE
from .models.transformer_prompt_free import WanTransformer3DModelPromptFree
from .pipeline import SwiftVRPipeline, _as_dtype
from .streaming.dit_prompt_free import StreamingDiTPromptFree


class _RunnerCompatiblePromptFreeDiT(StreamingDiTPromptFree):
    """Adapt the prompt-free DiT to the unchanged runner call signature.

    The original runner passes an empty-prompt embedding to ``denoise`` and
    ``denoise_last_chunk``. Keeping that runner untouched avoids changing the
    behavior of the original SwiftVR pipeline. This adapter discards the unused
    argument and delegates to :class:`StreamingDiTPromptFree`.
    """

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


class SwiftVRPromptFreePipeline(SwiftVRPipeline):
    """SwiftVR pipeline using the prompt-free DiT student.

    ``restore_video`` and ``stream`` are inherited from
    :class:`SwiftVRPipeline`. The compatibility wrapper above allows both APIs
    to reuse the original runner and stream session without a second copy of
    the I/O and frame-alignment logic.
    """

    def __init__(self, reae, transformer, upscale_mode: str = "bilinear"):
        super().__init__(
            reae=reae,
            transformer=transformer,
            prompt_emb=None,
            upscale_mode=upscale_mode,
        )
        self.dit_stream = _RunnerCompatiblePromptFreeDiT(transformer, overlap=0)

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
    ) -> "SwiftVRPromptFreePipeline":
        """Load a converted prompt-free checkpoint directory.

        Expected layout::

            checkpoint_dir/
            ├── reae.safetensors
            └── transformer/
                ├── config.json
                └── diffusion_pytorch_model.safetensors

        No prompt embedding file is read. When ``dtype`` is supplied, it is
        forwarded to the Diffusers loader so the multi-billion-parameter DiT
        is not first materialized in float32 and cast only afterward.
        """

        root = Path(checkpoint_dir)
        reae = ReAE(str(root / reae_filename))

        transformer_load_kwargs = {"subfolder": transformer_subfolder}
        if dtype is not None:
            transformer_load_kwargs["torch_dtype"] = _as_dtype(dtype)
        transformer = WanTransformer3DModelPromptFree.from_pretrained(
            str(root),
            **transformer_load_kwargs,
        )

        pipe = cls(reae, transformer, upscale_mode=upscale_mode)
        if device is not None or dtype is not None:
            pipe.to(device or "cpu", dtype=dtype or "float32")
        return pipe
