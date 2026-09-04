#!/usr/bin/env python3
"""Component-swappable streaming inference for compressed SwiftVR students.

The ReAE encoder, DiT Transformer, and decoder can come from separate checkpoints.
This keeps real-video evaluation independent from training checkpoint layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from swiftvr import SwiftVRPromptFreeNoTimePipeline
from swiftvr.models import ReAE
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.models.transformer_prompt_free_no_time_moe import (
    WanTransformer3DModelPromptFreeNoTimeMoE,
)
from swiftvr.streaming import StreamingTAE


def _parse_resolution(value: str | None):
    if value is None:
        return None
    w, sep, h = value.lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    return int(w), int(h)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint root providing the frozen ReAE encoder.",
    )
    p.add_argument(
        "--transformer-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint root containing the selected transformer/ subfolder.",
    )
    p.add_argument(
        "--transformer-type",
        choices=("auto", "dense", "moe"),
        default="auto",
    )
    p.add_argument(
        "--decoder-type",
        choices=("original", "slim"),
        default="original",
    )
    p.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=None,
        help="SlimReAE tiny_decoder directory when --decoder-type=slim.",
    )
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--resolution", default=None)
    p.add_argument("--upscale", type=int, default=3)
    p.add_argument("--clip-len", type=int, default=24)
    p.add_argument("--dit-overlap", type=int, default=0)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--quality", type=int, default=85)
    p.add_argument("--png", action="store_true")
    p.add_argument("--save-format", default="")
    p.add_argument("--ffmpeg-preset", default="")
    p.add_argument("--queue-size", type=int, default=3)
    p.add_argument(
        "--attention-backend",
        default="auto",
        choices=("auto", "sdpa", "flash_attn_2", "flash_attn_3", "sageattention", "xformers"),
    )
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    p.add_argument("--quiet", action="store_true")
    return p


def _detect_transformer_type(root: Path, subfolder: str) -> str:
    config_path = root / subfolder / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Transformer config is not a JSON object: {config_path}")
    class_name = str(config.get("_class_name", ""))
    moe_keys = {"shared_expert_dim", "normal_expert_dim", "num_experts", "top_k"}
    if "MoE" in class_name or moe_keys.issubset(config):
        return "moe"
    return "dense"


def _load_transformer(
    root: Path,
    *,
    subfolder: str,
    transformer_type: str,
    dtype: torch.dtype,
):
    kind = (
        _detect_transformer_type(root, subfolder)
        if transformer_type == "auto"
        else transformer_type
    )
    cls = (
        WanTransformer3DModelPromptFreeNoTimeMoE
        if kind == "moe"
        else WanTransformer3DModelPromptFreeNoTime
    )
    transformer = cls.from_pretrained(
        str(root),
        subfolder=subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return transformer, kind


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_len <= 0 or args.clip_len % 4:
        raise ValueError("--clip-len must be a positive multiple of 4")
    if args.upscale <= 0:
        raise ValueError("--upscale must be positive")
    if args.decoder_type == "slim" and args.decoder_checkpoint is None:
        raise ValueError("--decoder-type=slim requires --decoder-checkpoint")
    if args.decoder_type == "original" and args.decoder_checkpoint is not None:
        raise ValueError("--decoder-checkpoint is only valid with --decoder-type=slim")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    base_root = args.base_checkpoint.expanduser().resolve()
    transformer_root = args.transformer_checkpoint.expanduser().resolve()

    reae_path = base_root / args.reae_filename
    if not reae_path.is_file():
        raise FileNotFoundError(reae_path)
    reae = ReAE(str(reae_path))
    transformer, transformer_kind = _load_transformer(
        transformer_root,
        subfolder=args.transformer_subfolder,
        transformer_type=args.transformer_type,
        dtype=dtype,
    )

    pipe = SwiftVRPromptFreeNoTimePipeline(reae, transformer)
    decoder_description = "original_reae"
    if args.decoder_type == "slim":
        decoder_root = args.decoder_checkpoint.expanduser().resolve()
        slim = SlimReAEDecoder.from_pretrained(
            decoder_root,
            device="cpu",
            dtype=torch.float32,
        )
        if int(slim.latent_channels) != 48:
            raise ValueError(f"Unexpected SlimReAE latent channels: {slim.latent_channels}")
        if int(slim.patch_size) != int(pipe.reae.patch_size):
            raise ValueError("SlimReAE patch size does not match base ReAE")
        if int(slim.frames_to_trim) != int(pipe.reae.frames_to_trim):
            raise ValueError("SlimReAE temporal trim does not match base ReAE")
        pipe.reae.decoder = slim.decoder
        pipe.tae_stream = StreamingTAE(pipe.reae)
        decoder_description = str(decoder_root)

    pipe.to(
        args.device,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        torch_compile=args.torch_compile,
    )
    print(
        json.dumps(
            {
                "base_reae": str(reae_path),
                "transformer_checkpoint": str(transformer_root),
                "transformer_type": transformer_kind,
                "decoder_type": args.decoder_type,
                "decoder_checkpoint": decoder_description,
                "dtype": args.dtype,
                "device": args.device,
            },
            indent=2,
        ),
        flush=True,
    )

    stats = pipe.restore_video(
        args.input,
        args.output,
        resolution=_parse_resolution(args.resolution),
        upscale=args.upscale,
        clip_len=args.clip_len,
        dit_overlap=args.dit_overlap,
        fps=args.fps,
        quality=args.quality,
        png_save=args.png,
        save_format=args.save_format,
        ffmpeg_preset=args.ffmpeg_preset,
        queue_size=args.queue_size,
        verbose=not args.quiet,
    )
    print(
        f"\nDone. {stats['frames']} frames in {stats['seconds']:.2f}s "
        f"({stats['fps']:.2f} fps) -> {stats['output']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
