"""Streaming inference for the Stage-A prompt-free/no-time DiT plus SlimReAE B1 decoder.

Example: 720p -> 3x (3840x2160)

    python scripts/inference_b1_slim.py \
      --input input_720p.mp4 \
      --output outputs/b1_slim100_3x.mp4 \
      --checkpoint /path/to/checkpoints_prompt_free_no_time \
      --slim-decoder /path/to/checkpoints/epoch_050_step_00012400/tiny_decoder \
      --upscale 3 --clip-len 24 --dtype bfloat16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from swiftvr import SwiftVRPromptFreeNoTimePipeline
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder
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
    p.add_argument("--checkpoint", required=True,
                   help="Stage-A prompt-free/no-time checkpoint root.")
    p.add_argument("--slim-decoder", required=True,
                   help="SlimReAE checkpoint/tiny_decoder directory.")
    p.add_argument("--resolution", default=None,
                   help="Optional output WxH; overrides --upscale.")
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
        choices=["auto", "sdpa", "flash_attn_2", "flash_attn_3", "sageattention", "xformers"],
    )
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_len <= 0 or args.clip_len % 4:
        raise ValueError(f"--clip-len must be a positive multiple of 4, got {args.clip_len}")
    if args.upscale <= 0:
        raise ValueError("--upscale must be positive")

    pipe = SwiftVRPromptFreeNoTimePipeline.from_pretrained(args.checkpoint)
    slim = SlimReAEDecoder.from_pretrained(args.slim_decoder, device="cpu", dtype=torch.float32)
    if int(slim.latent_channels) != 48:
        raise ValueError(f"unexpected SlimReAE latent channels: {slim.latent_channels}")
    if int(slim.patch_size) != int(pipe.reae.patch_size):
        raise ValueError("SlimReAE patch size does not match base ReAE")
    if int(slim.frames_to_trim) != int(pipe.reae.frames_to_trim):
        raise ValueError("SlimReAE temporal trim does not match base ReAE")

    # Keep the original ReAE encoder and replace only its decoder stack.
    pipe.reae.decoder = slim.decoder
    pipe.tae_stream = StreamingTAE(pipe.reae)
    pipe.to(
        args.device,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        torch_compile=args.torch_compile,
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
