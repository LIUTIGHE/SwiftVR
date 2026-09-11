#!/usr/bin/env python3
"""Profile engineering knobs inside the real SwiftVR streaming pipeline.

Unlike ``profile_short_clip_sweep.py`` this tool does NOT replace streaming with
whole-clip inference.  It runs the existing fixed FIRST/MIDDLE/LAST protocol on
one real input video and sweeps:

* RGB streaming ``clip_len`` (must be a multiple of four);
* optional DiT temporal subchunking in latent space while keeping the ReAE RGB
  chunk unchanged;
* DiT overlap.

The RuntimeMacCounter is enabled around the actual ``restore_video`` call, so
reported MACs include every FIRST/MIDDLE/LAST encoder, DiT and decoder execution
performed by the standard runner.  Wall time therefore also includes the normal
reader/H2D/GPU/writer pipeline.

Canonical FLOP conversion uses 1 MAC = 2 FLOPs.  SDPA is required so QK/AV MACs
remain observable.  ``dit_subchunk_latents=0`` means the unmodified DiT path.
The LAST-chunk path is deliberately kept identical to the production pipeline;
subchunking only changes regular FIRST/MIDDLE ``denoise`` calls.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr import SwiftVRPromptFreeNoTimePipeline
from swiftvr.io import get_video_info
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder
from swiftvr.streaming import StreamingTAE
from swiftvr.streaming.chunk import build_chunk_specs
from swiftvr.streaming.dit_prompt_free_no_time import StreamingDiTPromptFreeNoTime
from tools.runtime_macs import RuntimeMacCounter

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _csv_ints(value: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    try:
        result = tuple(dict.fromkeys(int(v.strip()) for v in value.split(",") if v.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    minimum = 0 if allow_zero else 1
    if any(v < minimum for v in result):
        raise argparse.ArgumentTypeError(f"values must be >= {minimum}")
    return result


def _parse_resolution(value: str) -> tuple[int, int]:
    w, sep, h = value.lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    try:
        w_i, h_i = int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must contain integers") from exc
    if w_i <= 0 or h_i <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return w_i, h_i


class SubchunkedPromptFreeNoTimeDiT(StreamingDiTPromptFreeNoTime):
    """Runner-compatible DiT that optionally splits regular latent chunks."""

    def __init__(self, transformer, *, overlap: int = 0, subchunk_latents: int = 0):
        super().__init__(transformer, overlap=overlap)
        self.subchunk_latents = int(subchunk_latents)

    @torch.inference_mode()
    def denoise(self, lq, prompt_emb=None):
        del prompt_emb
        size = self.subchunk_latents
        if size <= 0 or int(lq.shape[2]) <= size:
            return super().denoise(lq)
        outputs = []
        for start in range(0, int(lq.shape[2]), size):
            outputs.append(super().denoise(lq[:, :, start : start + size]))
        return torch.cat(outputs, dim=2)

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


def _install_slim_decoder(pipe, slim_dir: Path, *, dtype: torch.dtype) -> dict[str, object]:
    slim = SlimReAEDecoder.from_pretrained(slim_dir, device="cpu", dtype=dtype)
    if int(slim.latent_channels) != 48:
        raise ValueError(f"unexpected SlimReAE latent channels: {slim.latent_channels}")
    if int(slim.patch_size) != int(pipe.reae.patch_size):
        raise ValueError("SlimReAE patch_size does not match base ReAE")
    if int(slim.frames_to_trim) != int(pipe.reae.frames_to_trim):
        raise ValueError("SlimReAE frames_to_trim does not match base ReAE")
    pipe.reae.decoder = slim.decoder
    pipe.tae_stream = StreamingTAE(pipe.reae)
    return {
        "channels": list(slim.channels),
        "pruning_metadata": dict(slim.pruning_metadata),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Stage-A prompt-free/no-time checkpoint root.")
    p.add_argument("--slim-decoder", type=Path, default=None,
                   help="Optional SlimReAE checkpoint/tiny_decoder directory.")
    p.add_argument("--resolution", type=_parse_resolution, default=(1920, 1080))
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--clip-lens", type=lambda s: _csv_ints(s), default=(8, 12, 16, 24, 32))
    p.add_argument("--dit-subchunks", type=lambda s: _csv_ints(s, allow_zero=True), default=(0,))
    p.add_argument("--dit-overlaps", type=lambda s: _csv_ints(s, allow_zero=True), default=(0,))
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--queue-size", type=int, default=3)
    p.add_argument("--quality", type=int, default=20)
    p.add_argument("--ffmpeg-preset", default="ultrafast")
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/streaming_engineering_sweep"))
    p.add_argument("--output-json", type=Path, default=Path("outputs/streaming_engineering_sweep.json"))
    return p


def _validate_args(args: argparse.Namespace) -> None:
    for clip_len in args.clip_lens:
        if clip_len <= 0 or clip_len % 4:
            raise ValueError(f"clip_len must be a positive multiple of 4, got {clip_len}")
    if args.upscale <= 0:
        raise ValueError("--upscale must be positive")
    if args.queue_size <= 0:
        raise ValueError("--queue-size must be positive")


def _chunk_summary(total_frames: int, clip_len: int) -> list[dict[str, object]]:
    return [
        {
            "type": spec.ctype.value,
            "clip_idx": int(spec.clip_idx),
            "frame_start": int(spec.frame_start),
            "frame_count": int(spec.frame_count),
            "b": int(spec.b),
        }
        for spec in build_chunk_specs(total_frames, clip_len)
    ]


def _record_from_summary(*, args, stats, summary, clip_len, subchunk, overlap, peak_gb, chunks):
    by_root = summary.get("by_root_gmacs_per_output_frame", {})
    by_type = summary.get("by_type_gmacs", {})
    frames = int(stats["frames"])
    attn_clip = float(by_type.get("self_attn_qk", 0.0)) + float(by_type.get("self_attn_av", 0.0))
    attn_frame = attn_clip / max(frames, 1)
    total_frame = float(summary["gmacs_per_output_frame"])
    return {
        "clip_len": int(clip_len),
        "dit_subchunk_latents": int(subchunk),
        "dit_overlap_latents": int(overlap),
        "frames": frames,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "wall_seconds": float(stats["seconds"]),
        "end_to_end_fps": float(stats["fps"]),
        "peak_allocated_gb": float(peak_gb),
        "total_gmac_per_output_frame": total_frame,
        "total_gflops_per_output_frame_if_1mac_2flops": 2.0 * total_frame,
        "encoder_gmac_per_output_frame": float(by_root.get("encoder", 0.0)),
        "dit_gmac_per_output_frame": float(by_root.get("transformer", 0.0)),
        "decoder_gmac_per_output_frame": float(by_root.get("decoder", 0.0)),
        "self_attention_gmac_per_output_frame": attn_frame,
        "self_attention_fraction_of_dit": (
            attn_frame / float(by_root.get("transformer", 1.0))
            if float(by_root.get("transformer", 0.0)) > 0 else 0.0
        ),
        "by_type_gmac": dict(by_type),
        "count_errors": list(summary.get("count_errors", [])),
    }


def _print_table(records: list[dict[str, object]], reference: dict[str, object]) -> None:
    ref = float(reference["total_gmac_per_output_frame"])
    print("\n================ real streaming engineering sweep ================")
    print("clip sub ol | chunks  encG/f   ditG/f   decG/f  attnG/f attn% | totalG/f GFLOPs/f | FPS PeakGB | vsRef")
    print("-" * 124)
    for r in records:
        print(
            f"{int(r['clip_len']):4d} {int(r['dit_subchunk_latents']):3d} {int(r['dit_overlap_latents']):2d} | "
            f"{int(r['chunk_count']):6d} "
            f"{float(r['encoder_gmac_per_output_frame']):7.1f} "
            f"{float(r['dit_gmac_per_output_frame']):8.1f} "
            f"{float(r['decoder_gmac_per_output_frame']):7.1f} "
            f"{float(r['self_attention_gmac_per_output_frame']):8.1f} "
            f"{100*float(r['self_attention_fraction_of_dit']):5.1f}% | "
            f"{float(r['total_gmac_per_output_frame']):8.1f} "
            f"{float(r['total_gflops_per_output_frame_if_1mac_2flops']):8.1f} | "
            f"{float(r['end_to_end_fps']):5.2f} {float(r['peak_allocated_gb']):6.2f} | "
            f"{float(r['total_gmac_per_output_frame'])/ref:5.3f}"
        )
    print("Reference = clip_len=24, subchunk=0, overlap=0 when present; otherwise first row.")
    print("sub=0 means original DiT call; sub>0 splits regular FIRST/MIDDLE latent chunks only.")
    print("===================================================================\n")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    raw_total, lq_h, lq_w, source_fps = get_video_info(args.input)
    total_frames = 4 * ((int(raw_total) - 1) // 4) + 1
    if total_frames <= 0:
        raise ValueError("input has no usable 4k+1 frames")

    pipe = SwiftVRPromptFreeNoTimePipeline.from_pretrained(args.checkpoint)
    slim_meta = None
    if args.slim_decoder is not None:
        slim_meta = _install_slim_decoder(pipe, args.slim_decoder, dtype=torch.float32)
    pipe.to(device, dtype=dtype, attention_backend="sdpa", torch_compile=False)

    counter = RuntimeMacCounter()
    counter.add_module("encoder", pipe.reae.encoder)
    counter.add_module("transformer", pipe.transformer)
    counter.add_module("decoder", pipe.reae.decoder)

    out_dir = args.output_dir.expanduser().resolve()
    if args.keep_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    try:
        for clip_len in args.clip_lens:
            chunks = _chunk_summary(total_frames, clip_len)
            for subchunk in args.dit_subchunks:
                for overlap in args.dit_overlaps:
                    pipe.tae_stream = StreamingTAE(pipe.reae)
                    pipe.dit_stream = SubchunkedPromptFreeNoTimeDiT(
                        pipe.transformer,
                        overlap=int(overlap),
                        subchunk_latents=int(subchunk),
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)
                        torch.cuda.synchronize(device)

                    if args.keep_outputs:
                        output_path = out_dir / f"clip{clip_len}_sub{subchunk}_ol{overlap}.mp4"
                        temp_ctx = None
                    else:
                        temp_ctx = tempfile.TemporaryDirectory(prefix="swiftvr_stream_sweep_")
                        output_path = Path(temp_ctx.name) / "out.mp4"

                    with counter.count(reset=True):
                        stats = pipe.restore_video(
                            args.input,
                            output_path,
                            resolution=args.resolution,
                            upscale=args.upscale,
                            clip_len=int(clip_len),
                            dit_overlap=int(overlap),
                            fps=source_fps,
                            quality=args.quality,
                            png_save=False,
                            ffmpeg_preset=args.ffmpeg_preset,
                            queue_size=args.queue_size,
                            verbose=False,
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                    else:
                        peak_gb = 0.0
                    summary = counter.summary(emitted_frames=int(stats["frames"]))
                    record = _record_from_summary(
                        args=args,
                        stats=stats,
                        summary=summary,
                        clip_len=clip_len,
                        subchunk=subchunk,
                        overlap=overlap,
                        peak_gb=peak_gb,
                        chunks=chunks,
                    )
                    records.append(record)
                    print(
                        f"[clip={clip_len:2d} sub={subchunk:2d} ol={overlap}] "
                        f"total={record['total_gmac_per_output_frame']:.3f} GMAC/f "
                        f"DiT={record['dit_gmac_per_output_frame']:.3f} "
                        f"FPS={record['end_to_end_fps']:.2f}",
                        flush=True,
                    )
                    if temp_ctx is not None:
                        temp_ctx.cleanup()
    finally:
        counter.close()

    reference = next(
        (
            r for r in records
            if int(r["clip_len"]) == 24
            and int(r["dit_subchunk_latents"]) == 0
            and int(r["dit_overlap_latents"]) == 0
        ),
        records[0],
    )
    for r in records:
        r["total_gmac_ratio_vs_reference"] = (
            float(r["total_gmac_per_output_frame"])
            / float(reference["total_gmac_per_output_frame"])
        )

    result = {
        "input": str(args.input.expanduser().resolve()),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "slim_decoder": None if args.slim_decoder is None else str(args.slim_decoder.expanduser().resolve()),
        "slim_decoder_metadata": slim_meta,
        "raw_total_frames": int(raw_total),
        "profiled_total_frames": int(total_frames),
        "lq_geometry": [int(lq_w), int(lq_h)],
        "output_resolution": list(args.resolution),
        "queue_size": int(args.queue_size),
        "reference": {
            "clip_len": int(reference["clip_len"]),
            "dit_subchunk_latents": int(reference["dit_subchunk_latents"]),
            "dit_overlap_latents": int(reference["dit_overlap_latents"]),
        },
        "records": records,
    }
    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _print_table(records, reference)
    print(f"Saved: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
