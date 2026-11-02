"""Command-line entry point for SwiftVR inference.

Thin wrapper around ``swiftvr.SwiftVRPipeline``; all defaults live in
``SwiftVRPipeline.restore_video``.

    python scripts/inference.py \
        --input low_quality.mp4 --output restored.mp4 \
        --checkpoint checkpoints/ --upscale 4 --clip-len 24 --dtype bfloat16
"""

import argparse
import time
from pathlib import Path

import imageio
import torch
from PIL import Image

from swiftvr import SwiftVRPipeline
from swiftvr.io import (
    get_video_info,
    iter_video_clips_fixed_scheme,
    ntchw_to_uint8_frames,
    open_stream_video_writer,
)


def _parse_resolution(value):
    if value is None:
        return None
    w, h = value.lower().split("x")
    return int(w), int(h)


def build_parser():
    p = argparse.ArgumentParser(description="SwiftVR streaming video restoration.")

    p.add_argument("--input", required=True, help="Low-quality video file or image folder.")
    p.add_argument("--output", required=True, help="Output mp4 path or directory.")
    p.add_argument("--checkpoint", required=True, help="Checkpoint directory (see README layout).")

    p.add_argument("--resolution", type=str, default=None,
                   help="Output resolution as WxH (e.g. 1920x1080). Overrides --upscale.")
    p.add_argument("--upscale", type=int, default=4, help="Upscale factor when --resolution is unset.")
    p.add_argument("--clip-len", type=int, default=24, help="MIDDLE chunk size (multiple of 4).")
    p.add_argument("--dit-overlap", type=int, default=0, help="Temporal overlap (latents) for blending.")

    p.add_argument("--fps", type=float, default=None, help="Output fps (defaults to source fps).")
    p.add_argument("--quality", type=int, default=60, help="Output quality 0-100 (maps to x265 CRF).")
    p.add_argument("--png", action="store_true", help="Write a PNG sequence instead of an mp4.")
    p.add_argument("--save-format", type=str, default="", help="Set to 'yuv444p' for 4:4:4 mp4.")
    p.add_argument("--ffmpeg-preset", type=str, default="", help="x265 preset (e.g. fast, medium).")
    p.add_argument("--queue-size", type=int, default=3, help="Pipeline queue depth.")
    p.add_argument("--attention_backend", type=str, default="auto",
                   choices=["auto", "sdpa", "flash_attn_2", "flash_attn_3", "sageattention", "xformers"],
                   help="Attention backend. 'auto' lets SwiftVR pick the fastest available backend.")
    p.add_argument("--torch_compile", action="store_true",
                   help="Enable torch.compile. Disabled by default to avoid long recompilation on dynamic paths.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--chunk-trace",
        default=None,
        help=(
            "Optional fresh diagnostic directory for the fixed-scheme restore_video path; "
            "requires --png, --dit-overlap 0, and cannot be combined with --stream."
        ),
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--stream", action="store_true")
    return p


def _write_ntchw_chunk(
    video_ntchw,
    *,
    writer,
    output_dir,
    png_save,
    start_idx,
    max_frames=None,
):
    """
    video_ntchw: [1, T, 3, H, W] in [0, 1]
    Returns number of frames written.
    """
    if video_ntchw is None:
        return 0
    if video_ntchw.numel() == 0 or video_ntchw.shape[1] == 0:
        return 0

    frames = ntchw_to_uint8_frames(video_ntchw)
    if frames is None or len(frames) == 0:
        return 0

    if max_frames is not None:
        remain = max_frames - start_idx
        if remain <= 0:
            return 0
        frames = frames[:remain]

    if png_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(output_dir / f"{start_idx + i:05d}.png")
    else:
        for frame in frames:
            writer.append_data(frame)

    return int(len(frames))


def run_stream(args, pipe):
    input_path = Path(args.input)
    output_path = Path(args.output)

    raw_total, lq_h, lq_w, src_fps = get_video_info(
        input_path, fallback_fps=args.fps or 30
    )

    # Match official restore_video() frame-count convention.
    # Official code also truncates raw_total to 4k+1.
    total_frames = 4 * ((raw_total - 1) // 4) + 1

    resolution = _parse_resolution(args.resolution)

    session = pipe.stream(
        clip_len=args.clip_len,
        resolution=resolution,
        upscale=args.upscale,
        dit_overlap=args.dit_overlap,
    )

    if args.png:
        output_dir = output_path
        writer = None
        final_output = str(output_dir)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = output_path.parent
        writer = open_stream_video_writer(
            str(output_path),
            fps=args.fps or src_fps,
            video_format=args.save_format,
            preset=args.ffmpeg_preset,
            quality=args.quality,
        )
        final_output = str(output_path)

    written = 0
    t0 = time.perf_counter()

    try:
        for spec, lq_chunk in iter_video_clips_fixed_scheme(
            input_path,
            clip_len=args.clip_len,
            total_frames=total_frames,
            crop_h=lq_h,
            crop_w=lq_w,
        ):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            hq = session.step(lq_chunk)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            n = _write_ntchw_chunk(
                hq,
                writer=writer,
                output_dir=output_dir,
                png_save=args.png,
                start_idx=written,
                max_frames=total_frames,
            )

            t2 = time.perf_counter()
            print(
                f"step_time={t1 - t0:.3f}s, "
                f"write_time={t2 - t1:.3f}s, "
                f"out={0 if hq is None else hq.shape[1]}f"
            )
            written += n

            if not args.quiet:
                out_n = 0 if hq is None else int(hq.shape[1])
                print(
                    f"  [stream] {spec.ctype.value:6s} "
                    f"clip {spec.clip_idx}: in={lq_chunk.shape[0]}f "
                    f"out={out_n}f written={written}"
                )

        tail = session.flush()

        n = _write_ntchw_chunk(
            tail,
            writer=writer,
            output_dir=output_dir,
            png_save=args.png,
            start_idx=written,
            max_frames=total_frames,
        )
        written += n

        if not args.quiet:
            out_n = 0 if tail is None else int(tail.shape[1])
            print(f"  [stream] flush: out={out_n}f written={written}")

    finally:
        if writer is not None:
            writer.close()

    wall = time.perf_counter() - t0
    return {
        "frames": written,
        "seconds": wall,
        "fps": written / wall if wall > 0 else 0.0,
        "output": final_output,
    }


def main():
    args = build_parser().parse_args()

    if args.stream and args.chunk_trace is not None:
        raise ValueError(
            "--chunk-trace currently observes the canonical fixed-scheme "
            "restore_video path only; do not combine it with --stream."
        )

    pipe = SwiftVRPipeline.from_pretrained(args.checkpoint).to(
        args.device,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        torch_compile=args.torch_compile,
    )

    if args.stream:
        stats = run_stream(args, pipe)
    else:
        trace = None
        if args.chunk_trace is not None:
            from swiftvr.streaming.chunk_trace import ChunkTrace

            trace = ChunkTrace(pipe, args.chunk_trace, args)

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

        if trace is not None:
            trace.finish(stats)

    print(
        f"\nDone. {stats['frames']} frames in {stats['seconds']:.2f}s "
        f"({stats['fps']:.2f} fps) -> {stats['output']}"
    )


if __name__ == "__main__":
    main()
