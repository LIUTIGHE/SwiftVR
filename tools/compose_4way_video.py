#!/usr/bin/env python3
"""Compose exact-size four-way SwiftVR comparison videos by ordinal frame index.

Order is fixed to: LQ | BasicCNN | Original SwiftVR | Ours.
Outputs:
  * quadrant.mp4: 2x2 layout within the target W x H canvas.
  * vertical_quarters.mp4: full-height quarter-width slices, preserving W x H.

BasicCNN alignment defaults to the Ours output timeline:
  * same frame count -> identity mapping;
  * more BasicCNN frames -> normalized-time downsampling with aligned endpoints;
  * fewer BasicCNN frames -> error by default (do not silently duplicate frames).
Manual N*scale+offset mapping remains available for exact known baselines.
Source FPS/timestamps are reported but never used to choose corresponding frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from tools.compare_720p3x_outputs import FrameSource, _logical_frame_count, _resize_rgb


LABELS = ("LQ", "Basic CNN", "Original SwiftVR", "Ours")


def _auto_basic_index(index: int, *, basic_count: int, reference_count: int) -> int:
    """Map an Ours timeline index to BasicCNN by normalized ordinal time."""
    if reference_count <= 0 or basic_count <= 0:
        raise ValueError("frame counts must be positive")
    if not 0 <= index < reference_count:
        raise IndexError(index)
    if basic_count < reference_count:
        raise ValueError(
            f"BasicCNN has fewer frames than Ours: {basic_count} < {reference_count}; "
            "refusing to duplicate/interpolate missing BasicCNN frames"
        )
    if basic_count == reference_count or reference_count == 1:
        return index if basic_count == reference_count else 0
    return int(round(index * (basic_count - 1) / (reference_count - 1)))


def _draw_label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    # In-frame label: do not change output geometry.
    bbox = draw.textbbox((0, 0), text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y, pad = 12, 10, 6
    draw.rectangle((x - pad, y - pad, x + tw + pad, y + th + pad), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _quadrant(frames: list[np.ndarray], width: int, height: int, labels: bool) -> np.ndarray:
    if width % 2 or height % 2:
        raise ValueError(f"target resolution must be divisible by 2, got {width}x{height}")
    pw, ph = width // 2, height // 2
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        panel = _resize(frame, pw, ph)
        if labels:
            panel = _draw_label(panel, LABELS[index])
        row, col = divmod(index, 2)
        canvas[row * ph:(row + 1) * ph, col * pw:(col + 1) * pw] = panel
    return canvas


def _quarters(frames: list[np.ndarray], width: int, height: int, labels: bool) -> np.ndarray:
    if width % 4:
        raise ValueError(f"target width must be divisible by 4, got {width}")
    q = width // 4
    pieces = []
    for index, frame in enumerate(frames):
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = _resize(frame, width, height)
        piece = frame[:, index * q:(index + 1) * q].copy()
        if labels:
            piece = _draw_label(piece, LABELS[index])
        pieces.append(piece)
    return np.concatenate(pieces, axis=1)


def _writer(path: Path, fps: float, pix_fmt: str):
    return imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        pixelformat=pix_fmt,
        ffmpeg_params=["-crf", "12", "-preset", "medium"],
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lq", type=Path, required=True)
    p.add_argument("--basiccnn", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--basiccnn-align",
        choices=("auto", "manual"),
        default="auto",
        help="auto: align BasicCNN to Ours by normalized frame timeline; manual: use N*scale+offset",
    )
    p.add_argument("--basiccnn-index-scale", type=int, default=2)
    p.add_argument("--basiccnn-index-offset", type=int, default=0)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--pix-fmt", default="yuv444p", choices=("yuv420p", "yuv444p"))
    p.add_argument("--no-labels", action="store_true")
    args = p.parse_args()

    if args.basiccnn_index_scale <= 0 or args.basiccnn_index_offset < 0:
        raise ValueError("invalid BasicCNN ordinal mapping")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")

    lq = FrameSource(args.lq)
    basic = FrameSource(args.basiccnn, fallback_fps=lq.fps)
    original = FrameSource(args.original, fallback_fps=lq.fps)
    ours = FrameSource(args.ours, fallback_fps=lq.fps)

    width, height = original.width, original.height
    for label, source in (("BasicCNN", basic), ("Ours", ours)):
        if (source.width, source.height) != (width, height):
            raise ValueError(
                f"{label} resolution {source.width}x{source.height} != "
                f"Original {width}x{height}"
            )

    reference_count = len(ours)
    if reference_count <= 0:
        raise RuntimeError("Ours output is empty")
    if len(lq) != reference_count:
        raise ValueError(
            f"LQ/Ours frame-count mismatch: LQ={len(lq)} Ours={reference_count}; "
            "refusing to resample the reference input timeline"
        )
    if len(original) != reference_count:
        raise ValueError(
            f"Original/Ours frame-count mismatch: Original={len(original)} "
            f"Ours={reference_count}; refusing to hide SwiftVR frame-count drift"
        )

    if args.basiccnn_align == "auto":
        if len(basic) < reference_count:
            raise ValueError(
                f"BasicCNN has fewer frames than Ours: {len(basic)} < {reference_count}. "
                "Auto mode only downsamples extra BasicCNN frames."
            )
        common = reference_count
        basic_mapping = (
            "identity"
            if len(basic) == reference_count
            else "normalized_timeline_downsample"
        )
    else:
        basic_logical = _logical_frame_count(
            basic, args.basiccnn_index_scale, args.basiccnn_index_offset
        )
        if basic_logical < reference_count:
            raise ValueError(
                f"manual BasicCNN mapping exposes only {basic_logical} logical frames "
                f"for Ours reference length {reference_count}"
            )
        common = reference_count
        basic_mapping = (
            f"N*{args.basiccnn_index_scale}+{args.basiccnn_index_offset}"
        )

    if args.max_frames:
        common = min(common, args.max_frames)
    if common <= 0:
        raise RuntimeError("no aligned frames")

    print(
        "Alignment: "
        f"Ours/LQ/Original={reference_count} frames, "
        f"BasicCNN={len(basic)} frames, mode={args.basiccnn_align}, "
        f"mapping={basic_mapping}, output_frames={common}",
        flush=True,
    )

    fps = float(args.fps if args.fps is not None else lq.fps)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {output}")
    output.mkdir(parents=True, exist_ok=True)

    quadrant_writer = _writer(output / "quadrant.mp4", fps, args.pix_fmt)
    quarters_writer = _writer(output / "vertical_quarters.mp4", fps, args.pix_fmt)
    labels = not args.no_labels
    try:
        for n in range(common):
            lq_up = _resize_rgb(lq.frame(n), width, height)
            if args.basiccnn_align == "auto":
                basic_index = _auto_basic_index(
                    n,
                    basic_count=len(basic),
                    reference_count=reference_count,
                )
            else:
                basic_index = (
                    n * args.basiccnn_index_scale + args.basiccnn_index_offset
                )
            basic_frame = basic.frame(basic_index)
            frames = [lq_up, basic_frame, original.frame(n), ours.frame(n)]
            quadrant_writer.append_data(_quadrant(frames, width, height, labels))
            quarters_writer.append_data(_quarters(frames, width, height, labels))
            if (n + 1) % 100 == 0 or n + 1 == common:
                print(f"compose {n + 1}/{common}", flush=True)
    finally:
        quadrant_writer.close()
        quarters_writer.close()

    print(
        f"Done: {common} aligned frames, {width}x{height}, fps={fps:g}, "
        f"BasicCNN mapping={basic_mapping}\n"
        f"  {output / 'quadrant.mp4'}\n"
        f"  {output / 'vertical_quarters.mp4'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
