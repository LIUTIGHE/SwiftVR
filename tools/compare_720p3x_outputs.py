#!/usr/bin/env python3
"""Create research-oriented visual comparisons for 720p -> 3x restoration.

The tool does not run any method.  It reads the original 720p input plus already
rendered outputs from Original SwiftVR, B1 Slim100, and optionally AVerNet.  It
writes a compact 2-column comparison video, selected-frame PNGs, and optional
native target-resolution crop strips for inspecting texture/detail.

All method outputs are required to share the same spatial resolution.  Frame
count differences are handled by using the common prefix and are recorded in
metadata.json; use outputs from the same source clip for a fair comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import decord
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(v.strip()) for v in value.split(",") if v.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("frame indices must be non-negative")
    return values


def _parse_crop(value: str) -> tuple[str, int, int, int, int]:
    label = "crop"
    raw = value
    if ":" in value:
        label, raw = value.split(":", 1)
        label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label.strip()) or "crop"
    try:
        x, y, w, h = (int(v.strip()) for v in raw.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError("crop must be [LABEL:]x,y,w,h in target coordinates") from exc
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("crop coordinates must be non-negative with positive size")
    return label, x, y, w, h


def _open_video(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    vr = decord.VideoReader(path.as_posix())
    if len(vr) <= 0:
        raise ValueError(f"empty video: {path}")
    frame = vr[0].asnumpy() if hasattr(vr[0], "asnumpy") else np.asarray(vr[0])
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = 30.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    return vr, int(frame.shape[1]), int(frame.shape[0]), fps


def _frame(vr, index: int) -> np.ndarray:
    value = vr[index]
    if hasattr(value, "asnumpy"):
        value = value.asnumpy()
    return np.asarray(value, dtype=np.uint8)


def _resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BICUBIC), dtype=np.uint8)


def _labeled_panel(frame: np.ndarray, label: str, panel_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_height = max(1, int(round(h * panel_width / w)))
    resized = Image.fromarray(frame).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    bar_h = max(28, panel_width // 30)
    canvas = Image.new("RGB", (panel_width, panel_height + bar_h), (0, 0, 0))
    canvas.paste(resized, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, max(4, bar_h // 5)), label, fill=(255, 255, 255))
    return np.asarray(canvas, dtype=np.uint8)


def _grid(panels: list[np.ndarray], columns: int = 2) -> np.ndarray:
    if not panels:
        raise ValueError("no panels")
    ph = max(p.shape[0] for p in panels)
    pw = max(p.shape[1] for p in panels)
    rows = int(math.ceil(len(panels) / columns))
    canvas = np.zeros((rows * ph, columns * pw, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, columns)
        canvas[r * ph : r * ph + panel.shape[0], c * pw : c * pw + panel.shape[1]] = panel
    return canvas


def _crop_strip(method_frames: list[tuple[str, np.ndarray]], crop, label_height: int = 28) -> np.ndarray:
    _, x, y, w, h = crop
    pieces = []
    for label, frame in method_frames:
        fh, fw = frame.shape[:2]
        if x + w > fw or y + h > fh:
            raise ValueError(f"crop {crop} exceeds {label} frame {fw}x{fh}")
        patch = Image.fromarray(frame[y : y + h, x : x + w])
        canvas = Image.new("RGB", (w, h + label_height), (0, 0, 0))
        canvas.paste(patch, (0, label_height))
        ImageDraw.Draw(canvas).text((6, 6), label, fill=(255, 255, 255))
        pieces.append(np.asarray(canvas, dtype=np.uint8))
    return np.concatenate(pieces, axis=1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lq", type=Path, required=True, help="Original 720p low-quality input video.")
    p.add_argument("--original-swiftvr", type=Path, required=True)
    p.add_argument("--b1", type=Path, required=True, help="B1 Slim100 output video.")
    p.add_argument("--avernet", type=Path, default=None, help="Optional externally generated AVerNet output.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--panel-width", type=int, default=960)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--frame-indices", type=_csv_ints, default=(0, 8, 16, 24, 32))
    p.add_argument("--crop", type=_parse_crop, action="append", default=[])
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--quality", type=int, default=8, help="imageio H.264 comparison-video quality 0-10.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.panel_width <= 0:
        raise ValueError("--panel-width must be positive")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    lq_vr, lq_w, lq_h, lq_fps = _open_video(args.lq)
    original_vr, target_w, target_h, original_fps = _open_video(args.original_swiftvr)
    b1_vr, b1_w, b1_h, b1_fps = _open_video(args.b1)
    if (b1_w, b1_h) != (target_w, target_h):
        raise ValueError(f"B1 resolution {b1_w}x{b1_h} != Original SwiftVR {target_w}x{target_h}")

    methods = [("Original SwiftVR", original_vr), ("B1 Slim100", b1_vr)]
    frame_counts = {
        "LQ": len(lq_vr),
        "Original SwiftVR": len(original_vr),
        "B1 Slim100": len(b1_vr),
    }
    if args.avernet is not None:
        aver_vr, aw, ah, aver_fps = _open_video(args.avernet)
        if (aw, ah) != (target_w, target_h):
            raise ValueError(f"AVerNet resolution {aw}x{ah} != target {target_w}x{target_h}")
        methods.append(("AVerNet", aver_vr))
        frame_counts["AVerNet"] = len(aver_vr)

    common = min(frame_counts.values())
    if args.max_frames > 0:
        common = min(common, int(args.max_frames))
    if common <= 0:
        raise RuntimeError("no common frames")
    output_fps = float(args.fps or original_fps or lq_fps)

    montage_writer = imageio.get_writer(
        str(output / "comparison.mp4"), fps=output_fps, codec="libx264",
        macro_block_size=1, quality=int(args.quality)
    )
    crop_writers = {
        crop[0]: imageio.get_writer(
            str(output / f"crop_{crop[0]}.mp4"), fps=output_fps, codec="libx264",
            macro_block_size=1, quality=int(args.quality)
        )
        for crop in args.crop
    }

    selected = set(int(v) for v in args.frame_indices if int(v) < common)
    try:
        for index in range(common):
            lq = _frame(lq_vr, index)
            lq_up = _resize_rgb(lq, target_w, target_h)
            method_frames = [("LQ Bicubic 3x", lq_up)]
            for label, vr in methods:
                method_frames.append((label, _frame(vr, index)))

            panels = [_labeled_panel(frame, label, args.panel_width) for label, frame in method_frames]
            montage = _grid(panels, columns=2)
            montage_writer.append_data(montage)
            if index in selected:
                Image.fromarray(montage).save(output / f"comparison_frame_{index:05d}.png")

            for crop in args.crop:
                strip = _crop_strip(method_frames, crop)
                crop_writers[crop[0]].append_data(strip)
                if index in selected:
                    Image.fromarray(strip).save(output / f"crop_{crop[0]}_frame_{index:05d}.png")
    finally:
        montage_writer.close()
        for writer in crop_writers.values():
            writer.close()

    metadata = {
        "lq": str(args.lq.expanduser().resolve()),
        "original_swiftvr": str(args.original_swiftvr.expanduser().resolve()),
        "b1": str(args.b1.expanduser().resolve()),
        "avernet": None if args.avernet is None else str(args.avernet.expanduser().resolve()),
        "lq_resolution": [lq_w, lq_h],
        "target_resolution": [target_w, target_h],
        "scale_ratio": [target_w / lq_w, target_h / lq_h],
        "frame_counts": frame_counts,
        "compared_frames": common,
        "fps": output_fps,
        "selected_frames": sorted(selected),
        "crops": [list(crop) for crop in args.crop],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
