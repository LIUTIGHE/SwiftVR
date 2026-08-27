#!/usr/bin/env python3
"""Create research-oriented visual comparisons for 720p -> 3x restoration.

Inputs may be video files or image-sequence directories. For publication-quality
inspection, prefer PNG directories from SwiftVR ``--png`` inference so codec
artifacts cannot hide or invent high-frequency detail. The tool writes a compact
2-column comparison video, selected-frame PNGs, and optional native target-space
crop strips.

The historical ``--original-swiftvr`` / ``--b1`` / ``--avernet`` interface is
kept for backward compatibility. Labels are configurable so later compression
stages (for example B2-A + B1 Slim100) can reuse the same comparison tool without
mislabeling the candidate. An optional GT source can be included when a paired
real/test sample is available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import decord
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(v.strip()) for v in value.split(",") if v.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("frame indices must be non-negative")
    return values


def _parse_crop(value: str) -> tuple[str, int, int, int, int]:
    label, raw = "crop", value
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


def _sort_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


class FrameSource:
    def __init__(self, path: Path, fallback_fps: float = 30.0):
        self.path = path.expanduser().resolve()
        self.kind = "images" if self.path.is_dir() else "video"
        self._fps = float(fallback_fps)
        if self.kind == "images":
            self.files = sorted(
                (p for p in self.path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                key=_sort_key,
            )
            if not self.files:
                raise ValueError(f"no images in {self.path}")
            with Image.open(self.files[0]) as image:
                self.width, self.height = image.convert("RGB").size
            self.reader = None
        else:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            self.reader = decord.VideoReader(self.path.as_posix())
            if len(self.reader) <= 0:
                raise ValueError(f"empty video: {self.path}")
            first = self._video_frame(0)
            self.height, self.width = first.shape[:2]
            try:
                fps = float(self.reader.get_avg_fps())
                if math.isfinite(fps) and fps > 0:
                    self._fps = fps
            except Exception:
                pass
            self.files = []

    def __len__(self):
        return len(self.files) if self.kind == "images" else len(self.reader)

    @property
    def fps(self) -> float:
        return self._fps

    def _video_frame(self, index: int) -> np.ndarray:
        value = self.reader[index]
        if hasattr(value, "asnumpy"):
            value = value.asnumpy()
        return np.asarray(value, dtype=np.uint8)

    def frame(self, index: int) -> np.ndarray:
        if self.kind == "video":
            return self._video_frame(index)
        with Image.open(self.files[index]) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BICUBIC), dtype=np.uint8)


def _labeled_panel(frame: np.ndarray, label: str, panel_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_height = max(1, int(round(h * panel_width / w)))
    resized = Image.fromarray(frame).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    bar_h = max(28, panel_width // 30)
    canvas = Image.new("RGB", (panel_width, panel_height + bar_h), (0, 0, 0))
    canvas.paste(resized, (0, bar_h))
    ImageDraw.Draw(canvas).text((10, max(4, bar_h // 5)), label, fill=(255, 255, 255))
    return np.asarray(canvas, dtype=np.uint8)


def _grid(panels: list[np.ndarray], columns: int = 2) -> np.ndarray:
    if not panels:
        raise ValueError("no panels")
    ph, pw = max(p.shape[0] for p in panels), max(p.shape[1] for p in panels)
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
    p.add_argument("--lq", type=Path, required=True, help="Original 720p LQ video or image directory.")
    p.add_argument("--gt", type=Path, default=None,
                   help="Optional paired GT video/image directory at target resolution.")
    p.add_argument("--gt-label", default="GT")
    p.add_argument("--original-swiftvr", type=Path, required=True)
    p.add_argument("--original-label", default="Original SwiftVR")
    p.add_argument("--b1", type=Path, required=True,
                   help="Required candidate video/image directory; historically B1 Slim100.")
    p.add_argument("--b1-label", default="B1 Slim100")
    p.add_argument("--avernet", type=Path, default=None,
                   help="Optional additional method video/image directory.")
    p.add_argument("--avernet-label", default="AVerNet")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--panel-width", type=int, default=960)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--frame-indices", type=_csv_ints, default=(0, 8, 16, 24, 32))
    p.add_argument("--crop", type=_parse_crop, action="append", default=[])
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--quality", type=int, default=8)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.panel_width <= 0:
        raise ValueError("--panel-width must be positive")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    lq = FrameSource(args.lq)
    original = FrameSource(args.original_swiftvr, fallback_fps=lq.fps)
    b1 = FrameSource(args.b1, fallback_fps=original.fps)
    target_w, target_h = original.width, original.height
    if (b1.width, b1.height) != (target_w, target_h):
        raise ValueError(f"candidate resolution {b1.width}x{b1.height} != Original {target_w}x{target_h}")

    methods: list[tuple[str, FrameSource]] = []
    if args.gt is not None:
        gt = FrameSource(args.gt, fallback_fps=original.fps)
        if (gt.width, gt.height) != (target_w, target_h):
            raise ValueError(f"GT resolution {gt.width}x{gt.height} != target {target_w}x{target_h}")
        methods.append((args.gt_label, gt))
    methods.extend(((args.original_label, original), (args.b1_label, b1)))
    if args.avernet is not None:
        avernet = FrameSource(args.avernet, fallback_fps=original.fps)
        if (avernet.width, avernet.height) != (target_w, target_h):
            raise ValueError(f"additional method resolution {avernet.width}x{avernet.height} != target {target_w}x{target_h}")
        methods.append((args.avernet_label, avernet))

    sources = [("LQ", lq)] + methods
    frame_counts = {label: len(source) for label, source in sources}
    common = min(frame_counts.values())
    if args.max_frames > 0:
        common = min(common, int(args.max_frames))
    if common <= 0:
        raise RuntimeError("no common frames")
    output_fps = float(args.fps or original.fps or lq.fps)

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
    selected = set(v for v in args.frame_indices if v < common)

    try:
        for index in range(common):
            lq_up = _resize_rgb(lq.frame(index), target_w, target_h)
            method_frames = [("LQ Bicubic 3x", lq_up)] + [
                (label, source.frame(index)) for label, source in methods
            ]
            montage = _grid(
                [_labeled_panel(frame, label, args.panel_width) for label, frame in method_frames],
                columns=2,
            )
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
        "lq": str(lq.path),
        "gt": None if args.gt is None else str(args.gt.expanduser().resolve()),
        "original_swiftvr": str(original.path),
        "b1": str(b1.path),
        "avernet": None if args.avernet is None else str(args.avernet.expanduser().resolve()),
        "labels": {
            "gt": args.gt_label if args.gt is not None else None,
            "original_swiftvr": args.original_label,
            "b1": args.b1_label,
            "avernet": args.avernet_label if args.avernet is not None else None,
        },
        "source_kinds": {label: source.kind for label, source in sources},
        "lq_resolution": [lq.width, lq.height],
        "target_resolution": [target_w, target_h],
        "scale_ratio": [target_w / lq.width, target_h / lq.height],
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
