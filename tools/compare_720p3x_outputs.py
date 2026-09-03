#!/usr/bin/env python3
"""Create research-oriented visual comparisons for 720p -> 3x restoration.

Inputs may be video files or image-sequence directories. For publication-quality
inspection, prefer PNG directories from SwiftVR ``--png`` inference so codec
artifacts cannot hide or invent high-frequency detail.

Two interfaces are supported:

1. Generic repeated methods (recommended):

   --method "M1 Stage-A=/path/to/m1" \
   --method "M3 D1536=/path/to/m3" \
   --method "M7-A=/path/to/m7a"

2. Historical ``--original-swiftvr`` / ``--b1`` / ``--avernet`` arguments,
   retained for backward compatibility.

The first restoration method defines the target resolution. The tool writes a
labeled comparison video, selected-frame PNGs, and optional native target-space
crop strips.  LQ is bicubic-resized only for visualization; it is never used as
a quantitative reference.
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


def _parse_method(value: str) -> tuple[str, Path]:
    label, sep, raw_path = value.partition("=")
    if not sep or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("method must be LABEL=PATH")
    return label.strip(), Path(raw_path.strip())


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
    if columns <= 0:
        raise ValueError("columns must be positive")
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
    p.add_argument("--lq", type=Path, required=True, help="Original LQ video or image directory.")
    p.add_argument("--gt", type=Path, default=None,
                   help="Optional paired GT video/image directory at target resolution.")
    p.add_argument("--gt-label", default="GT")
    p.add_argument(
        "--method",
        type=_parse_method,
        action="append",
        default=[],
        help="Generic restoration source as LABEL=PATH; repeat for any number of methods.",
    )

    # Backward-compatible historical interface.
    p.add_argument("--original-swiftvr", type=Path, default=None)
    p.add_argument("--original-label", default="Original SwiftVR")
    p.add_argument("--b1", type=Path, default=None,
                   help="Historical candidate video/image directory; typically B1 Slim100.")
    p.add_argument("--b1-label", default="B1 Slim100")
    p.add_argument("--avernet", type=Path, default=None,
                   help="Historical optional additional method video/image directory.")
    p.add_argument("--avernet-label", default="AVerNet")

    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--panel-width", type=int, default=960)
    p.add_argument("--columns", type=int, default=2,
                   help="Number of columns in the full-frame comparison montage.")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--frame-indices", type=_csv_ints, default=(0, 8, 16, 24, 32))
    p.add_argument("--crop", type=_parse_crop, action="append", default=[])
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--quality", type=int, default=8)
    return p


def _collect_method_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    specs = list(args.method)
    legacy = (
        (args.original_label, args.original_swiftvr),
        (args.b1_label, args.b1),
        (args.avernet_label, args.avernet),
    )
    specs.extend((label, path) for label, path in legacy if path is not None)
    if not specs:
        raise ValueError(
            "provide at least one restoration source via --method LABEL=PATH or the legacy arguments"
        )
    labels = [label for label, _ in specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"method labels must be unique, got {labels}")
    return specs


def main() -> int:
    args = build_parser().parse_args()
    if args.panel_width <= 0 or args.columns <= 0:
        raise ValueError("--panel-width/--columns must be positive")
    if args.quality <= 0:
        raise ValueError("--quality must be positive")

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    lq = FrameSource(args.lq)
    method_specs = _collect_method_specs(args)
    methods: list[tuple[str, FrameSource]] = []
    for label, path in method_specs:
        methods.append((label, FrameSource(path, fallback_fps=lq.fps)))

    target_w, target_h = methods[0][1].width, methods[0][1].height
    for label, source in methods[1:]:
        if (source.width, source.height) != (target_w, target_h):
            raise ValueError(
                f"{label} resolution {source.width}x{source.height} != "
                f"first method target {target_w}x{target_h}"
            )

    gt = None
    if args.gt is not None:
        gt = FrameSource(args.gt, fallback_fps=methods[0][1].fps)
        if (gt.width, gt.height) != (target_w, target_h):
            raise ValueError(f"GT resolution {gt.width}x{gt.height} != target {target_w}x{target_h}")

    sources = [("LQ", lq)] + methods
    if gt is not None:
        sources.append((args.gt_label, gt))
    frame_counts = {label: len(source) for label, source in sources}
    common = min(frame_counts.values())
    if args.max_frames > 0:
        common = min(common, int(args.max_frames))
    if common <= 0:
        raise RuntimeError("no common frames")
    output_fps = float(args.fps or methods[0][1].fps or lq.fps)

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
            if gt is not None:
                method_frames.append((args.gt_label, gt.frame(index)))

            montage = _grid(
                [_labeled_panel(frame, label, args.panel_width) for label, frame in method_frames],
                columns=args.columns,
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
        "gt": None if gt is None else str(gt.path),
        "methods": [
            {"label": label, "path": str(source.path), "kind": source.kind}
            for label, source in methods
        ],
        "source_kinds": {label: source.kind for label, source in sources},
        "lq_resolution": [lq.width, lq.height],
        "target_resolution": [target_w, target_h],
        "scale_ratio": [target_w / lq.width, target_h / lq.height],
        "frame_counts": frame_counts,
        "compared_frames": common,
        "fps": output_fps,
        "selected_frames": sorted(selected),
        "crops": [list(crop) for crop in args.crop],
        "panel_width": args.panel_width,
        "columns": args.columns,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
