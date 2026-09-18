#!/usr/bin/env python3
"""Create research-oriented visual comparisons for 720p -> 3x restoration.

Inputs may be video files or image-sequence directories. For publication-quality
inspection, prefer PNG directories from SwiftVR ``--png`` inference so codec
artifacts cannot hide or invent high-frequency detail.

Synchronization is deliberately FRAME-INDEX based. Input FPS, duration, PTS/DTS,
and other container timing metadata are ignored when choosing corresponding
frames. By default, frame N from every source is compared with frame N from every
other source. The historical BasicCNN input can optionally use an explicit ordinal
mapping ``basiccnn_index = N * scale + offset`` for outputs that contain an
integer multiple of frames (for example 60-FPS / 2x-frame-count results compared
against a 30-FPS source). ``--fps`` controls only the playback rate of generated
comparison videos and never changes source-frame sampling.

Two interfaces are supported:

1. Generic repeated methods (recommended):

   --method "M1 Stage-A=/path/to/m1" \
   --method "M3 D1536=/path/to/m3" \
   --method "M8-A=/path/to/m8a"

2. Historical ``--original-swiftvr`` / ``--b1`` / ``--avernet`` / ``--basiccnn``
   arguments, retained for backward compatibility. BasicCNN is always appended
   after the other historical methods so a six-panel comparison naturally ends
   with BasicCNN. ``--basiccnn-index-scale`` and ``--basiccnn-index-offset`` apply
   only to this historical BasicCNN source.

The first restoration method defines the target resolution. The tool writes a
labeled comparison video, selected-frame PNGs, and optional native target-space
crop strips. LQ is bicubic-resized only for visualization; it is never used as
a quantitative reference. If ``--columns`` is omitted, six-panel comparisons
use three columns (2x3); all other comparisons use two columns.
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
    """Frame-addressable source whose public indexing is always ordinal.

    For videos, ``frame(i)`` means the i-th decoded frame in presentation order.
    We intentionally do not convert frame numbers through FPS or timestamps.
    For image directories, files are sorted numerically by stem when possible,
    then lexicographically as a fallback.
    """

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
            # Native FPS is metadata only. It never affects source-frame alignment.
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

    def _check_index(self, index: int) -> int:
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(f"frame index {index} outside [0, {len(self)}) for {self.path}")
        return index

    def _video_frame(self, index: int) -> np.ndarray:
        # Integer VideoReader indexing addresses decoded frames by ordinal index.
        # Never seek through timestamps or derive an index from FPS.
        value = self.reader[int(index)]
        if hasattr(value, "asnumpy"):
            value = value.asnumpy()
        return np.asarray(value, dtype=np.uint8)

    def frame(self, index: int) -> np.ndarray:
        index = self._check_index(index)
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


def _logical_frame_count(source: FrameSource, scale: int, offset: int) -> int:
    """Return how many logical frame indices are valid under an ordinal mapping.

    Logical frame N addresses source frame ``N * scale + offset``. Scale must be
    positive and offset non-negative, so logical frame 0 is always the first
    sampled source frame and the valid logical range is contiguous.
    """

    if scale <= 0:
        raise ValueError("frame-index scale must be positive")
    if offset < 0:
        raise ValueError("frame-index offset must be non-negative")
    if offset >= len(source):
        return 0
    return 1 + (len(source) - 1 - offset) // scale


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

    # Backward-compatible historical interface. Keep BasicCNN last.
    p.add_argument("--original-swiftvr", type=Path, default=None)
    p.add_argument("--original-label", default="Original SwiftVR")
    p.add_argument("--b1", type=Path, default=None,
                   help="Historical candidate video/image directory; typically B1 Slim100.")
    p.add_argument("--b1-label", default="B1 Slim100")
    p.add_argument("--avernet", type=Path, default=None,
                   help="Historical optional additional method video/image directory.")
    p.add_argument("--avernet-label", default="AVerNet")
    p.add_argument("--basiccnn", type=Path, default=None,
                   help="Optional BasicCNN video/image-sequence directory; appended last.")
    p.add_argument("--basiccnn-label", default="BasicCNN")
    p.add_argument(
        "--basiccnn-index-scale",
        type=int,
        default=1,
        help=(
            "Ordinal BasicCNN frame multiplier. Logical frame N reads BasicCNN "
            "frame N*scale+offset. Use 2 for 2x-frame-count outputs. Default: 1."
        ),
    )
    p.add_argument(
        "--basiccnn-index-offset",
        type=int,
        default=0,
        help=(
            "Non-negative ordinal BasicCNN frame offset used with "
            "--basiccnn-index-scale. Default: 0."
        ),
    )

    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--panel-width", type=int, default=960)
    p.add_argument(
        "--columns",
        type=int,
        default=None,
        help="Full-frame montage columns. Default: 3 for exactly six panels, otherwise 2.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Playback FPS of generated comparison videos only. Source alignment "
            "always uses ordinal frame mapping and ignores source FPS/timestamps."
        ),
    )
    p.add_argument("--frame-indices", type=_csv_ints, default=(0, 8, 16, 24, 32))
    p.add_argument("--crop", type=_parse_crop, action="append", default=[])
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--quality", type=int, default=8)
    return p


def _collect_method_specs(args: argparse.Namespace) -> list[tuple[str, Path, int, int]]:
    specs: list[tuple[str, Path, int, int]] = [
        (label, path, 1, 0) for label, path in args.method
    ]
    legacy = (
        (args.original_label, args.original_swiftvr, 1, 0),
        (args.b1_label, args.b1, 1, 0),
        (args.avernet_label, args.avernet, 1, 0),
        (
            args.basiccnn_label,
            args.basiccnn,
            int(args.basiccnn_index_scale),
            int(args.basiccnn_index_offset),
        ),
    )
    specs.extend(
        (label, path, scale, offset)
        for label, path, scale, offset in legacy
        if path is not None
    )
    if not specs:
        raise ValueError(
            "provide at least one restoration source via --method LABEL=PATH or the legacy arguments"
        )
    labels = [label for label, _, _, _ in specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"method labels must be unique, got {labels}")
    return specs


def main() -> int:
    args = build_parser().parse_args()
    if args.panel_width <= 0:
        raise ValueError("--panel-width must be positive")
    if args.columns is not None and args.columns <= 0:
        raise ValueError("--columns must be positive when provided")
    if args.quality <= 0:
        raise ValueError("--quality must be positive")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive when provided")
    if args.basiccnn_index_scale <= 0:
        raise ValueError("--basiccnn-index-scale must be positive")
    if args.basiccnn_index_offset < 0:
        raise ValueError("--basiccnn-index-offset must be non-negative")
    if args.basiccnn is None and (
        args.basiccnn_index_scale != 1 or args.basiccnn_index_offset != 0
    ):
        raise ValueError(
            "--basiccnn-index-scale/--basiccnn-index-offset require --basiccnn"
        )

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    lq = FrameSource(args.lq)
    method_specs = _collect_method_specs(args)
    methods: list[tuple[str, FrameSource, int, int]] = []
    for label, path, scale, offset in method_specs:
        methods.append((label, FrameSource(path, fallback_fps=lq.fps), scale, offset))

    target_w, target_h = methods[0][1].width, methods[0][1].height
    for label, source, _, _ in methods[1:]:
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

    # Each source exposes a logical frame sequence. LQ, GT, and ordinary methods
    # use identity mapping. BasicCNN may use N*scale+offset explicitly.
    mapped_sources: list[tuple[str, FrameSource, int, int]] = [("LQ", lq, 1, 0)] + methods
    if gt is not None:
        mapped_sources.append((args.gt_label, gt, 1, 0))

    source_frame_counts = {label: len(source) for label, source, _, _ in mapped_sources}
    logical_frame_counts = {
        label: _logical_frame_count(source, scale, offset)
        for label, source, scale, offset in mapped_sources
    }
    common = min(logical_frame_counts.values())
    if args.max_frames > 0:
        common = min(common, int(args.max_frames))
    if common <= 0:
        raise RuntimeError("no common logical frame indices under the requested mappings")

    # Output FPS controls display speed only. It is never used to map input
    # frame numbers. Explicit --fps is recommended for milestone comparisons.
    output_fps = float(args.fps if args.fps is not None else lq.fps)

    panel_count = 1 + len(methods) + (1 if gt is not None else 0)
    columns = int(args.columns) if args.columns is not None else (3 if panel_count == 6 else 2)

    print("Ordinal frame-index synchronization (FPS/timestamps ignored):", flush=True)
    for label, source, scale, offset in mapped_sources:
        mapping = f"source_index=N*{scale}+{offset}"
        print(
            f"  {label}: frames={len(source)} logical_frames={logical_frame_counts[label]} "
            f"native_fps={source.fps:.6g} mapping={mapping} "
            f"kind={source.kind} path={source.path}",
            flush=True,
        )
    print(
        f"  comparing logical frame indices 0..{common - 1}; "
        f"output playback fps={output_fps:.6g}",
        flush=True,
    )

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
            # 'index' is the logical/source-reference frame number. Each method
            # maps it deterministically to its own decoded ordinal frame.
            lq_up = _resize_rgb(lq.frame(index), target_w, target_h)
            method_frames = [("LQ Bicubic 3x", lq_up)] + [
                (label, source.frame(index * scale + offset))
                for label, source, scale, offset in methods
            ]
            if gt is not None:
                method_frames.append((args.gt_label, gt.frame(index)))

            montage = _grid(
                [_labeled_panel(frame, label, args.panel_width) for label, frame in method_frames],
                columns=columns,
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
        "alignment": {
            "mode": "ordinal_frame_index",
            "description": (
                "logical frame N maps to source frame N*index_scale+index_offset; "
                "source FPS/timestamps are ignored"
            ),
            "common_frame_count": common,
            "first_frame_index": 0,
            "last_frame_index": common - 1,
        },
        "lq": str(lq.path),
        "gt": None if gt is None else str(gt.path),
        "methods": [
            {
                "label": label,
                "path": str(source.path),
                "kind": source.kind,
                "frame_count": len(source),
                "logical_frame_count": _logical_frame_count(source, scale, offset),
                "native_fps": source.fps,
                "index_scale": scale,
                "index_offset": offset,
                "mapping": f"N*{scale}+{offset}",
            }
            for label, source, scale, offset in methods
        ],
        "source_kinds": {
            label: source.kind for label, source, _, _ in mapped_sources
        },
        "source_frame_counts": source_frame_counts,
        "source_logical_frame_counts": logical_frame_counts,
        "source_native_fps": {
            label: source.fps for label, source, _, _ in mapped_sources
        },
        "source_index_mappings": {
            label: {"scale": scale, "offset": offset}
            for label, _, scale, offset in mapped_sources
        },
        "lq_resolution": [lq.width, lq.height],
        "target_resolution": [target_w, target_h],
        "scale_ratio": [target_w / lq.width, target_h / lq.height],
        "compared_frames": common,
        "output_fps": output_fps,
        "selected_frames": sorted(selected),
        "crops": [list(crop) for crop in args.crop],
        "panel_width": args.panel_width,
        "panel_count": panel_count,
        "columns": columns,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())