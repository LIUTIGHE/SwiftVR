#!/usr/bin/env python3
"""Audit temporal, geometric, and photometric alignment of HR/HQ/LR triplets.

The input is a JSONL manifest produced by ``tools/build_triplet_manifest.py``.
Both encoded-video records and image-sequence records are supported. Image
sequences may use compact ``{frame:06d}`` patterns or explicit ordered
``hr_frames``/``hq_frames``/``lr_frames`` lists, including manifests created by
``--group-components``.

The tool never modifies source media. For encoded videos it uses ffprobe and
lazily imported decord. For image sequences it reads RGB images with Pillow.
It samples a small number of frames to:

* compare frame counts and dimensions (plus container metadata for videos);
* search small temporal offsets for HR<->HQ and HQ<->LR;
* report MAE, RMSE, PSNR, and global grayscale SSIM for HR->HQ and HQ->LR.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

MEDIA_FIELDS = ("hr", "hq", "lr")
VALID_MEDIA_TYPES = ("video", "frames")


@dataclass(frozen=True)
class MediaProbe:
    media_type: str
    path: str
    width: int | None
    height: int | None
    frame_count: int | None
    avg_fps: float | None
    nominal_fps: float | None
    duration: float | None
    pix_fmt: str | None
    color_range: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    timestamp_count: int
    timestamp_interval_median: float | None
    timestamp_interval_cv: float | None
    timestamp_interval_min: float | None
    timestamp_interval_max: float | None
    duplicate_timestamp_count: int
    probe_warnings: tuple[str, ...]


@dataclass(frozen=True)
class MediaSource:
    media_type: str
    video_path: str | None = None
    frame_paths: tuple[str, ...] = ()
    frame_indices: tuple[int, ...] = ()
    frame_path_mode: str | None = None

    @property
    def frame_count(self) -> int | None:
        if self.media_type == "frames":
            return len(self.frame_paths)
        return None


@dataclass(frozen=True)
class PairMetrics:
    mae: float
    rmse: float
    psnr: float
    ssim_gray_global: float


def _fraction_to_float(value: str | int | float | None) -> float | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        try:
            result = float(Fraction(str(value)))
        except (ValueError, ZeroDivisionError):
            return None
    return result if math.isfinite(result) and result > 0 else None


def _int_or_none(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _run_json_command(command: Sequence[str]) -> Mapping[str, object]:
    completed = subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(completed.stdout or "{}")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object from command: {' '.join(command)}")
    return payload


def _timestamp_stats(timestamps: Sequence[float]) -> dict[str, float | int | None]:
    clean = np.asarray(
        [value for value in timestamps if math.isfinite(value)], dtype=np.float64
    )
    if clean.size < 2:
        return {
            "timestamp_count": int(clean.size),
            "timestamp_interval_median": None,
            "timestamp_interval_cv": None,
            "timestamp_interval_min": None,
            "timestamp_interval_max": None,
            "duplicate_timestamp_count": 0,
        }
    intervals = np.diff(clean)
    duplicate_count = int(np.count_nonzero(intervals <= 1e-9))
    positive = intervals[intervals > 1e-9]
    if positive.size == 0:
        median = cv = minimum = maximum = None
    else:
        median = float(np.median(positive))
        mean = float(np.mean(positive))
        cv = float(np.std(positive) / mean) if mean > 0 else None
        minimum = float(np.min(positive))
        maximum = float(np.max(positive))
    return {
        "timestamp_count": int(clean.size),
        "timestamp_interval_median": median,
        "timestamp_interval_cv": cv,
        "timestamp_interval_min": minimum,
        "timestamp_interval_max": maximum,
        "duplicate_timestamp_count": duplicate_count,
    }


def probe_video(path: Path, *, timestamp_scan_limit: int = 256) -> MediaProbe:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe was not found on PATH")

    warnings: list[str] = []
    metadata = _run_json_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,"
                "duration,pix_fmt,color_range,color_space,color_transfer,"
                "color_primaries:format=duration"
            ),
            "-of",
            "json",
            str(path),
        ]
    )
    streams = metadata.get("streams", [])
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"No video stream found: {path}")
    stream = streams[0]
    if not isinstance(stream, Mapping):
        raise ValueError(f"Malformed ffprobe stream payload: {path}")
    format_payload = metadata.get("format", {})
    if not isinstance(format_payload, Mapping):
        format_payload = {}

    duration = _float_or_none(stream.get("duration"))
    if duration is None:
        duration = _float_or_none(format_payload.get("duration"))

    timestamp_values: list[float] = []
    if timestamp_scan_limit > 1:
        try:
            frame_payload = _run_json_command(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-read_intervals",
                    f"%+#{int(timestamp_scan_limit)}",
                    "-show_entries",
                    "frame=best_effort_timestamp_time",
                    "-of",
                    "json",
                    str(path),
                ]
            )
            frames = frame_payload.get("frames", [])
            if isinstance(frames, list):
                for frame in frames:
                    if isinstance(frame, Mapping):
                        value = _float_or_none(frame.get("best_effort_timestamp_time"))
                        if value is not None:
                            timestamp_values.append(value)
        except (subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"timestamp_scan_failed:{type(exc).__name__}")

    stats = _timestamp_stats(timestamp_values)
    return MediaProbe(
        media_type="video",
        path=str(path),
        width=_int_or_none(stream.get("width")),
        height=_int_or_none(stream.get("height")),
        frame_count=_int_or_none(stream.get("nb_frames")),
        avg_fps=_fraction_to_float(stream.get("avg_frame_rate")),
        nominal_fps=_fraction_to_float(stream.get("r_frame_rate")),
        duration=duration,
        pix_fmt=str(stream.get("pix_fmt")) if stream.get("pix_fmt") else None,
        color_range=str(stream.get("color_range")) if stream.get("color_range") else None,
        color_space=str(stream.get("color_space")) if stream.get("color_space") else None,
        color_transfer=(
            str(stream.get("color_transfer")) if stream.get("color_transfer") else None
        ),
        color_primaries=(
            str(stream.get("color_primaries")) if stream.get("color_primaries") else None
        ),
        probe_warnings=tuple(warnings),
        **stats,
    )


def _read_rgb_image(path: str | Path) -> np.ndarray:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image frame does not exist: {resolved}")
    with Image.open(resolved) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def probe_frame_sequence(source: MediaSource) -> MediaProbe:
    if source.media_type != "frames" or not source.frame_paths:
        raise ValueError("Frame-sequence probe requires non-empty frame paths")
    first = _read_rgb_image(source.frame_paths[0])
    return MediaProbe(
        media_type="frames",
        path=source.frame_paths[0],
        width=int(first.shape[1]),
        height=int(first.shape[0]),
        frame_count=len(source.frame_paths),
        avg_fps=None,
        nominal_fps=None,
        duration=None,
        pix_fmt="rgb24",
        color_range=None,
        color_space=None,
        color_transfer=None,
        color_primaries=None,
        timestamp_count=0,
        timestamp_interval_median=None,
        timestamp_interval_cv=None,
        timestamp_interval_min=None,
        timestamp_interval_max=None,
        duplicate_timestamp_count=0,
        probe_warnings=(),
    )


def _load_decord_reader(path: str):
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError(
            "decord is required for encoded-video decoding; install dependencies"
        ) from exc
    return decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)


def decode_video_frames(path: str, positions: Sequence[int]) -> list[np.ndarray]:
    reader = _load_decord_reader(path)
    if len(reader) <= 0:
        raise ValueError(f"Video has no decodable frames: {path}")
    clipped = [min(max(int(position), 0), len(reader) - 1) for position in positions]
    batch = reader.get_batch(clipped).asnumpy()
    if batch.ndim != 4 or batch.shape[-1] < 3:
        raise ValueError(f"Unexpected decoded frame shape {batch.shape}: {path}")
    return [np.asarray(frame[..., :3], dtype=np.uint8) for frame in batch]


def decode_source_frames(source: MediaSource, positions: Sequence[int]) -> list[np.ndarray]:
    positions = [int(position) for position in positions]
    if source.media_type == "video":
        if source.video_path is None:
            raise ValueError("Video source is missing video_path")
        return decode_video_frames(source.video_path, positions)
    if source.media_type != "frames":
        raise ValueError(f"Unsupported media type: {source.media_type}")
    count = len(source.frame_paths)
    frames: list[np.ndarray] = []
    for position in positions:
        if position < 0 or position >= count:
            raise IndexError(
                f"Frame position {position} is outside image sequence of length {count}"
            )
        frames.append(_read_rgb_image(source.frame_paths[position]))
    return frames


def _infer_media_type(record: Mapping[str, object]) -> str:
    declared = record.get("media_type")
    if declared in VALID_MEDIA_TYPES:
        return str(declared)
    if any(f"{name}_frames" in record for name in MEDIA_FIELDS):
        return "frames"
    if "frame_start" in record or "frame_indices" in record:
        return "frames"
    return "video"


def _coerce_int_sequence(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a JSON array")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain integers") from exc


def _coerce_path_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a JSON array")
    paths = tuple(str(Path(str(item)).resolve()) for item in value)
    if not paths:
        raise ValueError(f"{field} must not be empty")
    return paths


def _pattern_indices(record: Mapping[str, object]) -> tuple[int, ...]:
    if "frame_indices" in record:
        indices = _coerce_int_sequence(record["frame_indices"], field="frame_indices")
    else:
        start = _int_or_none(record.get("frame_start"))
        end = _int_or_none(record.get("frame_end"))
        if start is None or end is None or end < start:
            raise ValueError(
                "Pattern frame manifest requires frame_indices or valid frame_start/frame_end"
            )
        indices = tuple(range(start, end + 1))
    frame_count = _int_or_none(record.get("frame_count"))
    if frame_count is not None and frame_count != len(indices):
        raise ValueError(
            f"frame_count={frame_count} does not match {len(indices)} frame indices"
        )
    if len(set(indices)) != len(indices):
        raise ValueError("frame_indices contains duplicates")
    return indices


def resolve_media_sources(record: Mapping[str, object]) -> dict[str, MediaSource]:
    media_type = _infer_media_type(record)
    if media_type == "video":
        missing = [name for name in MEDIA_FIELDS if name not in record]
        if missing:
            raise ValueError(f"Video manifest record missing fields: {missing}")
        return {
            name: MediaSource(
                media_type="video", video_path=str(Path(str(record[name])).resolve())
            )
            for name in MEDIA_FIELDS
        }

    explicit_fields = tuple(f"{name}_frames" for name in MEDIA_FIELDS)
    has_explicit = [field in record for field in explicit_fields]
    if any(has_explicit) and not all(has_explicit):
        missing = [field for field, present in zip(explicit_fields, has_explicit) if not present]
        raise ValueError(f"Explicit frame manifest missing fields: {missing}")

    indices = _pattern_indices(record)
    if all(has_explicit):
        sources: dict[str, MediaSource] = {}
        for name in MEDIA_FIELDS:
            paths = _coerce_path_sequence(record[f"{name}_frames"], field=f"{name}_frames")
            if len(paths) != len(indices):
                raise ValueError(
                    f"{name}_frames length {len(paths)} does not match "
                    f"frame_indices length {len(indices)}"
                )
            sources[name] = MediaSource(
                media_type="frames",
                frame_paths=paths,
                frame_indices=indices,
                frame_path_mode="explicit",
            )
        return sources

    missing = [name for name in MEDIA_FIELDS if name not in record]
    if missing:
        raise ValueError(f"Pattern frame manifest missing fields: {missing}")
    sources = {}
    for name in MEDIA_FIELDS:
        pattern = str(record[name])
        try:
            paths = tuple(str(Path(pattern.format(frame=index)).resolve()) for index in indices)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid frame pattern for {name}: {pattern}") from exc
        sources[name] = MediaSource(
            media_type="frames",
            frame_paths=paths,
            frame_indices=indices,
            frame_path_mode="pattern",
        )
    return sources


def probe_source(source: MediaSource, *, timestamp_scan_limit: int) -> MediaProbe:
    if source.media_type == "video":
        if source.video_path is None:
            raise ValueError("Video source is missing video_path")
        return probe_video(Path(source.video_path), timestamp_scan_limit=timestamp_scan_limit)
    return probe_frame_sequence(source)


def resize_rgb(frame: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = (int(size_hw[0]), int(size_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid target size: {size_hw}")
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    resized = image.resize((width, height), resample=Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.uint8)


def _thumbnail_gray(frame: np.ndarray, max_side: int = 256) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    target = (max(1, round(height * scale)), max(1, round(width * scale)))
    small = resize_rgb(frame, target) if target != (height, width) else frame
    gray = (
        0.299 * small[..., 0].astype(np.float32)
        + 0.587 * small[..., 1].astype(np.float32)
        + 0.114 * small[..., 2].astype(np.float32)
    )
    return gray / 255.0


def pair_metrics(a: np.ndarray, b: np.ndarray) -> PairMetrics:
    if a.shape != b.shape:
        raise ValueError(f"Metric inputs must have identical shapes: {a.shape} vs {b.shape}")
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    diff = a32 - b32
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(255.0 / rmse)

    gray_a = 0.299 * a32[..., 0] + 0.587 * a32[..., 1] + 0.114 * a32[..., 2]
    gray_b = 0.299 * b32[..., 0] + 0.587 * b32[..., 1] + 0.114 * b32[..., 2]
    mean_a = float(np.mean(gray_a))
    mean_b = float(np.mean(gray_b))
    var_a = float(np.var(gray_a))
    var_b = float(np.var(gray_b))
    covariance = float(np.mean((gray_a - mean_a) * (gray_b - mean_b)))
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    denominator = (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    if denominator == 0:
        ssim = 1.0 if np.array_equal(a, b) else 0.0
    else:
        ssim = (
            (2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2)
        ) / denominator
    return PairMetrics(mae=mae, rmse=rmse, psnr=psnr, ssim_gray_global=float(ssim))


def evenly_spaced_indices(frame_count: int, count: int, *, margin: int = 0) -> list[int]:
    if frame_count <= 0 or count <= 0:
        return []
    low = min(max(int(margin), 0), frame_count - 1)
    high = max(low, frame_count - 1 - max(int(margin), 0))
    actual = min(int(count), high - low + 1)
    if actual <= 1:
        return [int((low + high) // 2)]
    return sorted({int(round(value)) for value in np.linspace(low, high, actual)})


def find_best_offset_from_sequences(
    source_frames: Mapping[int, np.ndarray],
    reference_frames: Mapping[int, np.ndarray],
    anchors: Sequence[int],
    offsets: Iterable[int],
    *,
    max_side: int = 256,
) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    for offset in offsets:
        errors: list[float] = []
        for anchor in anchors:
            source_position = int(anchor) + int(offset)
            if source_position not in source_frames or int(anchor) not in reference_frames:
                continue
            reference = reference_frames[int(anchor)]
            source = resize_rgb(source_frames[source_position], reference.shape[:2])
            source_gray = _thumbnail_gray(source, max_side=max_side)
            reference_gray = _thumbnail_gray(reference, max_side=max_side)
            errors.append(float(np.mean(np.abs(source_gray - reference_gray))))
        if errors:
            scores[int(offset)] = float(np.mean(errors))
    if not scores:
        raise ValueError("No valid frame pairs were available for offset search")
    best = min(scores, key=lambda value: (scores[value], abs(value), value))
    return int(best), scores


def _mean_pair_metrics(metrics: Sequence[PairMetrics]) -> dict[str, float | None] | None:
    if not metrics:
        return None
    result: dict[str, float | None] = {}
    for field in ("mae", "rmse", "psnr", "ssim_gray_global"):
        values = [float(getattr(item, field)) for item in metrics]
        finite = [value for value in values if math.isfinite(value)]
        result[field] = float(np.mean(finite)) if finite else None
    return result


def _relative_difference(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def _append_unique(values: list[str], message: str) -> None:
    if message not in values:
        values.append(message)


def _source_count(source: MediaSource, probe: MediaProbe) -> int:
    if source.media_type == "frames":
        return len(source.frame_paths)
    if probe.frame_count is not None:
        return probe.frame_count
    if source.video_path is None:
        raise ValueError("Video source is missing video_path")
    return len(_load_decord_reader(source.video_path))


def audit_record(
    record: Mapping[str, object],
    *,
    expected_scale: float | None,
    scale_tolerance: float,
    fps_tolerance: float,
    duration_tolerance: float,
    offset_radius: int,
    sample_frames: int,
    timestamp_scan_limit: int,
    metric_thumbnail_max_side: int,
) -> dict[str, object]:
    sample_id = str(record.get("sample_id", ""))
    if not sample_id:
        raise ValueError("Manifest record is missing sample_id")
    sources = resolve_media_sources(record)
    media_types = {source.media_type for source in sources.values()}
    if len(media_types) != 1:
        raise ValueError(f"Triplet mixes media types: {sorted(media_types)}")
    media_type = next(iter(media_types))
    probes = {
        name: probe_source(source, timestamp_scan_limit=timestamp_scan_limit)
        for name, source in sources.items()
    }

    warnings: list[str] = []
    errors: list[str] = []
    if media_type == "video":
        for name, probe in probes.items():
            for warning in probe.probe_warnings:
                _append_unique(warnings, f"{name}:{warning}")
            if probe.timestamp_interval_cv is not None and probe.timestamp_interval_cv > 0.01:
                _append_unique(warnings, f"{name}:possible_vfr")
            if probe.duplicate_timestamp_count:
                _append_unique(warnings, f"{name}:duplicate_timestamps")

        for field in ("color_range", "color_space", "color_transfer", "color_primaries"):
            known = {
                getattr(probe, field)
                for probe in probes.values()
                if getattr(probe, field)
            }
            if len(known) > 1:
                _append_unique(warnings, f"{field}_mismatch")

    hq, lr, hr = probes["hq"], probes["lr"], probes["hr"]
    if None not in (hq.width, hq.height, lr.width, lr.height):
        if (hq.width, hq.height) != (lr.width, lr.height):
            errors.append("hq_lr_resolution_mismatch")
    else:
        warnings.append("missing_hq_or_lr_resolution_metadata")

    scale_x = scale_y = None
    if None not in (hr.width, hr.height, hq.width, hq.height) and hq.width and hq.height:
        scale_x = float(hr.width) / float(hq.width)
        scale_y = float(hr.height) / float(hq.height)
        if abs(scale_x - scale_y) > scale_tolerance:
            errors.append("non_uniform_hr_to_hq_scale")
        if expected_scale is not None and (
            abs(scale_x - expected_scale) > scale_tolerance
            or abs(scale_y - expected_scale) > scale_tolerance
        ):
            errors.append("unexpected_hr_to_hq_scale")
    else:
        warnings.append("missing_hr_or_hq_resolution_metadata")

    for left, right in (("hr", "hq"), ("hq", "lr")):
        a, b = probes[left], probes[right]
        if a.frame_count is not None and b.frame_count is not None:
            difference = abs(a.frame_count - b.frame_count)
            if difference > 1:
                errors.append(f"{left}_{right}_frame_count_mismatch")
            elif difference == 1:
                warnings.append(f"{left}_{right}_frame_count_off_by_one")
        else:
            warnings.append(f"{left}_{right}_missing_frame_count")
        if media_type == "video":
            fps_difference = _relative_difference(a.avg_fps, b.avg_fps)
            if fps_difference is not None and fps_difference > fps_tolerance:
                errors.append(f"{left}_{right}_fps_mismatch")
            duration_difference = _relative_difference(a.duration, b.duration)
            if duration_difference is not None and duration_difference > duration_tolerance:
                errors.append(f"{left}_{right}_duration_mismatch")

    counts = {name: _source_count(sources[name], probes[name]) for name in MEDIA_FIELDS}
    common_count = min(counts.values())
    anchors = evenly_spaced_indices(
        common_count, sample_frames, margin=max(0, offset_radius)
    )
    if not anchors:
        raise ValueError("Triplet has no frames available for sampling")
    offsets = range(-max(0, offset_radius), max(0, offset_radius) + 1)
    required_hr_positions = sorted(
        {
            anchor + offset
            for anchor in anchors
            for offset in offsets
            if 0 <= anchor + offset < counts["hr"]
        }
    )
    required_hq_positions = sorted(
        set(anchors)
        | {
            anchor + offset
            for anchor in anchors
            for offset in offsets
            if 0 <= anchor + offset < counts["hq"]
        }
    )
    hr_frames = dict(
        zip(required_hr_positions, decode_source_frames(sources["hr"], required_hr_positions))
    )
    hq_frames = dict(
        zip(required_hq_positions, decode_source_frames(sources["hq"], required_hq_positions))
    )
    lr_frames = dict(zip(anchors, decode_source_frames(sources["lr"], anchors)))

    best_hr_hq_offset, hr_hq_offset_scores = find_best_offset_from_sequences(
        hr_frames,
        hq_frames,
        anchors,
        offsets,
        max_side=metric_thumbnail_max_side,
    )
    best_hq_lr_offset, hq_lr_offset_scores = find_best_offset_from_sequences(
        hq_frames,
        lr_frames,
        anchors,
        offsets,
        max_side=metric_thumbnail_max_side,
    )
    if best_hr_hq_offset != 0:
        errors.append(f"hr_hq_temporal_offset:{best_hr_hq_offset:+d}")
    if best_hq_lr_offset != 0:
        errors.append(f"hq_lr_temporal_offset:{best_hq_lr_offset:+d}")

    hr_hq_metrics: list[PairMetrics] = []
    hq_lr_metrics: list[PairMetrics] = []
    sampled_duplicates = {name: 0 for name in MEDIA_FIELDS}
    previous: dict[str, np.ndarray | None] = {name: None for name in MEDIA_FIELDS}
    for anchor in anchors:
        aligned_hr = hr_frames.get(anchor + best_hr_hq_offset)
        hq_frame = hq_frames[anchor]
        aligned_hq_for_lr = hq_frames.get(anchor + best_hq_lr_offset)
        lr_frame = lr_frames[anchor]
        if aligned_hr is not None:
            hr_hq_metrics.append(
                pair_metrics(resize_rgb(aligned_hr, hq_frame.shape[:2]), hq_frame)
            )
        if aligned_hq_for_lr is not None and aligned_hq_for_lr.shape == lr_frame.shape:
            hq_lr_metrics.append(pair_metrics(aligned_hq_for_lr, lr_frame))
        for name, frame in (("hr", aligned_hr), ("hq", hq_frame), ("lr", lr_frame)):
            if frame is None:
                continue
            thumb = _thumbnail_gray(frame, max_side=64)
            if previous[name] is not None and np.mean(np.abs(thumb - previous[name])) < 1e-5:
                sampled_duplicates[name] += 1
            previous[name] = thumb

    logical_frame_indices: list[int] | None = None
    if media_type == "frames":
        logical_frame_indices = list(sources["hq"].frame_indices)
    sampled_frame_indices = (
        [logical_frame_indices[position] for position in anchors]
        if logical_frame_indices is not None
        else None
    )

    status = "fail" if errors else ("warn" if warnings else "pass")
    return {
        "sample_id": sample_id,
        "split": record.get("split"),
        "media_type": media_type,
        "frame_path_mode": record.get("frame_path_mode"),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "probes": {name: asdict(probe) for name, probe in probes.items()},
        "geometry": {
            "hr_to_hq_scale_x": scale_x,
            "hr_to_hq_scale_y": scale_y,
            "expected_scale": expected_scale,
        },
        "temporal_alignment": {
            "sampled_positions": anchors,
            "sampled_frame_indices": sampled_frame_indices,
            "best_hr_offset_relative_to_hq": best_hr_hq_offset,
            "best_hq_offset_relative_to_lr": best_hq_lr_offset,
            "hr_hq_offset_mae_scores": {
                str(offset): score for offset, score in sorted(hr_hq_offset_scores.items())
            },
            "hq_lr_offset_mae_scores": {
                str(offset): score for offset, score in sorted(hq_lr_offset_scores.items())
            },
            "sampled_adjacent_duplicate_counts": sampled_duplicates,
        },
        "metrics": {
            "hr_downsample_vs_hq": _mean_pair_metrics(hr_hq_metrics),
            "hq_vs_lr": _mean_pair_metrics(hq_lr_metrics),
        },
    }


def _validate_manifest_record(payload: Mapping[str, object], line_number: int) -> None:
    if not payload.get("sample_id"):
        raise ValueError(f"Manifest line {line_number} missing sample_id")
    try:
        resolve_media_sources(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Manifest line {line_number} is invalid: {exc}") from exc


def read_manifest(path: Path, *, split: str | None = None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Manifest line {line_number} is not a JSON object")
            _validate_manifest_record(payload, line_number)
            if split is None or payload.get("split") == split:
                records.append(payload)
    return records


def summarize_results(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    status_counts = {"pass": 0, "warn": 0, "fail": 0, "error": 0}
    media_type_counts: dict[str, int] = {}
    frame_path_mode_counts: dict[str, int] = {}
    best_hr_hq_offsets: dict[str, int] = {}
    best_hq_lr_offsets: dict[str, int] = {}
    metric_values: dict[str, list[float]] = {
        "hr_downsample_vs_hq_psnr": [],
        "hq_vs_lr_psnr": [],
    }
    for result in results:
        status = str(result.get("status", "error"))
        status_counts[status if status in status_counts else "error"] += 1
        media_type = result.get("media_type")
        if media_type:
            key = str(media_type)
            media_type_counts[key] = media_type_counts.get(key, 0) + 1
        path_mode = result.get("frame_path_mode")
        if path_mode:
            key = str(path_mode)
            frame_path_mode_counts[key] = frame_path_mode_counts.get(key, 0) + 1
        temporal = result.get("temporal_alignment")
        if isinstance(temporal, Mapping):
            offset = temporal.get("best_hr_offset_relative_to_hq")
            if isinstance(offset, int):
                key = str(offset)
                best_hr_hq_offsets[key] = best_hr_hq_offsets.get(key, 0) + 1
            offset = temporal.get("best_hq_offset_relative_to_lr")
            if isinstance(offset, int):
                key = str(offset)
                best_hq_lr_offsets[key] = best_hq_lr_offsets.get(key, 0) + 1
        metrics = result.get("metrics")
        if isinstance(metrics, Mapping):
            for pair_name, output_name in (
                ("hr_downsample_vs_hq", "hr_downsample_vs_hq_psnr"),
                ("hq_vs_lr", "hq_vs_lr_psnr"),
            ):
                pair = metrics.get(pair_name)
                if isinstance(pair, Mapping):
                    value = pair.get("psnr")
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        metric_values[output_name].append(float(value))
    return {
        "sample_count": len(results),
        "status_counts": status_counts,
        "media_type_counts": media_type_counts,
        "frame_path_mode_counts": frame_path_mode_counts,
        "best_hr_hq_offset_counts": best_hr_hq_offsets,
        "best_hq_lr_offset_counts": best_hq_lr_offsets,
        "mean_metrics": {
            name: (float(np.mean(values)) if values else None)
            for name, values in metric_values.items()
        },
    }


def write_results(
    results: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    output: Path,
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--expected-scale", type=float, default=3.0)
    parser.add_argument("--scale-tolerance", type=float, default=0.01)
    parser.add_argument("--fps-tolerance", type=float, default=1e-3)
    parser.add_argument("--duration-tolerance", type=float, default=0.01)
    parser.add_argument("--offset-radius", type=int, default=5)
    parser.add_argument("--sample-frames", type=int, default=5)
    parser.add_argument("--timestamp-scan-limit", type=int, default=256)
    parser.add_argument("--metric-thumbnail-max-side", type=int, default=256)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_manifest(args.manifest, split=args.split)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise RuntimeError("No manifest records matched the requested split")

    results: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        sample_id = str(record.get("sample_id", f"record_{index}"))
        print(f"[{index}/{len(records)}] auditing {sample_id}", flush=True)
        try:
            result = audit_record(
                record,
                expected_scale=args.expected_scale,
                scale_tolerance=args.scale_tolerance,
                fps_tolerance=args.fps_tolerance,
                duration_tolerance=args.duration_tolerance,
                offset_radius=args.offset_radius,
                sample_frames=args.sample_frames,
                timestamp_scan_limit=args.timestamp_scan_limit,
                metric_thumbnail_max_side=args.metric_thumbnail_max_side,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            result = {
                "sample_id": sample_id,
                "split": record.get("split"),
                "media_type": _infer_media_type(record),
                "frame_path_mode": record.get("frame_path_mode"),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)

    summary = summarize_results(results)
    summary.update(
        {
            "manifest": str(args.manifest.resolve()),
            "output": str(args.output.resolve()),
            "split": args.split,
            "configuration": {
                "expected_scale": args.expected_scale,
                "scale_tolerance": args.scale_tolerance,
                "fps_tolerance": args.fps_tolerance,
                "duration_tolerance": args.duration_tolerance,
                "offset_radius": args.offset_radius,
                "sample_frames": args.sample_frames,
                "timestamp_scan_limit": args.timestamp_scan_limit,
                "metric_thumbnail_max_side": args.metric_thumbnail_max_side,
            },
        }
    )
    write_results(results, summary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
