#!/usr/bin/env python3
"""Plan deterministic SwiftVR training views per UltraVideo pilot clip.

The planner is read-only with respect to raw media. It decodes a deterministic
candidate pool from each selected UltraVideo clip, builds canonical clean HR/HQ
frames in memory, scores:

- clean 3x SR detail loss: HR high-frequency detail lost by HQ downsampling;
- structural motion: adjacent HQ change after removing per-frame mean luma;
- temporal spike ratio: rejects obvious cuts from the motion-focused pool.

It selects configurable detail / detail+motion / random quotas with a soft
spatio-temporal overlap penalty. The output is a lightweight
JSONL view plan; no HR/HQ/LR training pixels are materialized here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object in {path}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    tmp.replace(path)


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(int(base_seed)), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _decord_reader(path: str):
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("decord is required for UltraVideo view planning") from exc
    return decord.VideoReader(path, ctx=decord.cpu(0))


def _batch_to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return np.asarray(value)


def _cadence_stride(fps: float) -> int:
    """Keep effective cadence near the 24-30 fps regime without interpolation."""
    return 2 if float(fps) >= 45.0 else 1


def _canonical_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or target_width <= 0:
        raise ValueError("Invalid source/canonical dimensions")
    if width > target_width:
        scale = float(target_width) / float(width)
        width = int(target_width)
        height = max(1, round(height * scale))
    width -= width % 3
    height -= height % 3
    if width <= 0 or height <= 0:
        raise ValueError("Canonical dimensions collapsed after divisibility crop")
    return width, height


def _resize_rgb(array: np.ndarray, size: tuple[int, int], resample) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    if image.size != size:
        image = image.resize(size, resample=resample)
    return np.asarray(image, dtype=np.uint8)


def _gray01(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    return (
        0.299 * value[..., 0] + 0.587 * value[..., 1] + 0.114 * value[..., 2]
    ) / 255.0


def _highpass_l1(gray: np.ndarray) -> float:
    image = Image.fromarray(
        np.clip(np.asarray(gray) * 255.0, 0, 255).round().astype(np.uint8),
        mode="L",
    )
    low = np.asarray(image.filter(ImageFilter.BoxBlur(radius=1)), dtype=np.float32) / 255.0
    return float(np.mean(np.abs(np.asarray(gray, dtype=np.float32) - low)))


def _box_iou(a: Sequence[int], b: Sequence[int]) -> float:
    top_a, left_a, height_a, width_a = (int(value) for value in a)
    top_b, left_b, height_b, width_b = (int(value) for value in b)
    bottom_a, right_a = top_a + height_a, left_a + width_a
    bottom_b, right_b = top_b + height_b, left_b + width_b
    intersection = max(0, min(bottom_a, bottom_b) - max(top_a, top_b)) * max(
        0, min(right_a, right_b) - max(left_a, left_b)
    )
    union = height_a * width_a + height_b * width_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _temporal_iou(start_a: int, span_a: int, start_b: int, span_b: int) -> float:
    end_a = start_a + span_a
    end_b = start_b + span_b
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return float(intersection / union) if union > 0 else 0.0


def _candidate_specs(
    *,
    frame_count: int,
    fps: float,
    canonical_hq_size: tuple[int, int],
    clip_length: int,
    crop_size: int,
    candidate_count: int,
    seed: int,
    spatial_candidates_per_time: int = 3,
) -> list[dict[str, object]]:
    stride = _cadence_stride(fps)
    raw_span = (clip_length - 1) * stride + 1
    if frame_count < raw_span:
        return []
    hq_width, hq_height = canonical_hq_size
    if crop_size > hq_width or crop_size > hq_height:
        return []
    max_start = frame_count - raw_span
    max_top = hq_height - crop_size
    max_left = hq_width - crop_size
    spatial_per_time = int(spatial_candidates_per_time)
    if spatial_per_time <= 0:
        raise ValueError("spatial_candidates_per_time must be positive")
    rng = np.random.default_rng(int(seed))
    temporal_count = int(math.ceil(int(candidate_count) / spatial_per_time))
    possible_starts = max_start + 1
    if possible_starts >= temporal_count:
        temporal_starts = [
            int(value)
            for value in rng.choice(
                possible_starts,
                size=temporal_count,
                replace=False,
            ).tolist()
        ]
    else:
        temporal_starts = [
            int(rng.integers(0, max_start + 1))
            for _ in range(temporal_count)
        ]

    specs: list[dict[str, object]] = []
    for candidate_index in range(int(candidate_count)):
        temporal_index = candidate_index // spatial_per_time
        start = temporal_starts[temporal_index]
        top = int(rng.integers(0, max_top + 1))
        left = int(rng.integers(0, max_left + 1))
        horizontal_flip = bool(rng.integers(0, 2))
        frame_positions = [start + offset * stride for offset in range(clip_length)]
        specs.append(
            {
                "candidate_index": candidate_index,
                "temporal_candidate_index": temporal_index,
                "raw_frame_start": start,
                "raw_frame_stride": stride,
                "raw_frame_positions": frame_positions,
                "crop_box_hq": [top, left, crop_size, crop_size],
                "horizontal_flip": horizontal_flip,
            }
        )
    return specs


def _detail_score(
    raw_middle: np.ndarray,
    spec: Mapping[str, object],
    *,
    canonical_hr_size: tuple[int, int],
    crop_size: int,
    scale: int,
) -> dict[str, float]:
    top, left, _, _ = (int(value) for value in spec["crop_box_hq"])
    hr_full = _resize_rgb(raw_middle, canonical_hr_size, Image.Resampling.LANCZOS)
    hq_size = (canonical_hr_size[0] // scale, canonical_hr_size[1] // scale)
    hq_full = _resize_rgb(hr_full, hq_size, Image.Resampling.BOX)
    hq_crop = hq_full[top : top + crop_size, left : left + crop_size]
    hr_crop = hr_full[
        top * scale : (top + crop_size) * scale,
        left * scale : (left + crop_size) * scale,
    ]
    hq_up = _resize_rgb(
        hq_crop,
        (crop_size * scale, crop_size * scale),
        Image.Resampling.BICUBIC,
    )
    hr_detail = _highpass_l1(_gray01(hr_crop))
    hq_up_detail = _highpass_l1(_gray01(hq_up))
    return {
        "hr_highpass_l1": hr_detail,
        "clean_sr_highpass_gap": hr_detail - hq_up_detail,
    }


def _motion_score(
    hq_frames: Mapping[int, np.ndarray],
    spec: Mapping[str, object],
    *,
    score_offsets: Sequence[int],
    crop_size: int,
) -> dict[str, float]:
    all_positions = [int(value) for value in spec["raw_frame_positions"]]
    positions = [all_positions[int(offset)] for offset in score_offsets]
    top, left, _, _ = (int(value) for value in spec["crop_box_hq"])
    gray_clips: list[np.ndarray] = []
    for position in positions:
        hq = hq_frames[position]
        crop = hq[top : top + crop_size, left : left + crop_size]
        gray_clips.append(_gray01(crop))
    gray = np.stack(gray_clips, axis=0)
    frame_means = np.mean(gray, axis=(1, 2), keepdims=True)
    centered = gray - frame_means
    structural = np.mean(np.abs(centered[1:] - centered[:-1]), axis=(1, 2))
    raw_motion = np.mean(np.abs(gray[1:] - gray[:-1]), axis=(1, 2))
    luma = np.abs(frame_means[1:] - frame_means[:-1]).reshape(-1)
    median_structural = float(np.median(structural))
    spike_ratio = float(np.max(structural) / max(median_structural, 1e-8))
    return {
        "structural_motion_l1": float(np.mean(structural)),
        "raw_motion_l1": float(np.mean(raw_motion)),
        "luma_motion_l1": float(np.mean(luma)),
        "temporal_spike_ratio": spike_ratio,
    }


def _decode_hq_proxy_frames(
    reader,
    positions: Sequence[int],
    *,
    hq_size: tuple[int, int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Decode native frames in small batches and retain only HQ-size RGB proxies."""
    result: dict[int, np.ndarray] = {}
    ordered = sorted({int(value) for value in positions})
    for start in range(0, len(ordered), int(batch_size)):
        batch_positions = ordered[start : start + int(batch_size)]
        decoded = _batch_to_numpy(reader.get_batch(batch_positions))
        for index, position in enumerate(batch_positions):
            result[position] = _resize_rgb(
                decoded[index],
                hq_size,
                Image.Resampling.BOX,
            )
        del decoded
    return result


def _rank_normalized(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    denominator = max(len(values) - 1, 1)
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = rank / denominator
    return result


def _diversity_penalty(
    candidate: Mapping[str, object],
    selected: Sequence[Mapping[str, object]],
    *,
    raw_span: int,
) -> float:
    if not selected:
        return 0.0
    return max(
        0.65 * _box_iou(candidate["crop_box_hq"], item["crop_box_hq"])
        + 0.35
        * _temporal_iou(
            int(candidate["raw_frame_start"]),
            raw_span,
            int(item["raw_frame_start"]),
            raw_span,
        )
        for item in selected
    )


def _greedy_select(
    pool: Sequence[Mapping[str, object]],
    *,
    count: int,
    base_scores: Sequence[float],
    selected: list[dict[str, object]],
    raw_span: int,
    category: str,
    diversity_weight: float,
) -> None:
    available = {
        int(item["candidate_index"]): (item, float(score))
        for item, score in zip(pool, base_scores)
        if int(item["candidate_index"]) not in {
            int(value["candidate_index"]) for value in selected
        }
    }
    for _ in range(min(int(count), len(available))):
        best_index = max(
            available,
            key=lambda index: (
                available[index][1]
                - diversity_weight
                * _diversity_penalty(
                    available[index][0],
                    selected,
                    raw_span=raw_span,
                ),
                -index,
            ),
        )
        item, score = available.pop(best_index)
        chosen = dict(item)
        chosen["selection_category"] = category
        chosen["selection_base_score"] = score
        selected.append(chosen)


def _select_views(
    candidates: Sequence[Mapping[str, object]],
    *,
    clip_length: int,
    stride: int,
    detail_count: int,
    detail_motion_count: int,
    random_count: int,
    diversity_weight: float,
    spike_limit: float,
    global_spike_limit: float,
    seed: int,
) -> list[dict[str, object]]:
    requested = detail_count + detail_motion_count + random_count
    if len(candidates) < requested:
        raise ValueError("Candidate pool is smaller than requested selected views")
    if global_spike_limit <= 0:
        raise ValueError("global_spike_limit must be positive")

    safe = [
        {**item, "global_spike_guard_fallback": False}
        for item in candidates
        if float(item["temporal_spike_ratio"]) <= float(global_spike_limit)
    ]
    if len(safe) < requested:
        rejected = sorted(
            (
                item
                for item in candidates
                if float(item["temporal_spike_ratio"]) > float(global_spike_limit)
            ),
            key=lambda item: (
                float(item["temporal_spike_ratio"]),
                int(item["candidate_index"]),
            ),
        )
        safe.extend(
            {
                **item,
                "global_spike_guard_fallback": True,
            }
            for item in rejected[: requested - len(safe)]
        )
    candidates = safe

    gaps = [float(item["clean_sr_highpass_gap"]) for item in candidates]
    detail_rank = _rank_normalized(gaps)
    motion_rank = _rank_normalized(
        [float(item["structural_motion_l1"]) for item in candidates]
    )
    detail_motion_scores = [
        0.55 * detail + 0.45 * motion
        if float(item["temporal_spike_ratio"]) <= float(spike_limit)
        else -1.0
        for item, detail, motion in zip(candidates, detail_rank, motion_rank)
    ]
    raw_span = (clip_length - 1) * stride + 1
    selected: list[dict[str, object]] = []
    _greedy_select(
        candidates,
        count=detail_count,
        base_scores=detail_rank,
        selected=selected,
        raw_span=raw_span,
        category="detail",
        diversity_weight=diversity_weight,
    )
    _greedy_select(
        candidates,
        count=detail_motion_count,
        base_scores=detail_motion_scores,
        selected=selected,
        raw_span=raw_span,
        category="detail_motion",
        diversity_weight=diversity_weight,
    )

    remaining = [
        item
        for item in candidates
        if int(item["candidate_index"])
        not in {int(value["candidate_index"]) for value in selected}
    ]
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(remaining)).tolist()
    random_scores = [0.0] * len(remaining)
    ordered = [remaining[index] for index in order]
    _greedy_select(
        ordered,
        count=random_count,
        base_scores=random_scores,
        selected=selected,
        raw_span=raw_span,
        category="random",
        diversity_weight=diversity_weight,
    )
    return selected


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    clean = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if clean.size == 0:
        return {key: None for key in ("min", "p10", "median", "p90", "max", "mean")}
    return {
        "min": float(np.min(clean)),
        "p10": float(np.percentile(clean, 10)),
        "median": float(np.median(clean)),
        "p90": float(np.percentile(clean, 90)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-hr-width", type=int, default=3840)
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument(
        "--spatial-candidates-per-time",
        type=int,
        default=3,
        help=(
            "Reuse each sampled temporal window for this many spatial candidates. "
            "This reduces expensive random video seeks without reducing the total candidate pool."
        ),
    )
    parser.add_argument(
        "--motion-score-frames",
        type=int,
        default=7,
        help="Evenly spaced frames from each 13-frame view used only for motion scoring.",
    )
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=4,
        help="Native-resolution frames decoded at once while building HQ motion proxies.",
    )
    parser.add_argument("--detail-count", type=int, default=4)
    parser.add_argument("--detail-motion-count", type=int, default=2)
    parser.add_argument("--random-count", type=int, default=2)
    parser.add_argument("--diversity-weight", type=float, default=0.35)
    parser.add_argument("--spike-limit", type=float, default=4.0)
    parser.add_argument(
        "--global-spike-limit",
        type=float,
        default=10.0,
        help=(
            "Wide scene-cut guard applied to every selection category. "
            "If too few candidates remain, the lowest-spike rejected candidates "
            "are used only to preserve the requested view count."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split pilot rows into deterministic index-modulo shards for resumable/parallel planning.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index in [0, shard-count).",
    )
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clip_length <= 0 or args.clip_length % 4 != 1:
        raise ValueError("clip-length must be positive and satisfy T=4k+1")
    if args.crop_size <= 0 or args.scale <= 0 or args.candidate_count <= 0:
        raise ValueError("crop-size, scale and candidate-count must be positive")
    if args.spatial_candidates_per_time <= 0:
        raise ValueError("spatial-candidates-per-time must be positive")
    if args.motion_score_frames < 2 or args.motion_score_frames > args.clip_length:
        raise ValueError("motion-score-frames must be in [2, clip-length]")
    if args.decode_batch_size <= 0:
        raise ValueError("decode-batch-size must be positive")
    if args.spike_limit <= 0 or args.global_spike_limit <= 0:
        raise ValueError("spike limits must be positive")
    if args.spike_limit > args.global_spike_limit:
        raise ValueError("spike-limit must not exceed global-spike-limit")
    requested = args.detail_count + args.detail_motion_count + args.random_count
    if requested <= 0 or args.candidate_count < requested:
        raise ValueError("candidate-count must cover all requested selected views")
    if args.max_records < 0:
        raise ValueError("max-records must be non-negative")
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")

    all_rows = _read_jsonl(args.pilot.expanduser().resolve())
    indexed_rows = [
        (index, row)
        for index, row in enumerate(all_rows)
        if index % int(args.shard_count) == int(args.shard_index)
    ]
    if args.max_records:
        indexed_rows = indexed_rows[: int(args.max_records)]
    rows = [row for _, row in indexed_rows]
    source_row_indices = [index for index, _ in indexed_rows]
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    planned: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    selected_metric_values: dict[str, list[float]] = {
        "clean_sr_highpass_gap": [],
        "structural_motion_l1": [],
        "temporal_spike_ratio": [],
    }
    overlap_spatial: list[float] = []
    overlap_temporal: list[float] = []
    global_spike_guard_fallback_count = 0

    for record_index, (source_row_index, row) in enumerate(
        zip(source_row_indices, rows),
        start=1,
    ):
        path = str(row.get("raw_video", ""))
        fps = float(row.get("fps"))
        clip_id = str(row.get("clip_id", ""))
        if not path or not clip_id:
            raise ValueError("Pilot row is missing raw_video or clip_id")
        reader = _decord_reader(path)
        frame_count = len(reader)
        if frame_count <= 0:
            skipped.append({"clip_id": clip_id, "reason": "no_frames"})
            continue
        sample = _batch_to_numpy(reader.get_batch([0]))[0]
        source_height, source_width = sample.shape[:2]
        hr_width, hr_height = _canonical_size(
            int(source_width),
            int(source_height),
            int(args.target_hr_width),
        )
        hq_size = (hr_width // int(args.scale), hr_height // int(args.scale))
        seed = _stable_seed(args.seed, row.get("source_group_uid"), clip_id)
        specs = _candidate_specs(
            frame_count=frame_count,
            fps=fps,
            canonical_hq_size=hq_size,
            clip_length=int(args.clip_length),
            crop_size=int(args.crop_size),
            candidate_count=int(args.candidate_count),
            seed=seed,
            spatial_candidates_per_time=int(args.spatial_candidates_per_time),
        )
        if len(specs) < requested:
            skipped.append(
                {
                    "clip_id": clip_id,
                    "reason": "insufficient_frames_or_geometry",
                    "frame_count": frame_count,
                    "fps": fps,
                    "canonical_hr_size": [hr_width, hr_height],
                }
            )
            continue

        score_offsets = sorted(
            {
                int(round(value))
                for value in np.linspace(
                    0,
                    int(args.clip_length) - 1,
                    int(args.motion_score_frames),
                )
            }
        )
        motion_positions = sorted(
            {
                int(spec["raw_frame_positions"][offset])
                for spec in specs
                for offset in score_offsets
            }
        )
        center_specs: dict[int, list[Mapping[str, object]]] = {}
        for spec in specs:
            middle_position = int(
                spec["raw_frame_positions"][int(args.clip_length) // 2]
            )
            center_specs.setdefault(middle_position, []).append(spec)
        center_positions = sorted(center_specs)

        verbose_clip = record_index <= 3 or record_index % 25 == 0 or record_index == len(rows)
        if verbose_clip:
            print(
                f"[{record_index}/{len(rows)}] start clip={clip_id} "
                f"source={source_width}x{source_height} fps={fps:g} frames={frame_count} "
                f"candidates={len(specs)} temporal_windows={len(center_positions)} "
                f"motion_positions={len(motion_positions)}",
                flush=True,
            )

        stage_started = time.perf_counter()
        hq_proxy_frames = _decode_hq_proxy_frames(
            reader,
            motion_positions,
            hq_size=hq_size,
            batch_size=int(args.decode_batch_size),
        )
        if verbose_clip:
            print(
                f"[{record_index}/{len(rows)}] motion proxies ready "
                f"in {time.perf_counter() - stage_started:.1f}s",
                flush=True,
            )

        detail_by_candidate: dict[int, dict[str, float]] = {}
        stage_started = time.perf_counter()
        for batch_start in range(0, len(center_positions), int(args.decode_batch_size)):
            batch_positions = center_positions[
                batch_start : batch_start + int(args.decode_batch_size)
            ]
            decoded_centers = _batch_to_numpy(reader.get_batch(batch_positions))
            for local_index, middle_position in enumerate(batch_positions):
                raw_middle = decoded_centers[local_index]
                for spec in center_specs[middle_position]:
                    detail_by_candidate[int(spec["candidate_index"])] = _detail_score(
                        raw_middle,
                        spec,
                        canonical_hr_size=(hr_width, hr_height),
                        crop_size=int(args.crop_size),
                        scale=int(args.scale),
                    )
            del decoded_centers
        if verbose_clip:
            print(
                f"[{record_index}/{len(rows)}] detail centers ready "
                f"in {time.perf_counter() - stage_started:.1f}s",
                flush=True,
            )

        candidates: list[dict[str, object]] = []
        for spec in specs:
            motion = _motion_score(
                hq_proxy_frames,
                spec,
                score_offsets=score_offsets,
                crop_size=int(args.crop_size),
            )
            detail = detail_by_candidate[int(spec["candidate_index"])]
            candidates.append({**spec, **detail, **motion})
        del hq_proxy_frames, detail_by_candidate


        stride = _cadence_stride(fps)
        selected = _select_views(
            candidates,
            clip_length=int(args.clip_length),
            stride=stride,
            detail_count=int(args.detail_count),
            detail_motion_count=int(args.detail_motion_count),
            random_count=int(args.random_count),
            diversity_weight=float(args.diversity_weight),
            spike_limit=float(args.spike_limit),
            global_spike_limit=float(args.global_spike_limit),
            seed=_stable_seed(seed, "random_selection"),
        )
        selected.sort(
            key=lambda item: (
                {"detail": 0, "detail_motion": 1, "random": 2}[str(item["selection_category"])],
                int(item["candidate_index"]),
            )
        )
        raw_span = (int(args.clip_length) - 1) * stride + 1
        for left in range(len(selected)):
            for right in range(left + 1, len(selected)):
                overlap_spatial.append(
                    _box_iou(selected[left]["crop_box_hq"], selected[right]["crop_box_hq"])
                )
                overlap_temporal.append(
                    _temporal_iou(
                        int(selected[left]["raw_frame_start"]),
                        raw_span,
                        int(selected[right]["raw_frame_start"]),
                        raw_span,
                    )
                )
        for view_index, item in enumerate(selected):
            if bool(item.get("global_spike_guard_fallback", False)):
                global_spike_guard_fallback_count += 1
            for name in selected_metric_values:
                selected_metric_values[name].append(float(item[name]))
            planned.append(
                {
                    "dataset": "UltraVideo",
                    "subset": "short",
                    "pilot_row_index": int(source_row_index),
                    "split": row.get("split"),
                    "clip_id": clip_id,
                    "source_group_uid": row.get("source_group_uid"),
                    "source_url": row.get("source_url"),
                    "raw_video": path,
                    "fps": fps,
                    "source_frame_count": frame_count,
                    "canonical_hr_width": hr_width,
                    "canonical_hr_height": hr_height,
                    "scale": int(args.scale),
                    "hq_width": hq_size[0],
                    "hq_height": hq_size[1],
                    "clip_length": int(args.clip_length),
                    "view_index": view_index,
                    **item,
                }
            )

        if record_index == 1 or record_index % 25 == 0 or record_index == len(rows):
            print(
                f"planned {record_index}/{len(rows)} clips; "
                f"views={len(planned)} skipped={len(skipped)}",
                flush=True,
            )

    expected_views = (len(rows) - len(skipped)) * requested
    if len(planned) != expected_views:
        raise RuntimeError(
            f"Planned {len(planned)} views, expected {expected_views}"
        )

    _write_jsonl(output / "ultravideo_view_plan.jsonl", planned)
    _write_jsonl(output / "skipped_clips.jsonl", skipped)
    category_counts: dict[str, int] = {}
    for item in planned:
        key = str(item["selection_category"])
        category_counts[key] = category_counts.get(key, 0) + 1

    summary = {
        "kind": "ultravideo_deterministic_view_plan_v1",
        "pilot": str(args.pilot.expanduser().resolve()),
        "pilot_total_clip_count": len(all_rows),
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "input_clip_count": len(rows),
        "planned_clip_count": len(rows) - len(skipped),
        "skipped_clip_count": len(skipped),
        "selected_view_count": len(planned),
        "views_per_clip": requested,
        "selection_counts_per_clip": {
            "detail": int(args.detail_count),
            "detail_motion": int(args.detail_motion_count),
            "random": int(args.random_count),
        },
        "selected_category_counts": dict(sorted(category_counts.items())),
        "candidate_count": int(args.candidate_count),
        "spatial_candidates_per_time": int(args.spatial_candidates_per_time),
        "motion_score_frames": int(args.motion_score_frames),
        "decode_batch_size": int(args.decode_batch_size),
        "clip_length": int(args.clip_length),
        "crop_size_hq_lr": int(args.crop_size),
        "scale": int(args.scale),
        "target_hr_width": int(args.target_hr_width),
        "cadence": "stride=2 for fps>=45, otherwise stride=1",
        "diversity_weight": float(args.diversity_weight),
        "spike_limit": float(args.spike_limit),
        "global_spike_limit": float(args.global_spike_limit),
        "global_spike_guard_fallback_count": int(global_spike_guard_fallback_count),
        "seed": int(args.seed),
        "selected_metric_distributions": {
            name: _quantiles(values)
            for name, values in selected_metric_values.items()
        },
        "pairwise_selected_spatial_iou": _quantiles(overlap_spatial),
        "pairwise_selected_temporal_iou": _quantiles(overlap_temporal),
        "notes": [
            "The view selector uses only clean HR/HQ content; it is independent of synthetic degradation parameters.",
            "detail ranks clean 3x SR high-frequency loss within each clip.",
            "detail_motion combines within-clip detail and structural-motion ranks and excludes candidates above the temporal spike limit.",
            "All categories also use a wide global temporal-spike guard to avoid obvious scene cuts.",
            "random views remain unbiased apart from the same diversity penalty.",
            "Temporal candidate windows are reused across several spatial crops to reduce compressed-video random seeks.",
            "Motion scoring uses evenly spaced HQ-resolution proxy frames to bound native 4K/8K decode memory.",
            "Center native frames are decoded in batches and reused across spatial candidates sharing a temporal window.",
            "Detail scoring decodes only the center native-resolution frame for each candidate.",
            "No training pixels are materialized by this planner.",
        ],
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
