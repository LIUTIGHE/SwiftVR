#!/usr/bin/env python3
"""Merge deterministic UltraVideo view-plan shards with strict consistency checks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.is_file():
        return rows
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


def _temporal_iou(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    start_a = int(left["raw_frame_start"])
    start_b = int(right["raw_frame_start"])
    stride_a = int(left["raw_frame_stride"])
    stride_b = int(right["raw_frame_stride"])
    length_a = len(left["raw_frame_positions"])
    length_b = len(right["raw_frame_positions"])
    span_a = (length_a - 1) * stride_a + 1
    span_b = (length_b - 1) * stride_b + 1
    end_a = start_a + span_a
    end_b = start_b + span_b
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return float(intersection / union) if union > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_dirs = [path.expanduser().resolve() for path in args.shard_dir]
    summaries = [_read_json(path / "summary.json") for path in shard_dirs]
    if not summaries:
        raise ValueError("No shard summaries")

    reference = summaries[0]
    invariant_keys = (
        "kind",
        "pilot",
        "pilot_total_clip_count",
        "shard_count",
        "views_per_clip",
        "selection_counts_per_clip",
        "candidate_count",
        "spatial_candidates_per_time",
        "motion_score_frames",
        "decode_batch_size",
        "clip_length",
        "crop_size_hq_lr",
        "scale",
        "target_hr_width",
        "cadence",
        "diversity_weight",
        "spike_limit",
        "global_spike_limit",
        "seed",
    )
    for index, summary in enumerate(summaries[1:], start=1):
        mismatches = {
            key: (reference.get(key), summary.get(key))
            for key in invariant_keys
            if reference.get(key) != summary.get(key)
        }
        if mismatches:
            raise ValueError(f"Shard {index} configuration mismatch: {mismatches}")

    shard_count = int(reference.get("shard_count", 0))
    indices = sorted(int(summary.get("shard_index", -1)) for summary in summaries)
    if shard_count != len(summaries) or indices != list(range(shard_count)):
        raise ValueError(
            f"Expected all shard indices 0..{shard_count - 1}, got {indices}"
        )

    plans: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    seen_keys: set[tuple[int, int]] = set()
    for shard_dir in shard_dirs:
        for row in _read_jsonl(shard_dir / "ultravideo_view_plan.jsonl"):
            key = (int(row["pilot_row_index"]), int(row["view_index"]))
            if key in seen_keys:
                raise ValueError(f"Duplicate planned view key: {key}")
            seen_keys.add(key)
            plans.append(row)
        skipped.extend(_read_jsonl(shard_dir / "skipped_clips.jsonl"))

    plans.sort(key=lambda row: (int(row["pilot_row_index"]), int(row["view_index"])))
    skipped.sort(key=lambda row: str(row.get("clip_id", "")))
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in plans:
        grouped[int(row["pilot_row_index"])].append(row)

    expected_views = int(reference["views_per_clip"])
    bad = {
        index: len(rows)
        for index, rows in grouped.items()
        if len(rows) != expected_views
    }
    if bad:
        raise ValueError(f"Some planned clips do not have {expected_views} views: {bad}")

    categories = Counter(str(row["selection_category"]) for row in plans)
    metric_names = (
        "clean_sr_highpass_gap",
        "structural_motion_l1",
        "temporal_spike_ratio",
    )
    spatial: list[float] = []
    temporal: list[float] = []
    for rows in grouped.values():
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                spatial.append(_box_iou(rows[left]["crop_box_hq"], rows[right]["crop_box_hq"]))
                temporal.append(_temporal_iou(rows[left], rows[right]))

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "ultravideo_view_plan.jsonl", plans)
    _write_jsonl(output / "skipped_clips.jsonl", skipped)

    planned_clip_count = len(grouped)
    summary = {
        **{key: reference.get(key) for key in invariant_keys},
        "kind": "ultravideo_deterministic_view_plan_merged_v1",
        "merged_shard_dirs": [str(path) for path in shard_dirs],
        "input_clip_count": sum(int(item.get("input_clip_count", 0)) for item in summaries),
        "planned_clip_count": planned_clip_count,
        "skipped_clip_count": len(skipped),
        "selected_view_count": len(plans),
        "global_spike_guard_fallback_count": sum(
            int(item.get("global_spike_guard_fallback_count", 0))
            for item in summaries
        ),
        "selected_category_counts": dict(sorted(categories.items())),
        "selected_metric_distributions": {
            name: _quantiles([float(row[name]) for row in plans])
            for name in metric_names
        },
        "pairwise_selected_spatial_iou": _quantiles(spatial),
        "pairwise_selected_temporal_iou": _quantiles(temporal),
        "notes": [
            "Merged from all deterministic planner shards after strict configuration and duplicate checks.",
            "pilot_row_index preserves the source pilot ordering across shards.",
        ],
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
