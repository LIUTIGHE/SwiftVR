#!/usr/bin/env python3
"""Select a source-balanced UltraVideo pilot from indexed source manifests.

The selector preserves the source-level train/val split created by
index_ultravideo_sources.py. For each train source group it chooses up to N clips
that are spread across source time when start_time is available; for validation
it chooses up to M clips per source group. No media is copied or transcoded.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


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


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _temporal_key(row: Mapping[str, object]) -> tuple[int, float, str]:
    start = _float_or_none(row.get("start_time"))
    if start is None:
        start = _float_or_none(row.get("start_frame"))
    return (
        0 if start is not None else 1,
        0.0 if start is None else float(start),
        str(row.get("clip_id", "")),
    )


def _spread_positions(count: int, requested: int) -> list[int]:
    if count <= 0 or requested <= 0:
        return []
    actual = min(count, requested)
    if actual == 1:
        return [count // 2]
    return sorted(
        {
            int(round((slot + 1) * (count - 1) / (actual + 1)))
            for slot in range(actual)
        }
    )


def _select_group(rows: Sequence[Mapping[str, object]], count: int) -> list[dict[str, object]]:
    ordered = sorted(rows, key=_temporal_key)
    return [dict(ordered[position]) for position in _spread_positions(len(ordered), count)]


def _eligible(
    row: Mapping[str, object],
    *,
    min_fps: float,
    max_fps: float,
    min_width: int,
) -> bool:
    fps = _float_or_none(row.get("fps"))
    try:
        width = int(row.get("frame_width"))
    except (TypeError, ValueError):
        return False
    return (
        fps is not None
        and float(min_fps) <= fps <= float(max_fps)
        and width >= int(min_width)
    )


def _select(
    rows: Sequence[Mapping[str, object]],
    *,
    clips_per_source: int,
    min_fps: float,
    max_fps: float,
    min_width: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    rejected = Counter()
    for raw in rows:
        row = dict(raw)
        uid = str(row.get("source_group_uid", ""))
        if not uid:
            rejected["missing_source_group_uid"] += 1
            continue
        if not _eligible(row, min_fps=min_fps, max_fps=max_fps, min_width=min_width):
            rejected["eligibility_filter"] += 1
            continue
        grouped[uid].append(row)

    selected: list[dict[str, object]] = []
    selected_per_group = Counter()
    underfilled = 0
    for uid in sorted(grouped):
        chosen = _select_group(grouped[uid], clips_per_source)
        if len(chosen) < clips_per_source:
            underfilled += 1
        for row in chosen:
            row["pilot_selection"] = {
                "clips_per_source_requested": int(clips_per_source),
                "selection_method": "time_spread_quantiles_v1",
            }
            selected.append(row)
            selected_per_group[uid] += 1

    return selected, {
        "input_clip_count": len(rows),
        "eligible_clip_count": sum(len(values) for values in grouped.values()),
        "eligible_source_group_count": len(grouped),
        "selected_clip_count": len(selected),
        "selected_source_group_count": len(selected_per_group),
        "underfilled_source_group_count": underfilled,
        "rejected_counts": dict(sorted(rejected.items())),
        "selected_clips_per_source_histogram": dict(
            sorted(Counter(selected_per_group.values()).items())
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-clips-per-source", type=int, default=2)
    parser.add_argument("--val-clips-per-source", type=int, default=1)
    parser.add_argument("--min-fps", type=float, default=20.0)
    parser.add_argument("--max-fps", type=float, default=60.1)
    parser.add_argument("--min-width", type=int, default=3840)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.train_clips_per_source <= 0 or args.val_clips_per_source <= 0:
        raise ValueError("clips-per-source values must be positive")
    if args.min_fps <= 0 or args.max_fps < args.min_fps or args.min_width <= 0:
        raise ValueError("Invalid fps/width eligibility limits")

    train_rows = _read_jsonl(args.train_index.expanduser().resolve())
    val_rows = _read_jsonl(args.val_index.expanduser().resolve())
    train_sources = {str(row.get("source_group_uid", "")) for row in train_rows}
    val_sources = {str(row.get("source_group_uid", "")) for row in val_rows}
    overlap = (train_sources - {""}) & (val_sources - {""})
    if overlap:
        raise RuntimeError(f"Input train/val source groups overlap: {len(overlap)}")

    train_selected, train_summary = _select(
        train_rows,
        clips_per_source=int(args.train_clips_per_source),
        min_fps=float(args.min_fps),
        max_fps=float(args.max_fps),
        min_width=int(args.min_width),
    )
    val_selected, val_summary = _select(
        val_rows,
        clips_per_source=int(args.val_clips_per_source),
        min_fps=float(args.min_fps),
        max_fps=float(args.max_fps),
        min_width=int(args.min_width),
    )

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "ultravideo_pilot_train.jsonl", train_selected)
    _write_jsonl(output / "ultravideo_pilot_val.jsonl", val_selected)

    selected_train_sources = {
        str(row["source_group_uid"]) for row in train_selected
    }
    selected_val_sources = {
        str(row["source_group_uid"]) for row in val_selected
    }
    summary = {
        "kind": "ultravideo_source_balanced_pilot_v1",
        "train_index": str(args.train_index.expanduser().resolve()),
        "val_index": str(args.val_index.expanduser().resolve()),
        "selection": {
            "train_clips_per_source": int(args.train_clips_per_source),
            "val_clips_per_source": int(args.val_clips_per_source),
            "method": "time_spread_quantiles_v1",
            "min_fps": float(args.min_fps),
            "max_fps": float(args.max_fps),
            "min_width": int(args.min_width),
        },
        "train": train_summary,
        "validation": val_summary,
        "selected_train_val_source_overlap": len(
            selected_train_sources & selected_val_sources
        ),
        "notes": [
            "This is a selection manifest only; no raw videos are copied or transcoded.",
            "50/60 fps clips remain eligible. Cadence normalization belongs in preprocessing, not source selection.",
            "The pilot maximizes source coverage before adding more clips from the same upstream URL.",
        ],
    }
    if summary["selected_train_val_source_overlap"]:
        raise RuntimeError("Selected pilot has train/val source leakage")
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
