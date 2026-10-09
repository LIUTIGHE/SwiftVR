#!/usr/bin/env python3
"""Remove clips requiring global spike-guard fallback from a merged UltraVideo plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Expected JSON object in {path}")
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    tmp.replace(path)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views-per-clip", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.views_per_clip <= 0:
        raise ValueError("views-per-clip must be positive")
    plan = args.plan.expanduser().resolve()
    rows = _read_jsonl(plan)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["clip_id"])].append(row)

    bad_clips = {
        clip_id
        for clip_id, values in grouped.items()
        if any(bool(row.get("global_spike_guard_fallback", False)) for row in values)
    }
    clean: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for clip_id, values in grouped.items():
        values.sort(key=lambda row: int(row["view_index"]))
        if len(values) != int(args.views_per_clip):
            raise ValueError(
                f"{clip_id}: expected {args.views_per_clip} views, got {len(values)}"
            )
        if clip_id in bad_clips:
            excluded.extend(values)
        else:
            clean.extend(values)

    clean.sort(key=lambda row: (int(row["pilot_row_index"]), int(row["view_index"])))
    excluded.sort(key=lambda row: (int(row["pilot_row_index"]), int(row["view_index"])))
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "ultravideo_view_plan_clean.jsonl", clean)
    _write_jsonl(output / "excluded_fallback_views.jsonl", excluded)

    categories = Counter(str(row["selection_category"]) for row in clean)
    summary = {
        "kind": "ultravideo_view_plan_clean_v1",
        "source_plan": str(plan),
        "input_clip_count": len(grouped),
        "input_view_count": len(rows),
        "excluded_clip_count": len(bad_clips),
        "excluded_view_count": len(excluded),
        "excluded_clip_ids": sorted(bad_clips),
        "clean_clip_count": len(grouped) - len(bad_clips),
        "clean_view_count": len(clean),
        "views_per_clip": int(args.views_per_clip),
        "clean_category_counts": dict(sorted(categories.items())),
        "policy": "drop_entire_clip_if_any_selected_view_requires_global_spike_guard_fallback",
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
