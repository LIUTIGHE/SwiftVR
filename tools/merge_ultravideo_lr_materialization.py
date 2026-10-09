#!/usr/bin/env python3
"""Merge sharded UltraVideo LR materialization manifests without copying bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


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
        "source_plan",
        "degradation_version",
        "seed",
        "shard_count",
        "input_view_count",
        "input_clip_count",
        "stored_tensors",
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
    if len(summaries) != shard_count or indices != list(range(shard_count)):
        raise ValueError(
            f"Expected shard indices 0..{shard_count - 1}, got {indices}"
        )

    rows: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    seen_bundle_keys: set[tuple[str, str]] = set()
    for shard_dir in shard_dirs:
        for row in _read_jsonl(shard_dir / "materialized_views.jsonl"):
            materialized_index = int(row["materialized_index"])
            if materialized_index in seen_indices:
                raise ValueError(f"Duplicate materialized_index {materialized_index}")
            bundle_key = (str(row["bundle_path"]), str(row["bundle_key"]))
            if bundle_key in seen_bundle_keys:
                raise ValueError(f"Duplicate bundle tensor reference {bundle_key}")
            bundle_path = Path(str(row["bundle_path"])).expanduser().resolve()
            if not bundle_path.is_file():
                raise FileNotFoundError(bundle_path)
            seen_indices.add(materialized_index)
            seen_bundle_keys.add(bundle_key)
            rows.append(row)

    rows.sort(key=lambda row: int(row["materialized_index"]))
    expected_views = int(reference["input_view_count"])
    expected_indices = list(range(expected_views))
    actual_indices = [int(row["materialized_index"]) for row in rows]
    if actual_indices != expected_indices:
        missing = sorted(set(expected_indices) - set(actual_indices))
        extra = sorted(set(actual_indices) - set(expected_indices))
        raise ValueError(
            "Merged materialization does not cover the clean plan exactly: "
            f"rows={len(rows)} expected={expected_views} "
            f"missing={missing[:12]} extra={extra[:12]}"
        )

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    merged_manifest = output / "materialized_views.jsonl"
    _write_jsonl(merged_manifest, rows)

    total_bytes = sum(int(summary.get("bundle_bytes", 0)) for summary in summaries)
    total_clips = sum(int(summary.get("materialized_clip_count", 0)) for summary in summaries)
    total_views = sum(int(summary.get("materialized_view_count", 0)) for summary in summaries)
    if total_views != expected_views:
        raise ValueError(
            f"Shard summary view total {total_views} != expected {expected_views}"
        )

    summary = {
        "kind": "ultravideo_lr_materialization_merged_v1",
        "source_plan": reference["source_plan"],
        "degradation_version": reference["degradation_version"],
        "seed": int(reference["seed"]),
        "shard_count": shard_count,
        "input_clip_count": int(reference["input_clip_count"]),
        "input_view_count": expected_views,
        "materialized_clip_count": total_clips,
        "materialized_view_count": total_views,
        "bundle_count": total_clips,
        "bundle_bytes": total_bytes,
        "bundle_gib": float(total_bytes / (1024**3)),
        "materialized_manifest": str(merged_manifest),
        "shard_dirs": [str(path) for path in shard_dirs],
        "stored_tensors": reference["stored_tensors"],
        "degradation_definition": reference.get("degradation_definition"),
        "notes": [
            "Bundles remain in their shard directories; the merged manifest references them by absolute path.",
            "materialized_index was verified to cover the clean plan contiguously with no duplicates.",
        ],
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
