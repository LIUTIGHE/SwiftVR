#!/usr/bin/env python3
"""Index a sharded UltraVideo-Short download without modifying raw data.

The expected layout is compatible with manually downloaded Hugging Face shards:

    ROOT/
      short.csv
      clips_short_1/clips_short/*.mp4
      ...
      clips_short_36/clips_short/*.mp4

The tool joins MP4 filenames to UltraVideo's short.csv by clip_id, assigns a
deterministic train/val split at the upstream source-URL level when metadata is
available, and writes lightweight JSONL source manifests plus an inventory
summary. It does not decode videos, synthesize degradations, or create SwiftVR
HR/HQ/LR triplets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SHARD_RE = re.compile(r"^clips_short_(\d+)$")
DEFAULT_METADATA_NAME = "short.csv"


def _stable_unit_interval(value: str, seed: int) -> float:
    payload = f"{int(seed)}:{value}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _source_group(row: Mapping[str, str], clip_id: str) -> tuple[str, str]:
    url = str(row.get("url", "") or "").strip()
    if url:
        payload = json.dumps(
            {"method": "ultravideo_source_url_sha256_v1", "url": url},
            sort_keys=True,
            separators=(",", ":"),
        )
        return "source_url", hashlib.sha256(payload.encode("utf-8")).hexdigest()
    payload = json.dumps(
        {"method": "clip_id_fallback_sha256_v1", "clip_id": clip_id},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "clip_id_fallback", hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_for_group(group_uid: str, *, seed: int, val_fraction: float) -> str:
    return "val" if _stable_unit_interval(group_uid, seed) < float(val_fraction) else "train"


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _read_metadata(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    if not path.is_file():
        return {}, []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(name) for name in (reader.fieldnames or [])]
        if "clip_id" not in fieldnames:
            raise ValueError(f"{path} does not contain a clip_id column")
        rows: dict[str, dict[str, str]] = {}
        duplicates: list[str] = []
        for row in reader:
            raw_clip_id = str(row.get("clip_id", "") or "").strip()
            if not raw_clip_id:
                continue
            clip_name = Path(raw_clip_id).name
            clip_id = clip_name[:-4] if clip_name.lower().endswith(".mp4") else clip_name
            if clip_id in rows:
                duplicates.append(clip_id)
                continue
            normalized = {str(key): str(value or "") for key, value in row.items()}
            normalized["_metadata_clip_id"] = raw_clip_id
            rows[clip_id] = normalized
    if duplicates:
        raise ValueError(
            f"Metadata contains duplicate clip_id values; examples={sorted(set(duplicates))[:10]}"
        )
    return rows, fieldnames


def _discover_videos(root: Path) -> tuple[list[tuple[int, Path]], list[int]]:
    shard_dirs: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = SHARD_RE.fullmatch(path.name)
        if match:
            shard_dirs.append((int(match.group(1)), path))
    shard_dirs.sort()

    videos: list[tuple[int, Path]] = []
    for shard_index, shard_dir in shard_dirs:
        nested = shard_dir / "clips_short"
        if not nested.is_dir():
            raise FileNotFoundError(f"Expected nested clips_short directory: {nested}")
        for video in sorted(nested.glob("*.mp4")):
            videos.append((shard_index, video.resolve()))
    return videos, [index for index, _ in shard_dirs]


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


def _histogram(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        counter["missing" if value is None else str(value)] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=36)
    parser.add_argument("--split-seed", type=int, default=20260923)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--require-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if short.csv is unavailable or an MP4 cannot be joined to metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0,1)")
    if args.expected_shards <= 0:
        raise ValueError("--expected-shards must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    metadata_path = (
        args.metadata_csv.expanduser().resolve()
        if args.metadata_csv is not None
        else root / DEFAULT_METADATA_NAME
    )
    metadata, metadata_fields = _read_metadata(metadata_path)
    if args.require_metadata and not metadata:
        raise FileNotFoundError(
            f"UltraVideo short metadata was not found/readable: {metadata_path}"
        )

    videos, shard_indices = _discover_videos(root)
    if not videos:
        raise RuntimeError(f"No MP4 files found below {root}/clips_short_*/clips_short")

    expected = set(range(1, int(args.expected_shards) + 1))
    found = set(shard_indices)
    missing_shards = sorted(expected - found)
    extra_shards = sorted(found - expected)

    seen_clip_ids: dict[str, Path] = {}
    duplicate_video_ids: list[str] = []
    rows: list[dict[str, object]] = []
    missing_metadata_ids: list[str] = []
    group_to_clips: dict[str, list[str]] = defaultdict(list)
    group_method_counts: Counter[str] = Counter()

    for shard_index, video_path in videos:
        clip_id = video_path.stem
        if clip_id in seen_clip_ids:
            duplicate_video_ids.append(clip_id)
            continue
        seen_clip_ids[clip_id] = video_path
        meta = metadata.get(clip_id, {})
        if not meta:
            missing_metadata_ids.append(clip_id)
            if args.require_metadata:
                continue
        method, source_group_uid = _source_group(meta, clip_id)
        split = _split_for_group(
            source_group_uid,
            seed=int(args.split_seed),
            val_fraction=float(args.val_fraction),
        )
        group_to_clips[source_group_uid].append(clip_id)
        group_method_counts[method] += 1
        rows.append(
            {
                "dataset": "UltraVideo",
                "subset": "short",
                "clip_id": clip_id,
                "metadata_clip_id": str(meta.get("_metadata_clip_id", "") or "") or None,
                "raw_video": str(video_path),
                "shard": int(shard_index),
                "split": split,
                "source_group_method": method,
                "source_group_uid": source_group_uid,
                "source_url": str(meta.get("url", "") or "") or None,
                "frame_width": _int_or_none(meta.get("frame_width")),
                "frame_height": _int_or_none(meta.get("frame_height")),
                "fps": _float_or_none(meta.get("fps")),
                "start_frame": _int_or_none(meta.get("start_frame")),
                "end_frame": _int_or_none(meta.get("end_frame")),
                "total_frames": _int_or_none(meta.get("total_frames")),
                "start_time": _float_or_none(meta.get("start_time")),
                "end_time": _float_or_none(meta.get("end_time")),
                "duration": _float_or_none(meta.get("duration")),
                "vtss_score": _float_or_none(meta.get("vtss_score")),
                "motion_score": _float_or_none(meta.get("motion_score")),
                "video_clip_score": _float_or_none(meta.get("video_clip_score")),
                "metadata_present": bool(meta),
            }
        )

    if duplicate_video_ids:
        raise ValueError(
            "Duplicate MP4 clip IDs across shards; examples="
            f"{sorted(set(duplicate_video_ids))[:10]}"
        )
    if args.require_metadata and missing_metadata_ids:
        raise ValueError(
            f"{len(missing_metadata_ids)} MP4 clips are missing from {metadata_path}; "
            f"examples={missing_metadata_ids[:10]}"
        )

    rows.sort(key=lambda row: str(row["clip_id"]))
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    train_groups = {str(row["source_group_uid"]) for row in train_rows}
    val_groups = {str(row["source_group_uid"]) for row in val_rows}
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"Source-group split leakage detected: {len(overlap)} groups")

    metadata_only_ids = sorted(set(metadata) - set(seen_clip_ids))
    group_sizes = Counter(len(values) for values in group_to_clips.values())
    summary = {
        "kind": "ultravideo_short_raw_index_v1",
        "raw_root": str(root),
        "metadata_csv": str(metadata_path),
        "metadata_fieldnames": metadata_fields,
        "expected_shards": int(args.expected_shards),
        "found_shards": shard_indices,
        "missing_shards": missing_shards,
        "extra_shards": extra_shards,
        "video_count": len(rows),
        "metadata_row_count": len(metadata),
        "missing_metadata_video_count": len(missing_metadata_ids),
        "missing_metadata_video_examples": missing_metadata_ids[:20],
        "metadata_without_local_video_count": len(metadata_only_ids),
        "metadata_without_local_video_examples": metadata_only_ids[:20],
        "source_group_count": len(group_to_clips),
        "source_group_method_counts": dict(sorted(group_method_counts.items())),
        "clips_per_source_group_histogram": {
            str(key): value for key, value in sorted(group_sizes.items())
        },
        "split_seed": int(args.split_seed),
        "val_fraction_requested": float(args.val_fraction),
        "train_clip_count": len(train_rows),
        "val_clip_count": len(val_rows),
        "train_source_group_count": len(train_groups),
        "val_source_group_count": len(val_groups),
        "train_val_source_group_overlap": len(overlap),
        "resolution_width_histogram": _histogram(rows, "frame_width"),
        "resolution_height_histogram": _histogram(rows, "frame_height"),
        "fps_histogram": _histogram(rows, "fps"),
        "duration_histogram": _histogram(rows, "duration"),
        "notes": [
            "Raw UltraVideo files are not moved or renamed.",
            "Train/val assignment is grouped by upstream source URL when short.csv provides it.",
            "These JSONL files are source manifests only; they are not SwiftVR HR/HQ/LR triplet manifests.",
            "Do not combine UltraVideo source expansion with the D1 view-sampling experiment if clean attribution is required.",
        ],
    }

    _write_jsonl(output / "ultravideo_short_all.jsonl", rows)
    _write_jsonl(output / "ultravideo_short_train.jsonl", train_rows)
    _write_jsonl(output / "ultravideo_short_val.jsonl", val_rows)
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
