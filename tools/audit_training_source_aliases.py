#!/usr/bin/env python3
"""Sample-check whether same sample_id records across variants share HR content.

This is a read-only lineage diagnostic. It first compares manifest structure for
all repeated sample IDs, then decodes first/middle/last HR frames for a
deterministic subset and reports exact RGB equality plus a small-thumbnail MAE.
It does not rewrite source_uid or training manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from swiftvr.data import TripletSequenceRecord, read_triplet_manifests


def _stable_order(sample_ids: Sequence[str], seed: int) -> list[str]:
    return sorted(
        sample_ids,
        key=lambda value: hashlib.sha256(f"{int(seed)}:{value}".encode("utf-8")).hexdigest(),
    )


def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _thumbnail(array: np.ndarray, max_side: int) -> np.ndarray:
    height, width = array.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = Image.fromarray(array, mode="RGB")
    if target != (width, height):
        image = image.resize(target, resample=Image.Resampling.BOX)
    return np.asarray(image, dtype=np.uint8)


def _positions(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if count <= 1:
        return [frame_count // 2]
    actual = min(frame_count, count)
    return sorted({int(round(value)) for value in np.linspace(0, frame_count - 1, actual)})


def _record_uid(record: TripletSequenceRecord) -> str:
    return f"{record.variant}:{record.sample_id}"


def _compare_frame(left_path: str, right_path: str, max_side: int) -> dict[str, object]:
    left = _load_rgb(left_path)
    right = _load_rgb(right_path)
    result: dict[str, object] = {
        "left_path": left_path,
        "right_path": right_path,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "shape_match": left.shape == right.shape,
    }
    if left.shape != right.shape:
        result.update(
            exact_rgb_equal=False,
            rgb_mae=None,
            thumbnail_mae=None,
        )
        return result
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    result["exact_rgb_equal"] = bool(np.max(delta) == 0)
    result["rgb_mae"] = float(np.mean(delta) / 255.0)
    left_small = _thumbnail(left, max_side)
    right_small = _thumbnail(right, max_side)
    small_delta = np.abs(left_small.astype(np.int16) - right_small.astype(np.int16))
    result["thumbnail_mae"] = float(np.mean(small_delta) / 255.0)
    return result


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frames-per-pair", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--thumbnail-max-side", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-paths", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_count <= 0 or args.frames_per_pair <= 0 or args.thumbnail_max_side <= 0:
        raise ValueError("sample-count, frames-per-pair and thumbnail-max-side must be positive")

    records = read_triplet_manifests(
        args.manifest,
        split=args.split,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
    )
    grouped: dict[str, list[TripletSequenceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sample_id].append(record)

    repeated = {sample_id: values for sample_id, values in grouped.items() if len(values) > 1}
    cross_variant = {
        sample_id: values
        for sample_id, values in repeated.items()
        if len({record.variant for record in values}) > 1
    }
    same_frame_indices = sum(
        len({record.frame_indices for record in values}) == 1
        for values in cross_variant.values()
    )
    same_frame_count = sum(
        len({record.frame_count for record in values}) == 1
        for values in cross_variant.values()
    )

    selected_ids = _stable_order(list(cross_variant), args.seed)[: args.sample_count]
    comparisons: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    exact_frame_matches = 0
    compared_frames = 0
    rgb_maes: list[float] = []
    thumbnail_maes: list[float] = []

    control_rgb_maes: list[float] = []
    control_thumbnail_maes: list[float] = []

    for sample_id in selected_ids:
        variants = sorted(cross_variant[sample_id], key=lambda record: (_record_uid(record)))
        reference = variants[0]
        for other in variants[1:]:
            common_count = min(reference.frame_count, other.frame_count)
            frame_rows: list[dict[str, object]] = []
            for position in _positions(common_count, args.frames_per_pair):
                frame = _compare_frame(
                    reference.hr_paths[position],
                    other.hr_paths[position],
                    args.thumbnail_max_side,
                )
                frame["position"] = int(position)
                frame["left_frame_index"] = int(reference.frame_indices[position])
                frame["right_frame_index"] = int(other.frame_indices[position])
                frame_rows.append(frame)
                compared_frames += 1
                if bool(frame["exact_rgb_equal"]):
                    exact_frame_matches += 1
                if isinstance(frame["rgb_mae"], float):
                    rgb_maes.append(float(frame["rgb_mae"]))
                if isinstance(frame["thumbnail_mae"], float):
                    thumbnail_maes.append(float(frame["thumbnail_mae"]))
            comparisons.append(
                {
                    "sample_id": sample_id,
                    "left_record_uid": _record_uid(reference),
                    "right_record_uid": _record_uid(other),
                    "same_frame_indices": reference.frame_indices == other.frame_indices,
                    "frames": frame_rows,
                }
            )
        print(f"checked alias {sample_id}", flush=True)

    if len(selected_ids) > 1:
        for position, sample_id in enumerate(selected_ids):
            other_id = selected_ids[(position + 1) % len(selected_ids)]
            left_variants = sorted(cross_variant[sample_id], key=lambda record: _record_uid(record))
            right_variants = sorted(cross_variant[other_id], key=lambda record: _record_uid(record))
            left = left_variants[0]
            right = right_variants[-1]
            common_count = min(left.frame_count, right.frame_count)
            frame_rows: list[dict[str, object]] = []
            for frame_position in _positions(common_count, args.frames_per_pair):
                frame = _compare_frame(
                    left.hr_paths[frame_position],
                    right.hr_paths[frame_position],
                    args.thumbnail_max_side,
                )
                frame["position"] = int(frame_position)
                frame_rows.append(frame)
                if isinstance(frame["rgb_mae"], float):
                    control_rgb_maes.append(float(frame["rgb_mae"]))
                if isinstance(frame["thumbnail_mae"], float):
                    control_thumbnail_maes.append(float(frame["thumbnail_mae"]))
            controls.append(
                {
                    "left_sample_id": sample_id,
                    "right_sample_id": other_id,
                    "left_record_uid": _record_uid(left),
                    "right_record_uid": _record_uid(right),
                    "frames": frame_rows,
                }
            )

    report = {
        "kind": "swiftvr_training_source_alias_sample_audit_v1",
        "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
        "split": args.split,
        "record_count": len(records),
        "unique_sample_id_count": len(grouped),
        "repeated_sample_id_count": len(repeated),
        "cross_variant_sample_id_count": len(cross_variant),
        "cross_variant_same_frame_count": same_frame_count,
        "cross_variant_same_frame_indices": same_frame_indices,
        "sampled_cross_variant_ids": len(selected_ids),
        "frames_per_pair": int(args.frames_per_pair),
        "compared_frame_pairs": compared_frames,
        "exact_rgb_equal_frame_pairs": exact_frame_matches,
        "exact_rgb_equal_fraction": (
            float(exact_frame_matches / compared_frames) if compared_frames else None
        ),
        "rgb_mae_mean": float(np.mean(rgb_maes)) if rgb_maes else None,
        "rgb_mae_max": float(np.max(rgb_maes)) if rgb_maes else None,
        "thumbnail_mae_mean": float(np.mean(thumbnail_maes)) if thumbnail_maes else None,
        "thumbnail_mae_max": float(np.max(thumbnail_maes)) if thumbnail_maes else None,
        "control_rgb_mae_mean": float(np.mean(control_rgb_maes)) if control_rgb_maes else None,
        "control_rgb_mae_min": float(np.min(control_rgb_maes)) if control_rgb_maes else None,
        "control_thumbnail_mae_mean": (
            float(np.mean(control_thumbnail_maes)) if control_thumbnail_maes else None
        ),
        "control_thumbnail_mae_min": (
            float(np.min(control_thumbnail_maes)) if control_thumbnail_maes else None
        ),
        "matched_to_control_thumbnail_mae_ratio": (
            float(np.mean(thumbnail_maes) / np.mean(control_thumbnail_maes))
            if thumbnail_maes and control_thumbnail_maes and np.mean(control_thumbnail_maes) > 0
            else None
        ),
        "seed": int(args.seed),
        "comparisons": comparisons,
        "controls": controls,
        "interpretation": (
            "Exact equality on sampled decoded HR frames is strong evidence that same sample_id "
            "plain/text records duplicate the same clean source despite living at different paths. "
            "Non-zero MAE does not by itself prove different source videos; inspect the comparison "
            "rows before changing source identity."
        ),
    }
    _write_json(args.output.expanduser().resolve(), report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"comparisons", "controls"}},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
