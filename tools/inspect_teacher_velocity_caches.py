#!/usr/bin/env python3
"""Inspect independent train/validation teacher caches before DDP training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swiftvr.training.distillation import TeacherVelocityCache
from swiftvr.training.distillation_generalization import (
    cache_overlap_report,
    cache_selected_indices,
    selected_indices_sha256,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument(
        "--path-root",
        type=Path,
        default=Path("."),
        help=(
            "Root used to resolve relative media paths in legacy cache manifests. "
            "New caches embed source_uid and do not require reconstruction."
        ),
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow verified overlap of the same resolved HR source sequence.",
    )
    return parser


def _cache_summary(cache: TeacherVelocityCache) -> dict[str, object]:
    indices = cache_selected_indices(cache.metadata)
    missing_files: list[str] = []
    for item in cache.metadata["samples"]:
        path = cache.root / str(item["file"])
        if not path.is_file():
            missing_files.append(str(path))
    return {
        "root": str(cache.root),
        "sample_count": len(indices),
        "selected_indices_sha256": selected_indices_sha256(indices),
        "selection_mode": cache.metadata.get("selection_mode", "legacy_prefix"),
        "selection_seed": cache.metadata.get("selection_seed"),
        "base_record_count": cache.metadata.get("base_record_count"),
        "full_dataset_length": cache.metadata.get("full_dataset_length"),
        "split": cache.metadata.get("split"),
        "views_per_record": cache.metadata.get("views_per_record"),
        "view_seed": cache.metadata.get("view_seed"),
        "missing_files": missing_files,
    }


def main() -> int:
    args = build_parser().parse_args()
    train = TeacherVelocityCache(args.train_cache)
    val = TeacherVelocityCache(args.val_cache)
    path_root = args.path_root.expanduser().resolve()
    relationship = cache_overlap_report(
        train.metadata,
        val.metadata,
        train_path_root=path_root,
        val_path_root=path_root,
    )
    report = {
        "train": _cache_summary(train),
        "validation": _cache_summary(val),
        "relationship": relationship,
    }
    print(json.dumps(report, indent=2))

    missing = report["train"]["missing_files"] or report["validation"]["missing_files"]
    if missing:
        raise FileNotFoundError("At least one cache tensor file is missing")

    name_collisions = int(relationship["record_uid_collisions"])
    source_overlap = int(relationship["source_overlap_records"])
    if name_collisions:
        print(
            "WARNING: train/validation caches contain "
            f"{name_collisions} record_uid string collision(s), but these names "
            "are diagnostic only. Leakage is determined from resolved HR sources.",
            flush=True,
        )
    if source_overlap and not args.allow_overlap:
        raise RuntimeError(
            "Train/validation caches overlap on "
            f"{source_overlap} resolved HR source sequence(s); use genuinely "
            "independent manifests or pass --allow-overlap deliberately"
        )

    for key in (
        "reference_checkpoint_match",
        "prompt_embedding_sha256_match",
        "reae_sha256_match",
        "timestep_match",
    ):
        if not bool(relationship[key]):
            raise RuntimeError(f"Train/validation cache mismatch: {key}=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
