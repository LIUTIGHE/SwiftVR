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
    parser.add_argument("--allow-overlap", action="store_true")
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
    report = {
        "train": _cache_summary(train),
        "validation": _cache_summary(val),
        "relationship": cache_overlap_report(train.metadata, val.metadata),
    }
    print(json.dumps(report, indent=2))
    missing = report["train"]["missing_files"] or report["validation"]["missing_files"]
    if missing:
        raise FileNotFoundError("At least one cache tensor file is missing")
    overlap = int(report["relationship"]["overlap_records"])
    if overlap and not args.allow_overlap:
        raise RuntimeError(
            f"Train/validation caches overlap on {overlap} record_uid values; "
            "use genuinely independent manifests or pass --allow-overlap deliberately"
        )
    for key in (
        "reference_checkpoint_match",
        "prompt_embedding_sha256_match",
        "reae_sha256_match",
        "timestep_match",
    ):
        if not bool(report["relationship"][key]):
            raise RuntimeError(f"Train/validation cache mismatch: {key}=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
