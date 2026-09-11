#!/usr/bin/env python3
"""Audit immutable M8-C teacher/cache/component lineage before a GPU run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGE_A_KIND = "swiftvr_b2a_stage_a_teacher_velocity"
M8_DECODER_CHANNELS = [128, 96, 64, 64]
CACHE_LINEAGE_FIELDS = (
    "teacher_role",
    "teacher_delta_step",
    "teacher_delta_metadata_sha256",
    "teacher_delta_weights_sha256",
    "reae_sha256",
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument("--transformer-init", type=Path, required=True)
    p.add_argument("--decoder-init", type=Path, required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    train_root = args.train_cache.expanduser().resolve()
    val_root = args.val_cache.expanduser().resolve()
    transformer_root = args.transformer_init.expanduser().resolve()
    decoder_root = args.decoder_init.expanduser().resolve()

    train = _read_json(train_root / "metadata.json")
    val = _read_json(val_root / "metadata.json")
    for name, meta in (("train", train), ("val", val)):
        if meta.get("kind") != STAGE_A_KIND:
            raise ValueError(f"{name} cache kind={meta.get('kind')!r}, expected {STAGE_A_KIND!r}")

    mismatches = {
        field: {"train": train.get(field), "val": val.get(field)}
        for field in CACHE_LINEAGE_FIELDS
        if train.get(field) != val.get(field)
    }
    if mismatches:
        raise ValueError(
            "M8-C train/val Stage-A cache lineage mismatch:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )

    transformer_config = transformer_root / "transformer" / "config.json"
    if not transformer_config.is_file():
        raise FileNotFoundError(
            f"M8 transformer checkpoint is missing transformer/config.json: {transformer_root}"
        )

    decoder_config_path = decoder_root / "config.json"
    decoder = _read_json(decoder_config_path)
    channels = decoder.get("channels")
    if channels != M8_DECODER_CHANNELS:
        raise ValueError(
            f"Decoder init channels={channels!r}, expected {M8_DECODER_CHANNELS!r}"
        )

    report = {
        "status": "PASS",
        "stage_a_lineage": {field: train.get(field) for field in CACHE_LINEAGE_FIELDS},
        "train_cache": str(train_root),
        "train_samples": train.get("sample_count"),
        "train_split": train.get("split"),
        "val_cache": str(val_root),
        "val_samples": val.get("sample_count"),
        "val_split": val.get("split"),
        "transformer_init": str(transformer_root),
        "decoder_init": str(decoder_root),
        "decoder_channels": channels,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
