#!/usr/bin/env python3
"""Smoke-check real pre-materialized UltraVideo LR bundles against distillation interfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from swiftvr.data import UltraVideoMaterializedLRDataset
from swiftvr.training import distillation_sample_identity, prepare_training_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.samples <= 0:
        raise ValueError("batch-size and samples must be positive")

    dataset = UltraVideoMaterializedLRDataset(
        args.manifest.expanduser().resolve(),
        verify_paths=True,
    )
    count = min(len(dataset), int(args.samples))
    loader = DataLoader(
        torch.utils.data.Subset(dataset, range(count)),
        batch_size=min(int(args.batch_size), count),
        shuffle=False,
        num_workers=0,
    )

    observed = 0
    identities: list[dict[str, object]] = []
    for batch in loader:
        prepared = prepare_training_batch(batch, allow_missing_target=True)
        lq_input = prepared["lq_input"]
        if not isinstance(lq_input, torch.Tensor):
            raise TypeError("prepare_training_batch did not return tensor lq_input")
        if prepared["target"] is not None:
            raise ValueError("LR-only smoke unexpectedly returned RGB target")
        if tuple(lq_input.shape[-2:]) != (
            int(batch["target_height"][0].item()),
            int(batch["target_width"][0].item()),
        ):
            raise ValueError("Upscaled LR geometry does not match materialized metadata")
        if not torch.isfinite(lq_input).all():
            raise ValueError("Non-finite values in upscaled LR")

        for local_index in range(int(batch["lr"].shape[0])):
            identity = distillation_sample_identity(batch, local_index)
            identities.append(identity)
            observed += 1

    if observed != count:
        raise RuntimeError(f"Observed {observed} samples, expected {count}")
    if len({str(item["key"]) for item in identities}) != len(identities):
        raise ValueError("Distillation identity keys are not unique")

    first = dataset[0]
    report = {
        "kind": "ultravideo_materialized_lr_interface_smoke_v1",
        "dataset_length": len(dataset),
        "checked_samples": count,
        "first_lr_shape": list(first["lr"].shape),
        "first_lr_min": float(first["lr"].min().item()),
        "first_lr_max": float(first["lr"].max().item()),
        "first_target_height": int(first["target_height"]),
        "first_target_width": int(first["target_width"]),
        "identity_keys_unique": True,
        "target_is_absent_by_design": True,
        "prepared_target_geometry": [
            int(first["target_height"]),
            int(first["target_width"]),
        ],
        "first_identity": identities[0],
    }
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
