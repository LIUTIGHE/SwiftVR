#!/usr/bin/env python3
"""Validate Stage-B1 ``z_SR`` cache metadata, identities, files, and tensor shapes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.data import TripletVideoDataset
from swiftvr.training.distillation import DeterministicTripletViewDataset, distillation_sample_identity
from swiftvr.training.reference import sha256_file
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--check-all-files", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache = TinyDecoderLatentCache(args.cache)
    metadata = cache.metadata
    manifests_raw = metadata.get("manifests")
    if not isinstance(manifests_raw, list) or not manifests_raw:
        raise ValueError("Cache metadata is missing manifests")
    manifests = [Path(str(value)).expanduser().resolve() for value in manifests_raw]
    path_root = args.path_root.expanduser().resolve()

    base = TripletVideoDataset(
        manifests,
        split=str(metadata["split"]),
        training=True,
        clip_length=int(metadata["clip_length"]),
        crop_size=int(metadata["crop_size"]),
        scale=int(metadata["scale"]),
        load_hq=False,
        horizontal_flip_probability=float(metadata["horizontal_flip_probability"]),
        vertical_flip_probability=float(metadata["vertical_flip_probability"]),
        drop_short_sequences=True,
        path_root=path_root,
        verify_paths=args.verify_paths,
    )
    views = DeterministicTripletViewDataset(
        base,
        views_per_record=int(metadata["views_per_record"]),
        view_seed=int(metadata["view_seed"]),
    )
    cache.validate_dataset(
        manifests=manifests,
        split=str(metadata["split"]),
        clip_length=int(metadata["clip_length"]),
        crop_size=int(metadata["crop_size"]),
        scale=int(metadata["scale"]),
        views_per_record=int(metadata["views_per_record"]),
        view_seed=int(metadata["view_seed"]),
        horizontal_flip_probability=float(metadata["horizontal_flip_probability"]),
        vertical_flip_probability=float(metadata["vertical_flip_probability"]),
        dataset_length=len(views),
    )
    indices = cache.selected_indices()
    out_of_range = [index for index in indices if index < 0 or index >= len(views)]
    if out_of_range:
        raise ValueError(f"Selected indices outside dataset: {out_of_range[:8]}")
    if set(indices) != set(cache.samples_by_index):
        raise ValueError("selected_indices and cached sample indices differ")

    source_checkpoint = Path(str(metadata["source_checkpoint"])).expanduser().resolve()
    source_weights = source_checkpoint / "trainable.safetensors"
    source_metadata = source_checkpoint / "metadata.json"
    if source_weights.is_file():
        actual = sha256_file(source_weights)
        if actual != metadata.get("source_weights_sha256"):
            raise ValueError("Source checkpoint trainable.safetensors hash mismatch")
    if source_metadata.is_file():
        actual = sha256_file(source_metadata)
        if actual != metadata.get("source_metadata_sha256"):
            raise ValueError("Source checkpoint metadata.json hash mismatch")

    probe_positions = sorted(set((0, len(indices) // 2, len(indices) - 1)))
    probe_shapes: set[tuple[int, ...]] = set()
    probe_dtypes: set[str] = set()
    for position in probe_positions:
        index = indices[position]
        sample = views[index]
        batch = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)
            elif isinstance(value, (int, float, bool)):
                batch[key] = torch.tensor([value])
            else:
                batch[key] = [value]
        identity = distillation_sample_identity(batch, 0)
        latent = cache.load(identity, device="cpu")
        if latent.ndim != 4:
            raise ValueError(f"Expected cached z_SR [F,C,H,W], got {tuple(latent.shape)}")
        if int(latent.shape[1]) != 48:
            raise ValueError(f"Expected 48 latent channels, got shape={tuple(latent.shape)}")
        if not torch.isfinite(latent.float()).all():
            raise FloatingPointError(f"Non-finite z_SR in dataset index {index}")
        probe_shapes.add(tuple(int(value) for value in latent.shape))
        probe_dtypes.add(str(latent.dtype).removeprefix("torch."))

    if args.check_all_files:
        for index, item in cache.samples_by_index.items():
            filename = item.get("file")
            if not isinstance(filename, str) or not (cache.root / filename).is_file():
                raise FileNotFoundError(f"Missing cache file for dataset index {index}: {filename}")

    result = {
        "status": "PASS",
        "cache": str(cache.root),
        "source_checkpoint": str(source_checkpoint),
        "sample_count": int(metadata["sample_count"]),
        "base_record_count": int(metadata["base_record_count"]),
        "unique_source_count": int(metadata["unique_source_count"]),
        "selected_record_count": int(metadata["selected_record_count"]),
        "selected_unique_source_count": int(metadata["selected_unique_source_count"]),
        "views_per_record": int(metadata["views_per_record"]),
        "view_seed": int(metadata["view_seed"]),
        "selection_mode": str(metadata["selection_mode"]),
        "selected_indices_sha256": metadata.get("selected_indices_sha256"),
        "storage_dtype": metadata.get("storage_dtype"),
        "probe_shapes": [list(shape) for shape in sorted(probe_shapes)],
        "probe_dtypes": sorted(probe_dtypes),
        "all_files_checked": bool(args.check_all_files),
    }
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
