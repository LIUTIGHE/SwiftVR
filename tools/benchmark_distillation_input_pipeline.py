#!/usr/bin/env python3
"""Benchmark deterministic triplet decoding and teacher-cache loading on CPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from swiftvr.data import TripletVideoDataset
from swiftvr.training.distillation import (
    DeterministicTripletViewDataset,
    TeacherVelocityCache,
)
from swiftvr.training.distillation_generalization import build_cache_backed_subset
from swiftvr.training.input_pipeline import dataloader_worker_kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--views-per-record", type=int, default=2)
    parser.add_argument("--view-seed", type=int, default=20260805)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--load-hq", action="store_true")
    parser.add_argument("--warmup-samples", type=int, default=16)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--pin-memory", action="store_true")
    return parser


def _next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.samples <= 0 or args.warmup_samples < 0:
        raise ValueError("batch-size/samples must be positive and warmup non-negative")

    base = TripletVideoDataset(
        args.manifest,
        split=args.split,
        training=True,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        load_hq=args.load_hq,
        horizontal_flip_probability=args.horizontal_flip_probability,
        vertical_flip_probability=args.vertical_flip_probability,
        drop_short_sequences=True,
        path_root=args.path_root,
    )
    views = DeterministicTripletViewDataset(
        base,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
    )
    cache = TeacherVelocityCache(args.teacher_cache)
    cache.validate_dataset(
        manifests=args.manifest,
        split=args.split,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
        horizontal_flip_probability=args.horizontal_flip_probability,
        vertical_flip_probability=args.vertical_flip_probability,
        dataset_length=len(views),
    )
    dataset = build_cache_backed_subset(views, cache)
    generator = torch.Generator().manual_seed(0)
    worker_kwargs = dataloader_worker_kwargs(
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=args.pin_memory,
        generator=generator,
        **worker_kwargs,
    )
    iterator = iter(loader)

    for _ in range(args.warmup_samples):
        batch, iterator = _next_or_restart(iterator, loader)
        cache.load_batch(batch, device="cpu", dtype=torch.float16)

    data_seconds = 0.0
    cache_seconds = 0.0
    measured_samples = 0
    while measured_samples < args.samples:
        started = time.perf_counter()
        batch, iterator = _next_or_restart(iterator, loader)
        data_seconds += time.perf_counter() - started

        started = time.perf_counter()
        cache.load_batch(batch, device="cpu", dtype=torch.float16)
        cache_seconds += time.perf_counter() - started
        measured_samples += int(batch["frame_indices"].shape[0])

    effective = measured_samples
    result = {
        "requested_samples": int(args.samples),
        "measured_samples": effective,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": (
            int(args.prefetch_factor) if int(args.num_workers) > 0 else None
        ),
        "persistent_workers": bool(
            args.persistent_workers and int(args.num_workers) > 0
        ),
        "load_hq": bool(args.load_hq),
        "dataset_seconds_total": data_seconds,
        "dataset_seconds_per_sample": data_seconds / effective,
        "cache_seconds_total": cache_seconds,
        "cache_seconds_per_sample": cache_seconds / effective,
        "combined_seconds_per_sample": (data_seconds + cache_seconds) / effective,
        "samples_per_second": effective / (data_seconds + cache_seconds),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
