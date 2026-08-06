#!/usr/bin/env python3
"""Prefetched DDP teacher distillation with optional HQ decoding disabled.

This entry point deliberately wraps the validated generalization trainer rather
than duplicating its optimization loop.  It changes only the train Dataset and
DataLoader path, while retaining the existing loss, checkpoint, validation,
DDP, and same-world-size exact-resume implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader, DistributedSampler

try:
    import train_teacher_distillation_generalization_ddp as base
except ModuleNotFoundError:
    from tools import train_teacher_distillation_generalization_ddp as base

from swiftvr.training.distillation import (
    DeterministicTripletViewDataset,
    TeacherVelocityCache,
)
from swiftvr.training.distillation_generalization import build_cache_backed_subset
from swiftvr.training.input_pipeline import dataloader_worker_kwargs


_ORIGINAL_BUILD_PARSER = base.build_parser
_ORIGINAL_VALIDATE_ARGUMENTS = base.gate._validate_arguments
_ORIGINAL_FINGERPRINT = base._fingerprint
_RUNTIME_ARGS: argparse.Namespace | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = _ORIGINAL_BUILD_PARSER()
    parser.description = __doc__
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched by each worker when --num-workers is positive.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep each epoch's DataLoader workers alive while its iterator is active.",
    )
    parser.add_argument(
        "--load-train-hq",
        action="store_true",
        help="Decode the unused HQ reference in training batches (off by default).",
    )
    return parser


def _validate_arguments(args: argparse.Namespace) -> tuple[int, ...]:
    """Reuse all baseline checks while permitting a positive worker count."""

    workers = int(args.num_workers)
    args.num_workers = 0
    try:
        visual_indices = _ORIGINAL_VALIDATE_ARGUMENTS(args)
    finally:
        args.num_workers = workers

    dataloader_worker_kwargs(
        num_workers=workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    return visual_indices


def _is_train_cache(cache: TeacherVelocityCache) -> bool:
    if _RUNTIME_ARGS is None:
        raise RuntimeError("Throughput trainer runtime arguments are not initialized")
    expected = Path(_RUNTIME_ARGS.teacher_cache).expanduser().resolve()
    return cache.root == expected


def _build_cached_dataset(
    manifests: list[Path],
    cache: TeacherVelocityCache,
    *,
    split: str,
    path_root: Path,
    clip_length: int,
    crop_size: int,
    scale: int,
    views_per_record: int,
    view_seed: int,
    hflip: float,
    vflip: float,
    verify_paths: bool,
):
    if _RUNTIME_ARGS is None:
        raise RuntimeError("Throughput trainer runtime arguments are not initialized")
    is_train = _is_train_cache(cache)
    load_hq = bool(_RUNTIME_ARGS.load_train_hq) if is_train else True
    dataset = base.gate.TripletVideoDataset(
        manifests,
        split=split,
        training=True,
        clip_length=clip_length,
        crop_size=crop_size,
        scale=scale,
        load_hq=load_hq,
        horizontal_flip_probability=hflip,
        vertical_flip_probability=vflip,
        drop_short_sequences=True,
        path_root=path_root,
        verify_paths=verify_paths,
    )
    views = DeterministicTripletViewDataset(
        dataset,
        views_per_record=views_per_record,
        view_seed=view_seed,
    )
    cache.validate_dataset(
        manifests=manifests,
        split=split,
        clip_length=clip_length,
        crop_size=crop_size,
        scale=scale,
        views_per_record=views_per_record,
        view_seed=view_seed,
        horizontal_flip_probability=hflip,
        vertical_flip_probability=vflip,
        dataset_length=len(views),
    )
    return build_cache_backed_subset(views, cache)


def _make_train_loader(
    dataset,
    *,
    rank: int,
    world_size: int,
    epoch: int,
    args: argparse.Namespace,
) -> DataLoader:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    sampler.set_epoch(int(epoch))
    generator = torch.Generator()
    generator.manual_seed(args.seed + rank + 1000003 * int(epoch))
    worker_kwargs = dataloader_worker_kwargs(
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        pin_memory=args.pin_memory,
        generator=generator,
        **worker_kwargs,
    )


def _fingerprint(
    args: argparse.Namespace,
    *,
    base_checkpoint: Path,
    train_cache: TeacherVelocityCache,
    val_cache: TeacherVelocityCache | None,
    runtime_dtype: torch.dtype,
    world_size: int,
    train_dataset_length: int,
    batches_per_rank_epoch: int,
) -> dict[str, object]:
    result = _ORIGINAL_FINGERPRINT(
        args,
        base_checkpoint=base_checkpoint,
        train_cache=train_cache,
        val_cache=val_cache,
        runtime_dtype=runtime_dtype,
        world_size=world_size,
        train_dataset_length=train_dataset_length,
        batches_per_rank_epoch=batches_per_rank_epoch,
    )
    result.update(
        {
            "trainer": "teacher_distillation_throughput_ddp_v1",
            "num_workers": int(args.num_workers),
            "prefetch_factor": (
                int(args.prefetch_factor) if int(args.num_workers) > 0 else None
            ),
            "persistent_workers": bool(
                args.persistent_workers and int(args.num_workers) > 0
            ),
            "load_train_hq": bool(args.load_train_hq),
        }
    )
    return result


def main() -> int:
    global _RUNTIME_ARGS

    # Parse once to configure the wrapped Dataset path.  The baseline main then
    # parses the same argv through this parser and owns all execution afterward.
    _RUNTIME_ARGS = build_parser().parse_args()
    base.build_parser = build_parser
    base.gate._validate_arguments = _validate_arguments
    base._build_cached_dataset = _build_cached_dataset
    base._make_train_loader = _make_train_loader
    base._fingerprint = _fingerprint
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
