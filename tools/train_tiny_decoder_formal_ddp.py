#!/usr/bin/env python3
"""Formal 4-GPU Stage-B1 training for the compact SwiftVR Tiny Decoder.

The long-run Stage-A encoder/DiT do not run in this trainer. Their deterministic
SR latents (z_SR) are read from immutable offline caches. A frozen ReAE decoder
renders the same z_SR online to provide the decoder-teacher RGB target, while only
the compact TinyConditionalDecoder is optimized with the validated dual pixel +
LPIPS objective.

Checkpoints are deliberately written at epoch boundaries. Resume is exact at that
boundary for a fixed world size because the dataset views are deterministic and the
DistributedSampler permutation is a pure function of (seed, epoch).
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder
from swiftvr.training import build_fp32_adamw, build_grad_scaler, cast_trainable_parameters
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.input_pipeline import dataloader_worker_kwargs
from swiftvr.training.reference import sha256_file
from swiftvr.training.stage3 import VideoMetricAccumulator
from swiftvr.training.tiny_decoder import LPIPSAlexLoss, tiny_decoder_objective
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
STATE_FILENAME = "training_state.pt"
LATEST_FILENAME = "latest.json"
BEST_FILENAME = "best.json"
RUN_CONFIG_FILENAME = "run_config.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--init-decoder", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--val-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--val-crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--views-per-record", type=int, default=8)
    parser.add_argument("--view-seed", type=int, default=20260805)
    parser.add_argument("--val-views-per-record", type=int, default=1)
    parser.add_argument("--val-view-seed", type=int, default=9000001)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.0)
    parser.add_argument("--val-horizontal-flip-probability", type=float, default=0.0)
    parser.add_argument("--val-vertical-flip-probability", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-rank train batch size")
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gt-l2-weight", type=float, default=1.0)
    parser.add_argument("--teacher-l2-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=2.0)
    parser.add_argument("--lpips-microbatch-frames", type=int, default=16)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ddp-timeout-seconds", type=int, default=1800)
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument(
        "--resume",
        default=None,
        help="Epoch-boundary checkpoint directory or 'latest' under --output-dir.",
    )
    return parser


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "clip-length": args.clip_length,
        "crop-size": args.crop_size,
        "val-crop-size": args.val_crop_size,
        "scale": args.scale,
        "views-per-record": args.views_per_record,
        "val-views-per-record": args.val_views_per_record,
        "batch-size": args.batch_size,
        "val-batch-size": args.val_batch_size,
        "epochs": args.epochs,
        "learning-rate": args.learning_rate,
        "optimizer-eps": args.optimizer_eps,
        "max-grad-norm": args.max_grad_norm,
        "lpips-microbatch-frames": args.lpips_microbatch_frames,
        "log-every": args.log_every,
        "ddp-timeout-seconds": args.ddp_timeout_seconds,
    }
    bad = [name for name, value in positive.items() if float(value) <= 0]
    if bad:
        raise ValueError(f"Arguments must be positive: {bad}")
    if args.num_workers < 0 or args.val_num_workers < 0:
        raise ValueError("num-workers values must be non-negative")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if min(args.gt_l2_weight, args.teacher_l2_weight, args.lpips_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if args.clip_length % 4 != 1:
        raise ValueError("SwiftVR formal decoder clips must satisfy T=4k+1")
    for name, value in (
        ("horizontal-flip-probability", args.horizontal_flip_probability),
        ("vertical-flip-probability", args.vertical_flip_probability),
        ("val-horizontal-flip-probability", args.val_horizontal_flip_probability),
        ("val-vertical-flip-probability", args.val_vertical_flip_probability),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")


def _init_distributed(timeout_seconds: int) -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Stage-B1 training requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        raise RuntimeError("Use torchrun with multiple GPUs for formal Stage-B1 training")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=_datetime.timedelta(seconds=int(timeout_seconds)),
    )
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _cache_subset(
    manifests: list[Path],
    cache: TinyDecoderLatentCache,
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
    base = TripletVideoDataset(
        manifests,
        split=split,
        training=True,
        clip_length=clip_length,
        crop_size=crop_size,
        scale=scale,
        load_hq=False,
        horizontal_flip_probability=hflip,
        vertical_flip_probability=vflip,
        drop_short_sequences=True,
        path_root=path_root,
        verify_paths=verify_paths,
    )
    views = DeterministicTripletViewDataset(
        base,
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
    indices = cache.selected_indices()
    if any(index < 0 or index >= len(views) for index in indices):
        raise ValueError("Tiny-decoder cache selected index exceeds deterministic-view dataset")
    if set(indices) != set(cache.samples_by_index):
        raise ValueError("Tiny-decoder cache files do not match selected_indices")
    return Subset(views, list(indices))


def _validate_cache_pair(train: TinyDecoderLatentCache, val: TinyDecoderLatentCache) -> None:
    for key in (
        "base_checkpoint",
        "source_checkpoint",
        "source_weights_sha256",
        "source_metadata_sha256",
        "reae_sha256",
        "transformer_config_sha256",
    ):
        if train.metadata.get(key) != val.metadata.get(key):
            raise ValueError(
                f"Train/validation z_SR caches differ in {key}: "
                f"train={train.metadata.get(key)!r}, val={val.metadata.get(key)!r}"
            )


def _move_pixels(batch: Mapping[str, object], device: torch.device, dtype: torch.dtype):
    result = dict(batch)
    for key in ("lr", "hr"):
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, dtype=dtype, non_blocking=True)
    return result


def _train_loader(dataset, sampler, args: argparse.Namespace) -> DataLoader:
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
        **worker_kwargs,
    )


def _val_loader(dataset, args: argparse.Namespace) -> DataLoader:
    worker_kwargs = dataloader_worker_kwargs(
        num_workers=args.val_num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    return DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory,
        **worker_kwargs,
    )


def _fingerprint(
    args: argparse.Namespace,
    *,
    world_size: int,
    train_cache: TinyDecoderLatentCache,
    val_cache: TinyDecoderLatentCache,
    init_decoder: Path,
) -> dict[str, object]:
    config_path = init_decoder / "config.json"
    weights_path = init_decoder / "model.safetensors"
    return {
        "trainer": "swiftvr_stage_b1_tiny_decoder_formal_ddp_v1",
        "world_size": int(world_size),
        "base_checkpoint": str(args.base_checkpoint.expanduser().resolve()),
        "init_decoder": str(init_decoder),
        "init_decoder_config_sha256": sha256_file(config_path),
        "init_decoder_weights_sha256": sha256_file(weights_path),
        "train_cache": str(train_cache.root),
        "train_cache_metadata_sha256": sha256_file(train_cache.root / "metadata.json"),
        "val_cache": str(val_cache.root),
        "val_cache_metadata_sha256": sha256_file(val_cache.root / "metadata.json"),
        "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
        "val_manifests": [str(path.expanduser().resolve()) for path in args.val_manifest],
        "split": args.split,
        "val_split": args.val_split,
        "clip_length": int(args.clip_length),
        "crop_size": int(args.crop_size),
        "val_crop_size": int(args.val_crop_size),
        "scale": int(args.scale),
        "views_per_record": int(args.views_per_record),
        "view_seed": int(args.view_seed),
        "val_views_per_record": int(args.val_views_per_record),
        "val_view_seed": int(args.val_view_seed),
        "horizontal_flip_probability": float(args.horizontal_flip_probability),
        "vertical_flip_probability": float(args.vertical_flip_probability),
        "val_horizontal_flip_probability": float(args.val_horizontal_flip_probability),
        "val_vertical_flip_probability": float(args.val_vertical_flip_probability),
        "local_batch_size": int(args.batch_size),
        "global_batch_size": int(args.batch_size * world_size),
        "dtype": args.dtype,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "optimizer_eps": float(args.optimizer_eps),
        "max_grad_norm": float(args.max_grad_norm),
        "gt_l2_weight": float(args.gt_l2_weight),
        "teacher_l2_weight": float(args.teacher_l2_weight),
        "lpips_weight": float(args.lpips_weight),
        "lpips_microbatch_frames": int(args.lpips_microbatch_frames),
        "seed": int(args.seed),
    }


def _assert_fingerprint(saved: Mapping[str, object], current: Mapping[str, object]) -> None:
    # epochs/log cadence may change on resume; everything affecting optimization/data may not.
    differences = [
        key for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]
    if differences:
        detail = ", ".join(
            f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
            for key in differences[:16]
        )
        raise ValueError(f"Resume fingerprint mismatch: {detail}")


def _resolve_resume(run_dir: Path, resume: str) -> Path:
    if resume != "latest":
        path = Path(resume).expanduser().resolve()
    else:
        pointer = run_dir / LATEST_FILENAME
        if not pointer.is_file():
            raise FileNotFoundError(pointer)
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        candidate = Path(str(payload["checkpoint"]))
        path = candidate if candidate.is_absolute() else (run_dir / candidate)
        path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _save_checkpoint(
    run_dir: Path,
    *,
    model: TinyConditionalDecoder,
    optimizer: torch.optim.Optimizer,
    scaler,
    completed_epoch: int,
    global_step: int,
    fingerprint: Mapping[str, object],
    validation: Mapping[str, object],
    best_val_loss: float,
    is_best: bool,
) -> Path:
    checkpoint = run_dir / "checkpoints" / f"epoch_{completed_epoch:03d}_step_{global_step:08d}"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    model.save_pretrained(checkpoint / "tiny_decoder")
    state = {
        "completed_epoch": int(completed_epoch),
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "fingerprint": dict(fingerprint),
        "best_val_loss": float(best_val_loss),
        "validation": dict(validation),
    }
    temporary = checkpoint / (STATE_FILENAME + ".tmp")
    torch.save(state, temporary)
    temporary.replace(checkpoint / STATE_FILENAME)
    relative = checkpoint.relative_to(run_dir).as_posix()
    _write_json(run_dir / LATEST_FILENAME, {"checkpoint": relative})
    if is_best:
        _write_json(
            run_dir / BEST_FILENAME,
            {
                "checkpoint": relative,
                "completed_epoch": int(completed_epoch),
                "global_step": int(global_step),
                "val_loss": float(validation["loss"]),
                "validation": dict(validation),
            },
        )
    return checkpoint


def _load_training_state(path: Path):
    state_path = path / STATE_FILENAME
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise TypeError("training_state.pt must contain a mapping")
    return state


def _allreduce_interval(
    sums: Mapping[str, float],
    count: int,
    *,
    grad_sum: float,
    seconds: float,
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    keys = tuple(sums)
    packed = torch.tensor(
        [float(sums[key]) for key in keys] + [float(count), grad_sum, seconds],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    denominator = max(float(packed[len(keys)].item()), 1.0)
    result = {
        key: float(packed[index].item()) / denominator
        for index, key in enumerate(keys)
    }
    result["grad_norm"] = float(packed[-2].item()) / max(float(count * world_size), 1.0)
    result["step_seconds"] = float(packed[-1].item()) / world_size
    return result


def _validate(
    model: TinyConditionalDecoder,
    reae: ReAE,
    cache: TinyDecoderLatentCache,
    loader: DataLoader,
    perceptual: LPIPSAlexLoss | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    model.eval()
    reae.eval()
    objective_sums = {
        "loss": 0.0,
        "gt_l2": 0.0,
        "teacher_l2": 0.0,
        "gt_lpips": 0.0,
        "teacher_lpips": 0.0,
        "gt_temporal_mse": 0.0,
        "teacher_temporal_mse": 0.0,
    }
    sample_count = 0
    tiny_gt = VideoMetricAccumulator()
    tiny_teacher = VideoMetricAccumulator()
    teacher_gt = VideoMetricAccumulator()
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)

    for batch_cpu in loader:
        moved = _move_pixels(batch_cpu, device, dtype)
        prepared = prepare_training_batch(moved)
        lq_input = prepared["lq_input"]
        target = prepared["target"]
        if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("Validation batch is missing lq_input/target")
        z_sr = cache.load_batch(batch_cpu, device=device, dtype=dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            teacher = decode_reae_clip(
                reae,
                z_sr,
                output_frames=int(target.shape[1]),
                clamp=False,
            )
            prediction = model(
                z_sr,
                lq_input,
                output_frames=int(target.shape[1]),
                clamp=False,
            )
        # LPIPS needs gradients only in training; validation can remain no-grad.
        with torch.no_grad():
            objective = tiny_decoder_objective(
                prediction,
                target,
                teacher,
                perceptual=perceptual,
                gt_l2_weight=args.gt_l2_weight,
                teacher_l2_weight=args.teacher_l2_weight,
                lpips_weight=args.lpips_weight,
                lpips_microbatch_frames=args.lpips_microbatch_frames,
            )
        batch_size = int(target.shape[0])
        sample_count += batch_size
        for key in objective_sums:
            objective_sums[key] += float(objective[key].detach().item()) * batch_size
        tiny_gt.update(prediction, target, clamp=True)
        tiny_teacher.update(prediction, teacher, clamp=True)
        teacher_gt.update(teacher, target, clamp=True)

    if sample_count <= 0:
        raise RuntimeError("Validation loader is empty")
    result: dict[str, float | int] = {
        key: value / sample_count for key, value in objective_sums.items()
    }
    result["samples"] = sample_count
    for prefix, accumulator in (
        ("tiny_gt", tiny_gt),
        ("tiny_teacher", tiny_teacher),
        ("reae_teacher_gt", teacher_gt),
    ):
        for key, value in accumulator.compute().items():
            result[f"{prefix}_{key}"] = value
    return result


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    rank, local_rank, world_size, device = _init_distributed(args.ddp_timeout_seconds)
    try:
        _seed(args.seed + rank)
        dtype = DTYPES[args.dtype]
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

        run_dir = args.output_dir.expanduser().resolve()
        path_root = args.path_root.expanduser().resolve()
        base = args.base_checkpoint.expanduser().resolve()
        init_decoder = args.init_decoder.expanduser().resolve()
        train_cache = TinyDecoderLatentCache(args.train_cache)
        val_cache = TinyDecoderLatentCache(args.val_cache)
        _validate_cache_pair(train_cache, val_cache)

        train_dataset = _cache_subset(
            args.manifest,
            train_cache,
            split=args.split,
            path_root=path_root,
            clip_length=args.clip_length,
            crop_size=args.crop_size,
            scale=args.scale,
            views_per_record=args.views_per_record,
            view_seed=args.view_seed,
            hflip=args.horizontal_flip_probability,
            vflip=args.vertical_flip_probability,
            verify_paths=args.verify_paths,
        )
        val_dataset = _cache_subset(
            args.val_manifest,
            val_cache,
            split=args.val_split,
            path_root=path_root,
            clip_length=args.clip_length,
            crop_size=args.val_crop_size,
            scale=args.scale,
            views_per_record=args.val_views_per_record,
            view_seed=args.val_view_seed,
            hflip=args.val_horizontal_flip_probability,
            vflip=args.val_vertical_flip_probability,
            verify_paths=args.verify_paths,
        )
        if len(train_dataset) != 15896:
            raise ValueError(f"Formal B1 train cache must contain 15896 views, got {len(train_dataset)}")
        if len(val_dataset) != 13:
            raise ValueError(f"Primary formal B1 validation must contain 13 views, got {len(val_dataset)}")

        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        train_loader = _train_loader(train_dataset, sampler, args)
        val_loader = _val_loader(val_dataset, args) if rank == 0 else None

        reae = ReAE(str(base / args.reae_filename)).to(device=device, dtype=dtype).eval()
        for parameter in reae.parameters():
            parameter.requires_grad_(False)

        resume_checkpoint: Path | None = None
        if args.resume is not None:
            resume_checkpoint = _resolve_resume(run_dir, args.resume) if rank == 0 else None
            payload = [str(resume_checkpoint) if rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            resume_checkpoint = Path(str(payload[0])).resolve()
            tiny_root = resume_checkpoint / "tiny_decoder"
        else:
            tiny_root = init_decoder

        tiny = TinyConditionalDecoder.from_pretrained(tiny_root, device=device, dtype=None)
        if tiny.block_mode != "compact":
            raise ValueError(f"Formal B1 requires materialized compact decoder, got {tiny.block_mode!r}")
        if tuple(tiny.block_internal_channels or ()) != (80, 48, 24, 16):
            raise ValueError(
                "Formal B1 topology is frozen to keep_040 internal widths "
                f"(80,48,24,16), got {tiny.block_internal_channels}"
            )
        cast_trainable_parameters(tiny, dtype=torch.float32)
        tiny.train()
        optimizer = build_fp32_adamw(
            tiny,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.optimizer_eps,
        )
        scaler = build_grad_scaler(device, dtype)
        perceptual = LPIPSAlexLoss().to(device=device).eval() if args.lpips_weight > 0 else None

        fingerprint = _fingerprint(
            args,
            world_size=world_size,
            train_cache=train_cache,
            val_cache=val_cache,
            init_decoder=init_decoder,
        )
        start_epoch = 0
        global_step = 0
        best_val_loss = math.inf
        if resume_checkpoint is not None:
            state = _load_training_state(resume_checkpoint)
            saved_fingerprint = state.get("fingerprint")
            if not isinstance(saved_fingerprint, Mapping):
                raise TypeError("Resume checkpoint has no valid fingerprint")
            _assert_fingerprint(saved_fingerprint, fingerprint)
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_epoch = int(state["completed_epoch"])
            global_step = int(state["global_step"])
            best_val_loss = float(state.get("best_val_loss", math.inf))
            if start_epoch >= args.epochs:
                raise ValueError(
                    f"Checkpoint already completed {start_epoch} epochs; requested epochs={args.epochs}"
                )

        ddp = DDP(tiny, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        autocast_enabled = dtype in (torch.float16, torch.bfloat16)

        if rank == 0:
            if args.resume is None and run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                run_dir / RUN_CONFIG_FILENAME,
                {
                    **fingerprint,
                    "epochs": int(args.epochs),
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "steps_per_epoch_per_rank": len(train_loader),
                    "decoder_block_internal_channels": list(tiny.block_internal_channels or ()),
                    "decoder_parameters": sum(p.numel() for p in tiny.parameters()),
                },
            )
        dist.barrier()

        # Establish an epoch-0 quality baseline for a fresh run.
        if args.resume is None:
            dist.barrier()
            if rank == 0:
                assert val_loader is not None
                initial_val = _validate(
                    tiny,
                    reae,
                    val_cache,
                    val_loader,
                    perceptual,
                    device=device,
                    dtype=dtype,
                    args=args,
                )
                _write_json(run_dir / "validation_epoch_000.json", initial_val)
                _append_jsonl(run_dir / "val_log.jsonl", {"completed_epoch": 0, "global_step": 0, **initial_val})
                print(json.dumps({"phase": "initial_validation", **initial_val}, sort_keys=True), flush=True)
            dist.barrier()

        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            ddp.train()
            interval_sums = {
                "loss": 0.0,
                "gt_l2": 0.0,
                "teacher_l2": 0.0,
                "gt_lpips": 0.0,
                "teacher_lpips": 0.0,
                "gt_temporal_mse": 0.0,
                "teacher_temporal_mse": 0.0,
            }
            interval_count = 0
            interval_grad = 0.0
            interval_started = time.perf_counter()

            for batch_index, batch_cpu in enumerate(train_loader, start=1):
                moved = _move_pixels(batch_cpu, device, dtype)
                prepared = prepare_training_batch(moved)
                lq_input = prepared["lq_input"]
                target = prepared["target"]
                if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
                    raise TypeError("Training batch is missing lq_input/target")
                z_sr = train_cache.load_batch(batch_cpu, device=device, dtype=dtype)
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                    teacher = decode_reae_clip(
                        reae,
                        z_sr,
                        output_frames=int(target.shape[1]),
                        clamp=False,
                    )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                    prediction = ddp(
                        z_sr,
                        lq_input,
                        output_frames=int(target.shape[1]),
                        clamp=False,
                    )
                objective = tiny_decoder_objective(
                    prediction,
                    target,
                    teacher,
                    perceptual=perceptual,
                    gt_l2_weight=args.gt_l2_weight,
                    teacher_l2_weight=args.teacher_l2_weight,
                    lpips_weight=args.lpips_weight,
                    lpips_microbatch_frames=args.lpips_microbatch_frames,
                )
                scaler.scale(objective["loss"]).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), args.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite grad norm epoch={epoch + 1} batch={batch_index}: {grad_norm}"
                    )
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

                batch_size = int(target.shape[0])
                interval_count += batch_size
                interval_grad += float(grad_norm.detach().item()) * batch_size
                for key in interval_sums:
                    interval_sums[key] += float(objective[key].detach().item()) * batch_size

                should_log = batch_index % args.log_every == 0 or batch_index == len(train_loader)
                if should_log:
                    elapsed = time.perf_counter() - interval_started
                    reduced = _allreduce_interval(
                        interval_sums,
                        interval_count,
                        grad_sum=interval_grad,
                        seconds=elapsed,
                        device=device,
                        world_size=world_size,
                    )
                    if rank == 0:
                        record = {
                            "epoch": epoch + 1,
                            "batch": batch_index,
                            "batches_per_epoch": len(train_loader),
                            "global_step": global_step,
                            "global_batch_size": args.batch_size * world_size,
                            **reduced,
                        }
                        _append_jsonl(run_dir / "train_log.jsonl", record)
                        print(json.dumps(record, sort_keys=True), flush=True)
                    interval_sums = {key: 0.0 for key in interval_sums}
                    interval_count = 0
                    interval_grad = 0.0
                    interval_started = time.perf_counter()

            dist.barrier()
            validation: dict[str, float | int] | None = None
            if rank == 0:
                assert val_loader is not None
                validation = _validate(
                    tiny,
                    reae,
                    val_cache,
                    val_loader,
                    perceptual,
                    device=device,
                    dtype=dtype,
                    args=args,
                )
                completed_epoch = epoch + 1
                record = {"completed_epoch": completed_epoch, "global_step": global_step, **validation}
                _append_jsonl(run_dir / "val_log.jsonl", record)
                _write_json(run_dir / f"validation_epoch_{completed_epoch:03d}.json", validation)
                current_val_loss = float(validation["loss"])
                is_best = current_val_loss < best_val_loss
                if is_best:
                    best_val_loss = current_val_loss
                checkpoint = _save_checkpoint(
                    run_dir,
                    model=tiny,
                    optimizer=optimizer,
                    scaler=scaler,
                    completed_epoch=completed_epoch,
                    global_step=global_step,
                    fingerprint=fingerprint,
                    validation=validation,
                    best_val_loss=best_val_loss,
                    is_best=is_best,
                )
                print(
                    json.dumps(
                        {
                            "phase": "epoch_validation",
                            "checkpoint": str(checkpoint),
                            "is_best": is_best,
                            **record,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            best_payload = [best_val_loss if rank == 0 else None]
            dist.broadcast_object_list(best_payload, src=0)
            best_val_loss = float(best_payload[0])
            dist.barrier()

        if rank == 0:
            best = json.loads((run_dir / BEST_FILENAME).read_text(encoding="utf-8"))
            summary = {
                "status": "PASS",
                "completed_epochs": int(args.epochs),
                "global_step": int(global_step),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "world_size": world_size,
                "local_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "best": best,
            }
            _write_json(run_dir / "summary.json", summary)
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        dist.barrier()
        return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
