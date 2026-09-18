#!/usr/bin/env python3
"""Single-node multi-GPU DDP trainer for SwiftVR Stage-3 reconstruction.

The frozen ReAE and folded 5B DiT are replicated on each GPU, while only the
prompt-free adapters are synchronized and optimized. The implementation keeps
PyTorch DDP semantics explicit: rank-sharded training data, ``no_sync`` during
local gradient accumulation, globally reduced validation metrics, rank-0 delta
checkpoints, and per-rank RNG files for exact resume with the same world size.

A one-time ``--init-from-single-checkpoint`` migration can inherit model,
AdamW, and GradScaler state from the validated single-GPU trainer. Because the
data topology changes, that migration deliberately starts a fresh distributed
dataloader cursor and is not claimed to be sample-for-sample identical.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from smoke_training_forward import (
    _CANONICAL_DTYPE_NAME,
    _format_count,
    configure_train_scope,
    gradient_summary,
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training import (
    SwiftVRTrainingForward,
    TrainingCursor,
    VideoMetricAccumulator,
    advance_cursor_batches,
    append_jsonl,
    build_fp32_adamw,
    build_grad_scaler,
    capture_rng_state,
    cast_trainable_parameters,
    load_delta_checkpoint,
    load_trainer_state,
    optimizer_state_summary,
    resolve_resume_checkpoint,
    restore_rng_state,
    save_delta_checkpoint,
    save_trainer_state,
    seed_everything,
    skip_batches,
    stage3_reconstruction_objective,
    trainable_named_parameters,
    write_latest_checkpoint,
)
from swiftvr.training.distributed import (
    DistributedContext,
    DistributedEvalSampler,
    global_effective_batch_size,
)


DISTRIBUTED_STATE_FILENAME = "distributed_state.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SwiftVR Stage-3 adapters with single-node PyTorch DDP."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--val-manifest", type=Path, action="append", default=None)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--val-crop-size", type=int, default=None)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-GPU microbatch size.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.0)
    parser.add_argument(
        "--drop-short-validation",
        action="store_true",
        help="Filter validation sequences shorter than --clip-length.",
    )

    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    parser.add_argument("--allow-dtype-mismatch", action="store_true")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument(
        "--train-scope",
        choices=("adapter",),
        default="adapter",
        help="This DDP gate intentionally supports adapter-only optimization.",
    )
    parser.add_argument("--pixel-loss-weight", type=float, default=1.0)
    parser.add_argument("--temporal-loss-weight", type=float, default=1.0)

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Local accumulation on every rank before one synchronized step.",
    )
    parser.add_argument(
        "--expected-global-batch-size",
        type=int,
        default=None,
        help="Fail if world_size * local batch * accumulation differs.",
    )
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=20)
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=10,
        help="Global validation batch budget before rank sharding.",
    )
    parser.add_argument("--validate-at-start", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="Exact DDP checkpoint directory or 'latest' under --output-dir.",
    )
    parser.add_argument(
        "--init-from-single-checkpoint",
        type=Path,
        default=None,
        help="One-time topology migration from a single-GPU delta checkpoint.",
    )
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    return parser


def _init_distributed() -> tuple[DistributedContext, torch.device]:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "Launch this script with torchrun; missing environment variables: "
            + ", ".join(missing)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL DDP requires CUDA")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    context = DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)
    return context, torch.device("cuda", local_rank)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_pointer(
    run_dir: Path,
    filename: str,
    checkpoint_dir: Path,
    **metadata: object,
) -> None:
    try:
        stored = checkpoint_dir.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        stored = str(checkpoint_dir.resolve())
    _write_json(run_dir / filename, {"checkpoint": stored, **metadata})


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _broadcast_path(path: Path | None, context: DistributedContext) -> Path:
    value = [str(path) if context.is_main and path is not None else None]
    dist.broadcast_object_list(value, src=0)
    if not isinstance(value[0], str):
        raise RuntimeError("Rank 0 did not broadcast a checkpoint path")
    return Path(value[0]).expanduser().resolve()


def _build_train_dataset(args: argparse.Namespace) -> TripletVideoDataset:
    return TripletVideoDataset(
        args.manifest,
        split=args.split,
        training=True,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
        horizontal_flip_probability=args.horizontal_flip_probability,
        vertical_flip_probability=args.vertical_flip_probability,
    )


def _build_train_loader(
    dataset: TripletVideoDataset,
    args: argparse.Namespace,
    context: DistributedContext,
    epoch: int,
) -> tuple[DistributedSampler, DataLoader]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    sampler.set_epoch(int(epoch))
    generator = torch.Generator()
    generator.manual_seed(args.seed + context.rank + 1000003 * int(epoch))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=False,
        generator=generator,
    )
    return sampler, loader


def _build_val_loader(
    args: argparse.Namespace,
    context: DistributedContext,
) -> tuple[TripletVideoDataset | None, DataLoader | None]:
    if not args.val_manifest:
        return None, None
    crop_size = args.crop_size if args.val_crop_size is None else args.val_crop_size
    dataset = TripletVideoDataset(
        args.val_manifest,
        split=args.val_split,
        training=False,
        clip_length=args.clip_length,
        crop_size=crop_size,
        scale=args.scale,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        drop_short_sequences=args.drop_short_validation,
    )
    sample_limit = min(len(dataset), args.validation_batches * args.batch_size)
    limited = Subset(dataset, range(sample_limit))
    sampler = DistributedEvalSampler(
        limited,
        rank=context.rank,
        world_size=context.world_size,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed + 9000001 + context.rank)
    loader = DataLoader(
        limited,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=args.pin_memory,
        persistent_workers=False,
        generator=generator,
    )
    return dataset, loader


def _run_fingerprint(
    args: argparse.Namespace,
    *,
    context: DistributedContext,
    base_checkpoint: Path,
    dtype: torch.dtype,
    dataset_length: int,
    batches_per_rank_epoch: int,
) -> dict[str, object]:
    effective_batch = global_effective_batch_size(
        world_size=context.world_size,
        local_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    return {
        "trainer": "stage3_reconstruction_ddp_v1",
        "base_checkpoint": str(base_checkpoint),
        "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
        "val_manifests": [str(path.expanduser().resolve()) for path in (args.val_manifest or [])],
        "path_root": str(args.path_root.expanduser().resolve()),
        "split": args.split,
        "val_split": args.val_split,
        "clip_length": int(args.clip_length),
        "crop_size": int(args.crop_size),
        "val_crop_size": int(args.crop_size if args.val_crop_size is None else args.val_crop_size),
        "scale": int(args.scale),
        "world_size": int(context.world_size),
        "local_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "global_effective_batch_size": int(effective_batch),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "horizontal_flip_probability": float(args.horizontal_flip_probability),
        "vertical_flip_probability": float(args.vertical_flip_probability),
        "drop_short_validation": bool(args.drop_short_validation),
        "dataset_length": int(dataset_length),
        "batches_per_rank_epoch": int(batches_per_rank_epoch),
        "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
        "attention_backend": args.attention_backend,
        "train_scope": args.train_scope,
        "pixel_loss_weight": float(args.pixel_loss_weight),
        "temporal_loss_weight": float(args.temporal_loss_weight),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "optimizer_eps": float(args.optimizer_eps),
        "max_grad_norm": float(args.max_grad_norm),
        "validate_every": int(args.validate_every),
        "validation_batches": int(args.validation_batches),
    }


def _validate_resume_config(saved: Mapping[str, object], current: Mapping[str, object]) -> None:
    if dict(saved) == dict(current):
        return
    differences = [
        f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]
    raise ValueError("DDP resume configuration differs:\n  " + "\n  ".join(differences[:32]))


def _save_rank_rng(checkpoint_dir: Path, context: DistributedContext) -> None:
    path = checkpoint_dir / f"rng_rank_{context.rank:05d}.pt"
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(capture_rng_state(), temporary)
    temporary.replace(path)


def _load_rank_rng(checkpoint_dir: Path, context: DistributedContext):
    state_path = checkpoint_dir / DISTRIBUTED_STATE_FILENAME
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing DDP state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("world_size", -1)) != context.world_size:
        raise ValueError(
            "Exact DDP resume requires the same world size: "
            f"checkpoint={state.get('world_size')}, current={context.world_size}"
        )
    path = checkpoint_dir / f"rng_rank_{context.rank:05d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-rank RNG state: {path}")
    return _load_torch(path)


def _checkpoint(
    *,
    run_dir: Path,
    closure: SwiftVRTrainingForward,
    optimizer: torch.optim.Optimizer,
    scaler,
    cursor: TrainingCursor,
    run_config: Mapping[str, object],
    last_record: Mapping[str, object],
    context: DistributedContext,
) -> Path:
    checkpoint_dir = run_dir / "checkpoints" / f"step_{cursor.global_step:08d}"
    exists = [False]
    if context.is_main:
        exists[0] = checkpoint_dir.exists() and any(checkpoint_dir.iterdir())
    dist.broadcast_object_list(exists, src=0)
    if exists[0]:
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_dir}")

    if context.is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_delta_checkpoint(
            checkpoint_dir,
            closure,
            optimizer,
            step=cursor.global_step,
            grad_scaler=scaler,
            metadata={
                "cursor": {
                    "global_step": cursor.global_step,
                    "epoch": cursor.epoch,
                    "batch_in_epoch": cursor.batch_in_epoch,
                },
                "last_loss": last_record.get("loss"),
                "last_pixel_l1": last_record.get("pixel_l1"),
                "last_temporal_mse": last_record.get("temporal_mse"),
                "runtime_dtype": run_config["runtime_dtype"],
                "train_scope": run_config["train_scope"],
                "world_size": context.world_size,
            },
        )
        save_trainer_state(
            checkpoint_dir,
            cursor=cursor,
            config=run_config,
            rng_state=capture_rng_state(),
        )
    dist.barrier()

    _save_rank_rng(checkpoint_dir, context)
    dist.barrier()

    if context.is_main:
        _write_json(
            checkpoint_dir / DISTRIBUTED_STATE_FILENAME,
            {
                "format_version": 1,
                "world_size": context.world_size,
                "rng_files": [
                    f"rng_rank_{rank:05d}.pt" for rank in range(context.world_size)
                ],
            },
        )
        write_latest_checkpoint(run_dir, checkpoint_dir)
    dist.barrier()
    return checkpoint_dir


def _all_reduce_training_values(
    sums: Mapping[str, float],
    *,
    denominator: int,
    gradient_norm: float,
    step_seconds: float,
    device: torch.device,
) -> tuple[dict[str, float], float, float]:
    tensor = torch.tensor(
        [sums["loss"], sums["pixel_l1"], sums["temporal_mse"]],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    averages = {
        "loss": float(tensor[0].item()) / denominator,
        "pixel_l1": float(tensor[1].item()) / denominator,
        "temporal_mse": float(tensor[2].item()) / denominator,
    }
    maxima = torch.tensor([gradient_norm, step_seconds], device=device, dtype=torch.float64)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    return averages, float(maxima[0].item()), float(maxima[1].item())


@torch.no_grad()
def _validate(
    ddp_model: DistributedDataParallel,
    closure: SwiftVRTrainingForward,
    loader: DataLoader,
    *,
    device: torch.device,
    dtype: torch.dtype,
    pixel_weight: float,
    temporal_weight: float,
    reae_frozen: bool,
) -> dict[str, float | int]:
    ddp_model.eval()
    metrics = VideoMetricAccumulator()
    sums = {"loss": 0.0, "pixel_l1": 0.0, "temporal_mse": 0.0}
    processed = 0
    autocast_enabled = dtype in {torch.float16, torch.bfloat16}
    try:
        for batch_cpu in loader:
            batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
            with torch.autocast(
                device_type="cuda",
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                output = ddp_model(batch)
                objective = stage3_reconstruction_objective(
                    output,
                    pixel_weight=pixel_weight,
                    temporal_weight=temporal_weight,
                )
            for key in sums:
                sums[key] += float(objective[key].detach().float().item())
            prediction = output.get("prediction_clamped")
            target = output.get("target")
            if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise TypeError("Validation output is missing prediction/target tensors")
            metrics.update(prediction, target, clamp=True)
            processed += 1
    finally:
        ddp_model.train()
        if reae_frozen:
            closure.reae.eval()

    packed = torch.tensor(
        [
            metrics.sum_abs,
            metrics.sum_squared,
            float(metrics.elements),
            metrics.sum_ssim,
            float(metrics.ssim_frames),
            float(metrics.batches),
            sums["loss"],
            sums["pixel_l1"],
            sums["temporal_mse"],
            float(processed),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    (
        sum_abs,
        sum_squared,
        elements,
        sum_ssim,
        ssim_frames,
        batches,
        loss_sum,
        pixel_sum,
        temporal_sum,
        processed_total,
    ) = [float(value) for value in packed.tolist()]
    if elements <= 0 or ssim_frames <= 0 or processed_total <= 0:
        raise RuntimeError("Distributed validation produced no samples")
    mae = sum_abs / elements
    mse = sum_squared / elements
    return {
        "mae": mae,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "psnr": math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse),
        "ssim": sum_ssim / ssim_frames,
        "batches": int(batches),
        "frames": int(ssim_frames),
        "elements": int(elements),
        "loss": loss_sum / processed_total,
        "pixel_l1": pixel_sum / processed_total,
        "temporal_mse": temporal_sum / processed_total,
    }


def run(args: argparse.Namespace) -> dict[str, object] | None:
    context, device = _init_distributed()
    try:
        if args.num_workers != 0:
            raise ValueError("Exact DDP resume currently requires --num-workers 0")
        if args.resume is not None and args.init_from_single_checkpoint is not None:
            raise ValueError("Use either --resume or --init-from-single-checkpoint, not both")
        if args.max_steps <= 0 or args.gradient_accumulation_steps <= 0:
            raise ValueError("max-steps and gradient-accumulation-steps must be positive")
        if args.save_every <= 0 or args.log_every <= 0:
            raise ValueError("save-every and log-every must be positive")
        if args.validate_every > 0 and not args.val_manifest:
            raise ValueError("--validate-every > 0 requires --val-manifest")
        if args.validation_batches <= 0:
            raise ValueError("validation-batches must be positive")

        effective_batch = global_effective_batch_size(
            world_size=context.world_size,
            local_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        if (
            args.expected_global_batch_size is not None
            and effective_batch != args.expected_global_batch_size
        ):
            raise ValueError(
                f"Global effective batch is {effective_batch}, expected "
                f"{args.expected_global_batch_size}"
            )

        run_dir = args.output_dir.expanduser().resolve()
        train_log = run_dir / "train_log.jsonl"
        val_log = run_dir / "val_log.jsonl"
        config_path = run_dir / "run_config.json"
        if context.is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        cursor = TrainingCursor()
        resume_checkpoint: Path | None = None
        migration_checkpoint: Path | None = None
        saved_config = None
        pending_rng = None
        migration_source_step = None

        if args.resume is not None:
            resolved = (
                resolve_resume_checkpoint(args.resume, run_dir=run_dir)
                if context.is_main
                else None
            )
            resume_checkpoint = _broadcast_path(resolved, context)
            trainer_state = load_trainer_state(resume_checkpoint)
            cursor = trainer_state["cursor"]
            saved_config = trainer_state["config"]
            pending_rng = _load_rank_rng(resume_checkpoint, context)
            if not isinstance(cursor, TrainingCursor):
                raise TypeError("Invalid DDP training cursor")
        elif args.init_from_single_checkpoint is not None:
            migration_checkpoint = args.init_from_single_checkpoint.expanduser().resolve()
            if not migration_checkpoint.is_dir():
                raise FileNotFoundError(migration_checkpoint)
            source_state = load_trainer_state(migration_checkpoint)
            source_cursor = source_state["cursor"]
            if not isinstance(source_cursor, TrainingCursor):
                raise TypeError("Invalid single-GPU migration cursor")
            migration_source_step = source_cursor.global_step
            cursor = TrainingCursor(global_step=source_cursor.global_step)
            if context.is_main and (
                train_log.exists() or (run_dir / "latest.json").exists()
            ):
                raise FileExistsError(
                    "Migration output directory already contains a run; choose a new directory"
                )
        else:
            if context.is_main and (
                train_log.exists() or (run_dir / "latest.json").exists()
            ):
                raise FileExistsError(
                    "Output directory already contains a run; use --resume latest"
                )
        dist.barrier()

        base_checkpoint = args.checkpoint.expanduser().resolve()
        folded_config = validate_folded_checkpoint(
            base_checkpoint,
            reae_filename=args.reae_filename,
            transformer_subfolder=args.transformer_subfolder,
        )
        dtype = resolve_runtime_dtype(
            args.dtype,
            folded_config,
            device,
            allow_mismatch=args.allow_dtype_mismatch,
        )

        seed_everything(args.seed + context.rank)
        train_dataset = _build_train_dataset(args)
        _, first_loader = _build_train_loader(
            train_dataset, args, context, cursor.epoch
        )
        if len(first_loader) < args.gradient_accumulation_steps:
            raise RuntimeError(
                "Per-rank epoch has fewer batches than local gradient accumulation"
            )
        run_config = _run_fingerprint(
            args,
            context=context,
            base_checkpoint=base_checkpoint,
            dtype=dtype,
            dataset_length=len(train_dataset),
            batches_per_rank_epoch=len(first_loader),
        )
        if saved_config is not None:
            if not isinstance(saved_config, Mapping):
                raise TypeError("Saved DDP run configuration must be a mapping")
            _validate_resume_config(saved_config, run_config)
        elif context.is_main:
            if config_path.exists():
                existing = json.loads(config_path.read_text(encoding="utf-8"))
                if not isinstance(existing, Mapping):
                    raise ValueError("Invalid run_config.json")
                _validate_resume_config(existing, run_config)
            else:
                _write_json(config_path, run_config)
                if migration_checkpoint is not None:
                    _write_json(
                        run_dir / "migration.json",
                        {
                            "source_checkpoint": str(migration_checkpoint),
                            "source_step": migration_source_step,
                            "exact_data_order": False,
                            "reason": "single-GPU to DDP topology migration",
                        },
                    )
        dist.barrier()

        _, val_loader = _build_val_loader(args, context)

        reae = ReAE(str(base_checkpoint / args.reae_filename))
        transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
            str(base_checkpoint),
            subfolder=args.transformer_subfolder,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        parameter_counts = configure_train_scope(reae, transformer, args.train_scope)
        reae.to(device=device, dtype=dtype)
        transformer.to(device=device, dtype=dtype)
        closure = SwiftVRTrainingForward(
            reae,
            transformer,
            latent_loss_weight=0.0,
            training_safe_transformer=True,
            prepare_transformer=True,
            attention_backend=args.attention_backend,
        )
        closure.train()
        reae.eval()

        optimizer_precision = cast_trainable_parameters(closure, dtype=torch.float32)
        optimizer = build_fp32_adamw(
            closure,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.optimizer_eps,
        )
        scaler = build_grad_scaler(device, dtype)

        source_checkpoint = resume_checkpoint or migration_checkpoint
        if source_checkpoint is not None:
            metadata = load_delta_checkpoint(
                source_checkpoint,
                closure,
                optimizer,
                strict=True,
                map_location="cpu",
                grad_scaler=scaler,
            )
            if int(metadata["step"]) != cursor.global_step:
                raise ValueError(
                    "Delta checkpoint and selected start step disagree: "
                    f"delta={metadata['step']}, cursor={cursor.global_step}"
                )

        ddp_model = DistributedDataParallel(
            closure,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

        best_psnr = -float("inf")
        best_path = run_dir / "best.json"
        if best_path.is_file():
            best_value = json.loads(best_path.read_text(encoding="utf-8"))
            best_psnr = float(best_value.get("psnr", best_psnr))

        if cursor.global_step >= args.max_steps:
            if context.is_main:
                return {
                    "status": "ALREADY_COMPLETE",
                    "global_step": cursor.global_step,
                    "max_steps": args.max_steps,
                    "run_dir": str(run_dir),
                }
            return None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        restored_rng = pending_rng is None
        last_record: dict[str, object] = {}
        last_validation: dict[str, object] | None = None
        last_checkpoint = resume_checkpoint

        if args.validate_at_start and val_loader is not None:
            training_rng = capture_rng_state()
            baseline = _validate(
                ddp_model,
                closure,
                val_loader,
                device=device,
                dtype=dtype,
                pixel_weight=args.pixel_loss_weight,
                temporal_weight=args.temporal_loss_weight,
                reae_frozen=True,
            )
            restore_rng_state(training_rng)
            last_validation = {"global_step": cursor.global_step, **baseline}
            best_psnr = float(last_validation["psnr"])
            if context.is_main:
                append_jsonl(val_log, last_validation)
                print(
                    f"validation start step={cursor.global_step} "
                    f"psnr={last_validation['psnr']:.4f} "
                    f"ssim={last_validation['ssim']:.6f}",
                    flush=True,
                )

        while cursor.global_step < args.max_steps:
            _, loader = _build_train_loader(
                train_dataset, args, context, cursor.epoch
            )
            batches_per_epoch = len(loader)
            if batches_per_epoch != int(run_config["batches_per_rank_epoch"]):
                raise RuntimeError("Per-rank DataLoader length changed during the run")
            remaining = batches_per_epoch - cursor.batch_in_epoch
            if remaining < args.gradient_accumulation_steps:
                cursor = advance_cursor_batches(
                    cursor,
                    consumed_batches=remaining,
                    batches_per_epoch=batches_per_epoch,
                    optimizer_steps=0,
                )
                continue

            iterator = iter(loader)
            skip_batches(iterator, cursor.batch_in_epoch)
            if not restored_rng:
                if not isinstance(pending_rng, Mapping):
                    raise TypeError("Saved per-rank RNG state must be a mapping")
                restore_rng_state(pending_rng)
                restored_rng = True
                pending_rng = None

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            sums = {"loss": 0.0, "pixel_l1": 0.0, "temporal_mse": 0.0}
            autocast_enabled = dtype in {torch.float16, torch.bfloat16}

            for microbatch_index in range(args.gradient_accumulation_steps):
                batch_cpu = next(iterator)
                batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                synchronize = microbatch_index + 1 == args.gradient_accumulation_steps
                synchronization_context = nullcontext() if synchronize else ddp_model.no_sync()
                with synchronization_context:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=dtype if autocast_enabled else torch.float32,
                        enabled=autocast_enabled,
                    ):
                        output = ddp_model(batch)
                        objective = stage3_reconstruction_objective(
                            output,
                            pixel_weight=args.pixel_loss_weight,
                            temporal_weight=args.temporal_loss_weight,
                        )
                        scaled_loss = (
                            objective["loss"] / args.gradient_accumulation_steps
                        )
                    if not torch.isfinite(objective["loss"].detach()).item():
                        raise FloatingPointError(
                            f"Non-finite loss before step {cursor.global_step + 1}"
                        )
                    scaler.scale(scaled_loss).backward()
                for key in sums:
                    sums[key] += float(objective[key].detach().float().item())

            scaler.unscale_(optimizer)
            gradients = gradient_summary(closure.named_parameters())
            if int(gradients["gradient_tensors"]) == 0:
                raise RuntimeError("Backward produced no trainable gradients")
            if int(gradients["nonfinite_elements"]) != 0:
                raise FloatingPointError("Backward produced non-finite gradients")
            if int(gradients["missing_gradient_count"]) != 0:
                raise RuntimeError(
                    f"Missing gradients: {gradients['missing_gradient_examples']}"
                )
            grad_norm = float(gradients["global_l2"])
            if args.max_grad_norm > 0:
                clipped = torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in trainable_named_parameters(closure)],
                    max_norm=float(args.max_grad_norm),
                    error_if_nonfinite=True,
                )
                grad_norm = float(clipped.detach().float().item())

            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            overflow = torch.tensor(
                [1 if scale_after < scale_before else 0],
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(overflow, op=dist.ReduceOp.MAX)
            if int(overflow.item()) != 0:
                raise FloatingPointError("At least one rank skipped a step due to overflow")

            cursor = advance_cursor_batches(
                cursor,
                consumed_batches=args.gradient_accumulation_steps,
                batches_per_epoch=batches_per_epoch,
                optimizer_steps=1,
            )
            torch.cuda.synchronize(device)
            local_seconds = time.perf_counter() - step_started
            averages, grad_norm, step_seconds = _all_reduce_training_values(
                sums,
                denominator=(
                    args.gradient_accumulation_steps * context.world_size
                ),
                gradient_norm=grad_norm,
                step_seconds=local_seconds,
                device=device,
            )
            last_record = {
                "global_step": cursor.global_step,
                "epoch": cursor.epoch,
                "batch_in_epoch": cursor.batch_in_epoch,
                "world_size": context.world_size,
                "local_batch_size": args.batch_size,
                "local_microbatches": args.gradient_accumulation_steps,
                "global_effective_batch_size": effective_batch,
                **averages,
                "gradient_norm": grad_norm,
                "grad_scaler_scale": scale_after,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "step_seconds": step_seconds,
                "peak_allocated_gb_per_rank": (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                ),
            }
            if context.is_main:
                append_jsonl(train_log, last_record)
                if cursor.global_step % args.log_every == 0:
                    print(
                        f"step={cursor.global_step} loss={averages['loss']:.8f} "
                        f"pixel={averages['pixel_l1']:.8f} "
                        f"temp={averages['temporal_mse']:.8f} "
                        f"grad={grad_norm:.6g} time={step_seconds:.3f}s",
                        flush=True,
                    )

            validation_due = (
                val_loader is not None
                and args.validate_every > 0
                and (
                    cursor.global_step % args.validate_every == 0
                    or cursor.global_step == args.max_steps
                )
            )
            improved = False
            if validation_due:
                training_rng = capture_rng_state()
                last_validation = _validate(
                    ddp_model,
                    closure,
                    val_loader,
                    device=device,
                    dtype=dtype,
                    pixel_weight=args.pixel_loss_weight,
                    temporal_weight=args.temporal_loss_weight,
                    reae_frozen=True,
                )
                restore_rng_state(training_rng)
                last_validation = {"global_step": cursor.global_step, **last_validation}
                improved = float(last_validation["psnr"]) > best_psnr
                if improved:
                    best_psnr = float(last_validation["psnr"])
                if context.is_main:
                    append_jsonl(val_log, last_validation)
                    print(
                        f"validation step={cursor.global_step} "
                        f"psnr={last_validation['psnr']:.4f} "
                        f"ssim={last_validation['ssim']:.6f} "
                        f"mae={last_validation['mae']:.6f}",
                        flush=True,
                    )

            should_save = (
                cursor.global_step % args.save_every == 0
                or cursor.global_step == args.max_steps
                or validation_due
            )
            if should_save:
                if context.is_main:
                    state_summary = optimizer_state_summary(optimizer)
                    if int(state_summary["nonfinite_elements"]) != 0:
                        raise FloatingPointError("Optimizer state contains non-finite values")
                    if set(state_summary["dtype_counts"]) != {"float32"}:
                        raise RuntimeError(
                            f"Expected FP32 AdamW state, got {state_summary['dtype_counts']}"
                        )
                dist.barrier()
                last_checkpoint = _checkpoint(
                    run_dir=run_dir,
                    closure=closure,
                    optimizer=optimizer,
                    scaler=scaler,
                    cursor=cursor,
                    run_config=run_config,
                    last_record=last_record,
                    context=context,
                )
                if context.is_main:
                    if improved and last_validation is not None:
                        _write_pointer(
                            run_dir,
                            "best.json",
                            last_checkpoint,
                            global_step=cursor.global_step,
                            psnr=float(last_validation["psnr"]),
                            ssim=float(last_validation["ssim"]),
                        )
                    print(f"saved checkpoint: {last_checkpoint}", flush=True)

        result = {
            "status": "PASS",
            "global_step": cursor.global_step,
            "epoch": cursor.epoch,
            "batch_in_epoch": cursor.batch_in_epoch,
            "max_steps": args.max_steps,
            "run_dir": str(run_dir),
            "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
            "elapsed_seconds": time.perf_counter() - started,
            "device_name": torch.cuda.get_device_name(device),
            "world_size": context.world_size,
            "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
            "optimizer_parameter_dtype": optimizer_precision["target_dtype"],
            "train_scope": args.train_scope,
            "local_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_effective_batch_size": effective_batch,
            "parameters": parameter_counts,
            "last_record": last_record,
            "last_validation": last_validation,
            "best_psnr": None if best_psnr == -float("inf") else best_psnr,
            "migration_source": (
                str(migration_checkpoint) if migration_checkpoint is not None else None
            ),
        }
        if context.is_main:
            _write_json(run_dir / "summary.json", result)
            return result
        return None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def print_result(result: Mapping[str, object]) -> None:
    print("\n========== SwiftVR Stage-3 DDP reconstruction ==========")
    print("status                 :", result["status"])
    print("global step            :", result["global_step"], "/", result["max_steps"])
    if result["status"] == "PASS":
        print("cursor                 :", result["epoch"], result["batch_in_epoch"])
        print("device                 :", result["device_name"])
        print("world size             :", result["world_size"])
        print("runtime dtype          :", result["runtime_dtype"])
        print("optimizer dtype        :", result["optimizer_parameter_dtype"])
        print("train scope            :", result["train_scope"])
        print("local batch            :", result["local_batch_size"])
        print("local accumulation     :", result["gradient_accumulation_steps"])
        print("global effective batch :", result["global_effective_batch_size"])
        counts = result["parameters"]
        assert isinstance(counts, Mapping)
        print(
            "trainable params       :",
            _format_count(int(counts["reae_trainable"])),
            "ReAE +",
            _format_count(int(counts["transformer_trainable"])),
            "DiT",
        )
        print("best PSNR              :", result["best_psnr"])
        print("last checkpoint        :", result["last_checkpoint"])
        print("elapsed seconds        :", f"{float(result['elapsed_seconds']):.3f}")
    print("========================================================\n")


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    if result is not None:
        print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
