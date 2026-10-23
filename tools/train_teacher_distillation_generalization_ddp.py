#!/usr/bin/env python3
"""Resumable multi-GPU generalization training for SwiftVR teacher distillation.

This production D0 trainer keeps the validated velocity/GT-guard/BF16 behavior
from ``train_teacher_distillation_ddp.py`` and adds arbitrary deterministic cache
subsets, independent train/validation checks, exact same-world-size mid-epoch
resume for ``num_workers=0``, per-rank RNG checkpoints, and stable DDP data order.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

try:
    import train_teacher_distillation_ddp as gate
except ModuleNotFoundError:
    from tools import train_teacher_distillation_ddp as gate

from swiftvr.training import (
    TrainingCursor,
    advance_cursor_batches,
    capture_rng_state,
    load_delta_checkpoint,
    load_trainer_state,
    resolve_resume_checkpoint,
    restore_rng_state,
    save_trainer_state,
    skip_batches,
)
from swiftvr.training.distillation import (
    DeterministicTripletViewDataset,
    SwiftVRVelocityDistillationForward,
    TeacherVelocityCache,
    velocity_distillation_objective,
)
from swiftvr.training.distillation_generalization import (
    build_cache_backed_subset,
    cache_overlap_report,
    cache_selected_indices,
    selected_indices_sha256,
    validate_resume_fingerprint,
)
from swiftvr.training.reference import sha256_file

DISTRIBUTED_STATE_FILENAME = "distributed_state.json"


def build_parser() -> argparse.ArgumentParser:
    parser = gate.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--resume",
        default=None,
        help="Exact checkpoint directory or 'latest' under --output-dir.",
    )
    parser.add_argument(
        "--allow-train-val-overlap",
        action="store_true",
        help="Permit overlapping record_uid values only for deliberate diagnostics.",
    )
    return parser


def _broadcast_path(path: Path | None, rank: int) -> Path:
    value = [str(path) if rank == 0 and path is not None else None]
    dist.broadcast_object_list(value, src=0)
    if not isinstance(value[0], str):
        raise RuntimeError("Rank 0 did not broadcast a checkpoint path")
    return Path(value[0]).expanduser().resolve()


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_rank_rng(checkpoint: Path, rank: int) -> None:
    path = checkpoint / f"rng_rank_{rank:05d}.pt"
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(capture_rng_state(), temporary)
    temporary.replace(path)


def _load_rank_rng(checkpoint: Path, rank: int, world_size: int):
    state_path = checkpoint / DISTRIBUTED_STATE_FILENAME
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing distributed state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("world_size", -1)) != int(world_size):
        raise ValueError(
            "Exact resume requires the same world size: "
            f"checkpoint={state.get('world_size')}, current={world_size}"
        )
    path = checkpoint / f"rng_rank_{rank:05d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing rank RNG state: {path}")
    return _load_torch(path)


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
    base = gate.TripletVideoDataset(
        manifests,
        split=split,
        training=True,
        clip_length=clip_length,
        crop_size=crop_size,
        scale=scale,
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
    return DataLoader(
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
    train_indices = cache_selected_indices(train_cache.metadata)
    val_indices = () if val_cache is None else cache_selected_indices(val_cache.metadata)
    return {
        "trainer": "teacher_distillation_generalization_ddp_v1",
        "base_checkpoint": str(base_checkpoint),
        "teacher_cache": str(train_cache.root),
        "teacher_cache_metadata_sha256": sha256_file(train_cache.root / "metadata.json"),
        "teacher_selected_indices_sha256": selected_indices_sha256(train_indices),
        "val_teacher_cache": None if val_cache is None else str(val_cache.root),
        "val_teacher_cache_metadata_sha256": (
            None if val_cache is None else sha256_file(val_cache.root / "metadata.json")
        ),
        "val_selected_indices_sha256": (
            None if val_cache is None else selected_indices_sha256(val_indices)
        ),
        "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
        "val_manifests": [
            str(path.expanduser().resolve()) for path in (args.val_manifest or [])
        ],
        "split": args.split,
        "val_split": args.val_split,
        "clip_length": int(args.clip_length),
        "crop_size": int(args.crop_size),
        "val_crop_size": int(
            args.crop_size if args.val_crop_size is None else args.val_crop_size
        ),
        "scale": int(args.scale),
        "views_per_record": int(args.views_per_record),
        "view_seed": int(args.view_seed),
        "val_views_per_record": int(args.val_views_per_record),
        "val_view_seed": int(args.val_view_seed),
        "horizontal_flip_probability": float(args.horizontal_flip_probability),
        "vertical_flip_probability": float(args.vertical_flip_probability),
        "train_dataset_length": int(train_dataset_length),
        "batches_per_rank_epoch": int(batches_per_rank_epoch),
        "world_size": int(world_size),
        "local_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "global_effective_batch_size": int(
            world_size * args.batch_size * args.gradient_accumulation_steps
        ),
        "seed": int(args.seed),
        "runtime_dtype": gate._CANONICAL_DTYPE_NAME[runtime_dtype],
        "attention_backend": args.attention_backend,
        "velocity_mse_weight": float(args.velocity_mse_weight),
        "velocity_cosine_weight": float(args.velocity_cosine_weight),
        "output_l1_weight": float(args.output_l1_weight),
        "output_temporal_weight": float(args.output_temporal_weight),
        "gt_loss_mode": args.gt_loss_mode,
        "gt_pixel_weight": float(args.gt_pixel_weight),
        "gt_temporal_weight": float(args.gt_temporal_weight),
        "gt_loss_every": int(args.gt_loss_every),
        "loss_epsilon": float(args.loss_epsilon),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "optimizer_eps": float(args.optimizer_eps),
        "max_grad_norm": float(args.max_grad_norm),
    }


def _checkpoint(
    *,
    run_dir: Path,
    closure: SwiftVRVelocityDistillationForward,
    optimizer: torch.optim.Optimizer,
    scaler,
    cursor: TrainingCursor,
    fingerprint: Mapping[str, object],
    last_record: Mapping[str, object],
    runtime_dtype: torch.dtype,
    rank: int,
    world_size: int,
) -> Path:
    checkpoint = run_dir / "checkpoints" / f"step_{cursor.global_step:08d}"
    exists = [False]
    if rank == 0:
        exists[0] = checkpoint.exists()
    dist.broadcast_object_list(exists, src=0)
    if exists[0]:
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")

    if rank == 0:
        checkpoint.mkdir(parents=True, exist_ok=True)
        gate.save_delta_checkpoint(
            checkpoint,
            closure,
            optimizer,
            step=cursor.global_step,
            grad_scaler=scaler,
            metadata={
                "trainer": "teacher_distillation_generalization_ddp_v1",
                "cursor": {
                    "global_step": cursor.global_step,
                    "epoch": cursor.epoch,
                    "batch_in_epoch": cursor.batch_in_epoch,
                },
                "last_loss": last_record.get("loss"),
                "last_velocity_relative_l2": last_record.get("velocity_relative_l2"),
                "last_velocity_cosine": last_record.get("velocity_cosine"),
                "runtime_dtype": gate._CANONICAL_DTYPE_NAME[runtime_dtype],
                "grad_scaler_enabled": bool(scaler.is_enabled()),
                "world_size": world_size,
            },
        )
        save_trainer_state(
            checkpoint,
            cursor=cursor,
            config=fingerprint,
            rng_state=capture_rng_state(),
        )
    dist.barrier()

    _save_rank_rng(checkpoint, rank)
    dist.barrier()

    if rank == 0:
        gate.write_json(
            checkpoint / DISTRIBUTED_STATE_FILENAME,
            {
                "format_version": 1,
                "world_size": world_size,
                "rng_files": [
                    f"rng_rank_{index:05d}.pt" for index in range(world_size)
                ],
            },
        )
        gate.write_latest_checkpoint(run_dir, checkpoint)
    dist.barrier()
    return checkpoint


def _reduce_step(
    sums: Mapping[str, float],
    *,
    denominator: int,
    gradient_norm: float,
    step_seconds: float,
    device: torch.device,
    world_size: int,
) -> tuple[dict[str, float], float, float]:
    keys = tuple(sums)
    packed = torch.tensor(
        [sums[key] for key in keys] + [gradient_norm, step_seconds],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    averages = {
        key: float(packed[index].item()) / denominator
        for index, key in enumerate(keys)
    }
    return (
        averages,
        float(packed[-2].item()) / world_size,
        float(packed[-1].item()) / world_size,
    )


def main() -> int:
    args = build_parser().parse_args()
    visual_frame_indices = gate._validate_arguments(args)
    rank, local_rank, world_size, device = gate.init_distributed()
    writer = None
    try:
        effective_batch = world_size * args.batch_size * args.gradient_accumulation_steps
        if (
            args.expected_global_batch_size is not None
            and effective_batch != args.expected_global_batch_size
        ):
            raise ValueError(
                f"Global effective batch={effective_batch}, "
                f"expected={args.expected_global_batch_size}"
            )

        run_dir = args.output_dir.expanduser().resolve()
        train_log = run_dir / "train_log.jsonl"
        val_log = run_dir / "val_log.jsonl"
        cursor = TrainingCursor()
        resume_checkpoint: Path | None = None
        saved_fingerprint = None
        pending_rng = None

        if args.resume is not None:
            resolved = (
                resolve_resume_checkpoint(args.resume, run_dir=run_dir)
                if rank == 0
                else None
            )
            resume_checkpoint = _broadcast_path(resolved, rank)
            state = load_trainer_state(resume_checkpoint)
            cursor = state["cursor"]
            saved_fingerprint = state["config"]
            if not isinstance(cursor, TrainingCursor):
                raise TypeError("Invalid saved TrainingCursor")
            if not isinstance(saved_fingerprint, Mapping):
                raise TypeError("Invalid saved run fingerprint")
            pending_rng = _load_rank_rng(resume_checkpoint, rank, world_size)
        else:
            if rank == 0:
                if train_log.exists() or (run_dir / "latest.json").exists():
                    raise FileExistsError(
                        "Output directory already contains a run; use --resume latest"
                    )
                run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        base_checkpoint = args.checkpoint.expanduser().resolve()
        folded_config = gate.validate_folded_checkpoint(
            base_checkpoint,
            reae_filename=args.reae_filename,
            transformer_subfolder=args.transformer_subfolder,
        )
        dtype = gate.resolve_runtime_dtype(
            args.dtype,
            folded_config,
            device,
            allow_mismatch=args.allow_dtype_mismatch,
        )
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"{torch.cuda.get_device_name(device)} does not support BF16"
            )
        gate.seed_everything(args.seed + rank)

        train_cache = TeacherVelocityCache(args.teacher_cache)
        train_dataset = _build_cached_dataset(
            args.manifest,
            train_cache,
            split=args.split,
            path_root=args.path_root,
            clip_length=args.clip_length,
            crop_size=args.crop_size,
            scale=args.scale,
            views_per_record=args.views_per_record,
            view_seed=args.view_seed,
            hflip=args.horizontal_flip_probability,
            vflip=args.vertical_flip_probability,
            verify_paths=args.verify_paths,
        )
        val_cache = None
        val_loader = None
        if args.val_manifest and args.val_teacher_cache:
            val_cache = TeacherVelocityCache(args.val_teacher_cache)
            overlap = cache_overlap_report(train_cache.metadata, val_cache.metadata)
            if int(overlap["overlap_records"]) > 0 and not args.allow_train_val_overlap:
                raise ValueError(
                    "Train and validation caches share "
                    f"{overlap['overlap_records']} record_uid values"
                )
            for key in (
                "reference_checkpoint_match",
                "prompt_embedding_sha256_match",
                "reae_sha256_match",
                "timestep_match",
            ):
                if not bool(overlap[key]):
                    raise ValueError(f"Train/validation cache mismatch: {key}")

            val_crop = args.crop_size if args.val_crop_size is None else args.val_crop_size
            val_dataset = _build_cached_dataset(
                args.val_manifest,
                val_cache,
                split=args.val_split,
                path_root=args.path_root,
                clip_length=args.clip_length,
                crop_size=val_crop,
                scale=args.scale,
                views_per_record=args.val_views_per_record,
                view_seed=args.val_view_seed,
                hflip=0.0,
                vflip=0.0,
                verify_paths=args.verify_paths,
            )
            if rank == 0:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                    pin_memory=args.pin_memory,
                )

        first_loader = _make_train_loader(
            train_dataset,
            rank=rank,
            world_size=world_size,
            epoch=cursor.epoch,
            args=args,
        )
        if len(first_loader) < args.gradient_accumulation_steps:
            raise RuntimeError("Per-rank epoch is shorter than gradient accumulation")
        fingerprint = _fingerprint(
            args,
            base_checkpoint=base_checkpoint,
            train_cache=train_cache,
            val_cache=val_cache,
            runtime_dtype=dtype,
            world_size=world_size,
            train_dataset_length=len(train_dataset),
            batches_per_rank_epoch=len(first_loader),
        )
        if saved_fingerprint is not None:
            validate_resume_fingerprint(saved_fingerprint, fingerprint)

        reae = gate.ReAE(str(base_checkpoint / args.reae_filename))
        transformer = gate.WanTransformer3DModelPromptFreeNoTime.from_pretrained(
            str(base_checkpoint),
            subfolder=args.transformer_subfolder,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        gate.configure_train_scope(reae, transformer, "adapter")
        reae.to(device=device, dtype=dtype).eval()
        transformer.to(device=device, dtype=dtype)
        closure = SwiftVRVelocityDistillationForward(
            reae,
            transformer,
            attention_backend=args.attention_backend,
        )
        closure.train()
        closure.reae.eval()
        gate.cast_trainable_parameters(closure, dtype=torch.float32)
        optimizer = gate.build_fp32_adamw(
            closure,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.optimizer_eps,
        )
        scaler = gate.build_grad_scaler(device, dtype)

        if resume_checkpoint is not None:
            metadata = load_delta_checkpoint(
                resume_checkpoint,
                closure,
                optimizer,
                strict=True,
                map_location="cpu",
                grad_scaler=scaler,
            )
            if int(metadata["step"]) != cursor.global_step:
                raise ValueError(
                    "Delta checkpoint and trainer cursor disagree: "
                    f"{metadata['step']} vs {cursor.global_step}"
                )

        ddp_model = DistributedDataParallel(
            closure,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        writer = gate.create_writer(args, run_dir, rank)

        visualize_every = (
            args.validate_every if args.visualize_every is None else args.visualize_every
        )
        visuals_enabled = (
            not args.no_validation_visuals
            and args.visual_validation_samples > 0
            and visualize_every > 0
        )
        if rank == 0 and resume_checkpoint is None:
            gate.write_json(
                run_dir / "run_config.json",
                {
                    "fingerprint": fingerprint,
                    "max_steps": args.max_steps,
                    "save_every": args.save_every,
                    "validate_every": args.validate_every,
                    "visualize_every": visualize_every,
                    "visuals_enabled": visuals_enabled,
                    "exact_resume": True,
                },
            )
        dist.barrier()

        best_relative_l2 = float("inf")
        best_path = run_dir / "best.json"
        if best_path.is_file():
            best = json.loads(best_path.read_text(encoding="utf-8"))
            best_relative_l2 = float(best.get("velocity_relative_l2", best_relative_l2))

        if cursor.global_step >= args.max_steps:
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "status": "ALREADY_COMPLETE",
                            "global_step": cursor.global_step,
                            "max_steps": args.max_steps,
                            "run_dir": str(run_dir),
                        },
                        indent=2,
                    )
                )
            return 0

        def run_validation(step: int):
            nonlocal best_relative_l2
            assert val_loader is not None and val_cache is not None
            visual_due = visuals_enabled and (
                step == 0 or step % visualize_every == 0 or step == args.max_steps
            )
            validation, visual_samples = gate.validate_rank0(
                closure,
                val_loader,
                val_cache,
                device=device,
                dtype=dtype,
                args=args,
                visual_samples=args.visual_validation_samples if visual_due else 0,
            )
            gate.append_jsonl(val_log, {"global_step": step, **validation})
            gate._write_validation_scalars(writer, step, validation)
            if visual_due:
                report = gate.export_validation_visuals(
                    visual_samples,
                    output_root=run_dir,
                    step=step,
                    frame_indices=visual_frame_indices,
                    fps=args.visual_video_fps,
                    difference_scale=args.visual_difference_scale,
                    writer=writer,
                )
                if report["video_errors"]:
                    print(
                        "validation visual MP4 warnings: "
                        + json.dumps(report["video_errors"]),
                        flush=True,
                    )
            best_relative_l2 = min(
                best_relative_l2,
                float(validation["velocity_relative_l2"]),
            )
            return validation

        validation_configured = bool(args.val_manifest and args.val_teacher_cache)
        if args.validate_at_start and validation_configured and cursor.global_step == 0:
            dist.barrier()
            if rank == 0:
                training_rng = capture_rng_state()
                try:
                    baseline = run_validation(0)
                finally:
                    restore_rng_state(training_rng)
                print(
                    f"validation start rel_l2={baseline['velocity_relative_l2']:.6f} "
                    f"cos={baseline['velocity_cosine']:.6f} "
                    f"ref_psnr={baseline['student_teacher_psnr']:.4f}",
                    flush=True,
                )
            dist.barrier()

        restored_rng = pending_rng is None
        last_record: dict[str, object] = {}
        last_checkpoint = resume_checkpoint
        started = time.perf_counter()
        autocast_enabled = dtype in (torch.float16, torch.bfloat16)

        while cursor.global_step < args.max_steps:
            epoch = cursor.epoch
            loader = _make_train_loader(
                train_dataset,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                args=args,
            )
            if len(loader) != int(fingerprint["batches_per_rank_epoch"]):
                raise RuntimeError("Per-rank DataLoader length changed")
            iterator = iter(loader)
            skip_batches(iterator, cursor.batch_in_epoch)
            if not restored_rng:
                if not isinstance(pending_rng, Mapping):
                    raise TypeError("Saved rank RNG state must be a mapping")
                restore_rng_state(pending_rng)
                pending_rng = None
                restored_rng = True

            while cursor.global_step < args.max_steps and cursor.epoch == epoch:
                remaining = len(loader) - cursor.batch_in_epoch
                if remaining < args.gradient_accumulation_steps:
                    cursor = advance_cursor_batches(
                        cursor,
                        consumed_batches=remaining,
                        batches_per_epoch=len(loader),
                        optimizer_steps=0,
                    )
                    break

                optimizer.zero_grad(set_to_none=True)
                sums: dict[str, float] = {}
                step_started = time.perf_counter()
                apply_gt = gate._gt_due(args, cursor.global_step + 1)

                for micro_index in range(args.gradient_accumulation_steps):
                    batch_cpu = next(iterator)
                    teacher_velocity = train_cache.load_batch(
                        batch_cpu,
                        device=device,
                        dtype=dtype,
                    )
                    batch = gate.move_video_batch(
                        batch_cpu,
                        device=device,
                        dtype=dtype,
                    )
                    synchronize = micro_index + 1 == args.gradient_accumulation_steps
                    sync_context = nullcontext() if synchronize else ddp_model.no_sync()
                    with sync_context:
                        with torch.autocast(
                            "cuda",
                            dtype=dtype,
                            enabled=autocast_enabled,
                        ):
                            output = ddp_model(batch)
                            student_prediction = None
                            teacher_prediction = None
                            if gate._needs_rgb(args, apply_gt=apply_gt):
                                student_prediction, teacher_prediction = gate._decode_pair(
                                    output,
                                    teacher_velocity,
                                    closure,
                                )
                        objective = velocity_distillation_objective(
                            output["velocity"],
                            teacher_velocity,
                            student_prediction=student_prediction,
                            teacher_prediction=teacher_prediction,
                            target=output["target"] if apply_gt else None,
                            velocity_mse_weight=args.velocity_mse_weight,
                            velocity_cosine_weight=args.velocity_cosine_weight,
                            output_l1_weight=args.output_l1_weight,
                            output_temporal_weight=args.output_temporal_weight,
                            gt_loss_mode=args.gt_loss_mode if apply_gt else "none",
                            gt_pixel_weight=args.gt_pixel_weight if apply_gt else 0.0,
                            gt_temporal_weight=(
                                args.gt_temporal_weight if apply_gt else 0.0
                            ),
                            epsilon=args.loss_epsilon,
                        )
                        scaled_loss = objective["loss"] / args.gradient_accumulation_steps
                        if not torch.isfinite(objective["loss"].detach()).item():
                            raise FloatingPointError("Non-finite distillation loss")
                        if scaler.is_enabled():
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                    for key, value in objective.items():
                        sums[key] = sums.get(key, 0.0) + float(
                            value.detach().float().item()
                        )

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                gradients = gate.gradient_summary(closure.named_parameters())
                if int(gradients["gradient_tensors"]) == 0:
                    raise RuntimeError("Backward produced no trainable gradients")
                if int(gradients["nonfinite_elements"]) != 0:
                    raise FloatingPointError("Backward produced non-finite gradients")
                if int(gradients["missing_gradient_count"]) != 0:
                    raise RuntimeError(
                        f"Missing gradients: {gradients['missing_gradient_examples']}"
                    )
                gradient_norm = float(gradients["global_l2"])
                if args.max_grad_norm > 0:
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            [
                                parameter
                                for _, parameter in gate.trainable_named_parameters(closure)
                            ],
                            max_norm=args.max_grad_norm,
                            error_if_nonfinite=True,
                        )
                        .float()
                        .item()
                    )

                scale_before = float(scaler.get_scale())
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scale_after = float(scaler.get_scale())
                overflow = torch.tensor(
                    [
                        1
                        if scaler.is_enabled() and scale_after < scale_before
                        else 0
                    ],
                    device=device,
                    dtype=torch.int32,
                )
                dist.all_reduce(overflow, op=dist.ReduceOp.MAX)
                if int(overflow.item()) != 0:
                    raise FloatingPointError(
                        "At least one rank skipped an optimizer step"
                    )
                optimizer.zero_grad(set_to_none=True)

                cursor = advance_cursor_batches(
                    cursor,
                    consumed_batches=args.gradient_accumulation_steps,
                    batches_per_epoch=len(loader),
                    optimizer_steps=1,
                )
                averages, gradient_norm, step_seconds = _reduce_step(
                    sums,
                    denominator=args.gradient_accumulation_steps * world_size,
                    gradient_norm=gradient_norm,
                    step_seconds=time.perf_counter() - step_started,
                    device=device,
                    world_size=world_size,
                )
                relative_l2 = math.sqrt(
                    max(averages["velocity_mse"], 0.0)
                    / max(averages["teacher_velocity_power"], 1e-12)
                )
                last_record = {
                    "global_step": cursor.global_step,
                    "epoch": cursor.epoch,
                    "batch_in_epoch": cursor.batch_in_epoch,
                    "world_size": world_size,
                    "global_effective_batch_size": effective_batch,
                    **averages,
                    "velocity_relative_l2": relative_l2,
                    "gt_loss_due": float(apply_gt),
                    "gradient_norm": gradient_norm,
                    "grad_scaler_enabled": float(scaler.is_enabled()),
                    "grad_scaler_scale": scale_after,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "step_seconds": step_seconds,
                    "peak_allocated_gb_per_rank": (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    ),
                }
                if rank == 0:
                    gate.append_jsonl(train_log, last_record)
                    if cursor.global_step % args.log_every == 0:
                        print(
                            f"step={cursor.global_step} "
                            f"epoch={cursor.epoch} "
                            f"batch={cursor.batch_in_epoch} "
                            f"loss={averages['loss']:.8f} "
                            f"rel_l2={relative_l2:.6f} "
                            f"cos={averages['velocity_cosine']:.6f} "
                            f"gt_guard={averages['gt_pixel_guard']:.6f} "
                            f"time={step_seconds:.3f}s",
                            flush=True,
                        )
                    if writer is not None:
                        for key, value in last_record.items():
                            if isinstance(value, (int, float)) and key != "global_step":
                                prefix = (
                                    "train/gt_constraint/"
                                    if key.startswith("gt_")
                                    else "train/teacher_distillation/"
                                )
                                name = (
                                    key.removeprefix("gt_")
                                    if key.startswith("gt_")
                                    else key
                                )
                                writer.add_scalar(
                                    prefix + name,
                                    value,
                                    cursor.global_step,
                                )
                        writer.flush()

                validation_due = (
                    validation_configured
                    and args.validate_every > 0
                    and (
                        cursor.global_step % args.validate_every == 0
                        or cursor.global_step == args.max_steps
                    )
                )
                improved = False
                if validation_due:
                    dist.barrier()
                    if rank == 0:
                        previous_best = best_relative_l2
                        training_rng = capture_rng_state()
                        try:
                            validation = run_validation(cursor.global_step)
                        finally:
                            restore_rng_state(training_rng)
                        improved = (
                            float(validation["velocity_relative_l2"])
                            < previous_best
                        )
                        print(
                            f"validation step={cursor.global_step} "
                            f"rel_l2={validation['velocity_relative_l2']:.6f} "
                            f"cos={validation['velocity_cosine']:.6f} "
                            f"ref_psnr={validation['student_teacher_psnr']:.4f} "
                            f"gt_psnr={validation['student_gt_psnr']:.4f}",
                            flush=True,
                        )
                    marker = torch.tensor(
                        [1 if improved else 0],
                        device=device,
                        dtype=torch.int32,
                    )
                    dist.broadcast(marker, src=0)
                    improved = bool(marker.item())
                    dist.barrier()

                should_save = (
                    cursor.global_step % args.save_every == 0
                    or cursor.global_step == args.max_steps
                    or validation_due
                )
                if should_save:
                    if rank == 0:
                        state_summary = gate.optimizer_state_summary(optimizer)
                        if int(state_summary["nonfinite_elements"]) != 0:
                            raise FloatingPointError("Optimizer state is non-finite")
                    dist.barrier()
                    last_checkpoint = _checkpoint(
                        run_dir=run_dir,
                        closure=closure,
                        optimizer=optimizer,
                        scaler=scaler,
                        cursor=cursor,
                        fingerprint=fingerprint,
                        last_record=last_record,
                        runtime_dtype=dtype,
                        rank=rank,
                        world_size=world_size,
                    )
                    if rank == 0:
                        if improved:
                            gate.write_json(
                                run_dir / "best.json",
                                {
                                    "checkpoint": str(last_checkpoint.relative_to(run_dir)),
                                    "global_step": cursor.global_step,
                                    "velocity_relative_l2": best_relative_l2,
                                },
                            )
                        print(f"saved checkpoint: {last_checkpoint}", flush=True)

        if rank == 0:
            summary = {
                "status": "PASS",
                "global_step": cursor.global_step,
                "epoch": cursor.epoch,
                "batch_in_epoch": cursor.batch_in_epoch,
                "max_steps": args.max_steps,
                "elapsed_seconds": time.perf_counter() - started,
                "best_velocity_relative_l2": (
                    None if best_relative_l2 == float("inf") else best_relative_l2
                ),
                "last_record": last_record,
                "last_checkpoint": (
                    None if last_checkpoint is None else str(last_checkpoint)
                ),
                "run_dir": str(run_dir),
                "runtime_dtype": gate._CANONICAL_DTYPE_NAME[dtype],
                "grad_scaler_enabled": bool(scaler.is_enabled()),
                "world_size": world_size,
                "exact_resume": True,
            }
            gate.write_json(run_dir / "summary.json", summary)
            print(json.dumps(summary, indent=2))
        return 0
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
