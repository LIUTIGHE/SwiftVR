#!/usr/bin/env python3
"""DDP endpoint teacher distillation for prompt-free SwiftVR adapters.

This first D0 trainer is intentionally a gate implementation: deterministic
cached teacher velocities, adapter-only DDP, fixed validation, TensorBoard, and
rank-0 delta checkpoints. It starts from the zero-adapter folded checkpoint and
does not use GT pixel loss.
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
    configure_train_scope,
    gradient_summary,
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training import (
    VideoMetricAccumulator,
    append_jsonl,
    build_fp32_adamw,
    build_grad_scaler,
    cast_trainable_parameters,
    optimizer_state_summary,
    save_delta_checkpoint,
    seed_everything,
    trainable_named_parameters,
    write_latest_checkpoint,
)
from swiftvr.training.distillation import (
    DeterministicTripletViewDataset,
    DistillationMetricAccumulator,
    SwiftVRVelocityDistillationForward,
    TeacherVelocityCache,
    decode_student_prediction,
    decode_teacher_prediction,
    velocity_distillation_objective,
)
from swiftvr.training.reference import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--val-teacher-cache", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, action="append", default=None)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--val-crop-size", type=int, default=None)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--views-per-record", type=int, default=1)
    parser.add_argument("--view-seed", type=int, default=0)
    parser.add_argument("--val-views-per-record", type=int, default=1)
    parser.add_argument("--val-view-seed", type=int, default=9000001)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--allow-dtype-mismatch", action="store_true")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--velocity-mse-weight", type=float, default=1.0)
    parser.add_argument("--velocity-cosine-weight", type=float, default=1.0)
    parser.add_argument("--output-l1-weight", type=float, default=0.0)
    parser.add_argument("--output-temporal-weight", type=float, default=0.0)
    parser.add_argument("--loss-epsilon", type=float, default=1e-8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--expected-global-batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=20)
    parser.add_argument("--validate-at-start", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    return parser


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def init_distributed() -> tuple[int, int, int, torch.device]:
    missing = [name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ]
    if missing:
        raise RuntimeError("Launch with torchrun; missing: " + ", ".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL DDP requires CUDA")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def build_cached_dataset(
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
) -> Subset:
    base = TripletVideoDataset(
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
    count = int(cache.metadata["sample_count"])
    if count <= 0 or count > len(views):
        raise ValueError(f"Invalid cached sample_count={count} for dataset={len(views)}")
    if set(cache.samples_by_index) != set(range(count)):
        raise ValueError("Teacher cache must contain a contiguous dataset prefix")
    return Subset(views, range(count))


def make_train_loader(
    dataset: Subset,
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
    sampler.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=False,
    )


def create_writer(args: argparse.Namespace, run_dir: Path, rank: int):
    if rank != 0 or args.no_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        raise RuntimeError("TensorBoard is required unless --no-tensorboard is set") from exc
    log_dir = args.tensorboard_dir.expanduser().resolve() if args.tensorboard_dir else run_dir / "tensorboard"
    return SummaryWriter(log_dir=str(log_dir))


def objective_for_batch(
    output: Mapping[str, torch.Tensor],
    teacher_velocity: torch.Tensor,
    closure: SwiftVRVelocityDistillationForward,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    student_prediction = None
    teacher_prediction = None
    if args.output_l1_weight != 0.0 or args.output_temporal_weight != 0.0:
        output_frames = int(output["target"].shape[1])
        student_prediction = decode_student_prediction(
            reae=closure.reae,
            z_lq=output["z_lq"],
            student_velocity=output["velocity"],
            output_frames=output_frames,
        )
        teacher_prediction = decode_teacher_prediction(
            reae=closure.reae,
            z_lq=output["z_lq"],
            teacher_velocity=teacher_velocity,
            output_frames=output_frames,
        )
    return velocity_distillation_objective(
        output["velocity"],
        teacher_velocity,
        student_prediction=student_prediction,
        teacher_prediction=teacher_prediction,
        velocity_mse_weight=args.velocity_mse_weight,
        velocity_cosine_weight=args.velocity_cosine_weight,
        output_l1_weight=args.output_l1_weight,
        output_temporal_weight=args.output_temporal_weight,
        epsilon=args.loss_epsilon,
    )


@torch.no_grad()
def validate_rank0(
    closure: SwiftVRVelocityDistillationForward,
    loader: DataLoader,
    cache: TeacherVelocityCache,
    *,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    closure.eval()
    velocity_metrics = DistillationMetricAccumulator()
    student_teacher = VideoMetricAccumulator()
    student_gt = VideoMetricAccumulator()
    loss_sums: dict[str, float] = {}
    batches = 0
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    try:
        for batch_cpu in loader:
            teacher_velocity = cache.load_batch(batch_cpu, device=device, dtype=dtype)
            batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
            with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                output = closure(batch)
                output_frames = int(output["target"].shape[1])
                student_prediction = decode_student_prediction(
                    reae=closure.reae,
                    z_lq=output["z_lq"],
                    student_velocity=output["velocity"],
                    output_frames=output_frames,
                )
                teacher_prediction = decode_teacher_prediction(
                    reae=closure.reae,
                    z_lq=output["z_lq"],
                    teacher_velocity=teacher_velocity,
                    output_frames=output_frames,
                )
                objective = velocity_distillation_objective(
                    output["velocity"],
                    teacher_velocity,
                    student_prediction=student_prediction,
                    teacher_prediction=teacher_prediction,
                    velocity_mse_weight=args.velocity_mse_weight,
                    velocity_cosine_weight=args.velocity_cosine_weight,
                    output_l1_weight=args.output_l1_weight,
                    output_temporal_weight=args.output_temporal_weight,
                    epsilon=args.loss_epsilon,
                )
            velocity_metrics.update(output["velocity"], teacher_velocity)
            student_teacher.update(student_prediction, teacher_prediction, clamp=True)
            student_gt.update(student_prediction, output["target"], clamp=True)
            for key, value in objective.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value.float().item())
            batches += 1
    finally:
        closure.train()
        closure.reae.eval()
    if batches == 0:
        raise RuntimeError("Validation produced no batches")
    velocity = velocity_metrics.compute()
    st = student_teacher.compute()
    sg = student_gt.compute()
    result: dict[str, float | int] = {**velocity, "batches": batches}
    result.update({f"student_teacher_{key}": value for key, value in st.items()})
    result.update({f"student_gt_{key}": value for key, value in sg.items()})
    result.update({key: value / batches for key, value in loss_sums.items()})
    return result


def save_checkpoint(
    run_dir: Path,
    closure: SwiftVRVelocityDistillationForward,
    optimizer: torch.optim.Optimizer,
    scaler,
    *,
    step: int,
    last_record: Mapping[str, object],
) -> Path:
    checkpoint = run_dir / "checkpoints" / f"step_{step:08d}"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    save_delta_checkpoint(
        checkpoint,
        closure,
        optimizer,
        step=step,
        grad_scaler=scaler,
        metadata={
            "trainer": "teacher_distillation_ddp_gate_v1",
            "last_loss": last_record.get("loss"),
            "last_velocity_relative_l2": last_record.get("velocity_relative_l2"),
            "last_velocity_cosine": last_record.get("velocity_cosine"),
        },
    )
    write_latest_checkpoint(run_dir, checkpoint)
    return checkpoint


def main() -> int:
    args = build_parser().parse_args()
    rank, local_rank, world_size, device = init_distributed()
    writer = None
    try:
        if args.num_workers != 0:
            raise ValueError("The first D0 gate requires --num-workers 0")
        if args.max_steps <= 0 or args.gradient_accumulation_steps <= 0:
            raise ValueError("max-steps and accumulation must be positive")
        if args.validate_every > 0 and (not args.val_manifest or not args.val_teacher_cache):
            raise ValueError("Validation requires --val-manifest and --val-teacher-cache")
        if args.loss_epsilon <= 0:
            raise ValueError("loss-epsilon must be positive")
        effective_batch = world_size * args.batch_size * args.gradient_accumulation_steps
        if args.expected_global_batch_size is not None and effective_batch != args.expected_global_batch_size:
            raise ValueError(
                f"Global effective batch={effective_batch}, expected={args.expected_global_batch_size}"
            )

        run_dir = args.output_dir.expanduser().resolve()
        if rank == 0:
            if (run_dir / "train_log.jsonl").exists() or (run_dir / "latest.json").exists():
                raise FileExistsError("Output directory already contains a run")
            run_dir.mkdir(parents=True, exist_ok=True)
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
        seed_everything(args.seed + rank)

        train_cache = TeacherVelocityCache(args.teacher_cache)
        train_dataset = build_cached_dataset(
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
            val_crop = args.crop_size if args.val_crop_size is None else args.val_crop_size
            val_dataset = build_cached_dataset(
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

        run_config = {
            "trainer": "teacher_distillation_ddp_gate_v1",
            "base_checkpoint": str(base_checkpoint),
            "teacher_cache": str(args.teacher_cache.expanduser().resolve()),
            "teacher_cache_metadata_sha256": sha256_file(
                args.teacher_cache.expanduser().resolve() / "metadata.json"
            ),
            "val_teacher_cache": None if args.val_teacher_cache is None else str(args.val_teacher_cache.expanduser().resolve()),
            "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
            "val_manifests": [str(path.expanduser().resolve()) for path in (args.val_manifest or [])],
            "clip_length": args.clip_length,
            "crop_size": args.crop_size,
            "scale": args.scale,
            "views_per_record": args.views_per_record,
            "view_seed": args.view_seed,
            "world_size": world_size,
            "local_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_effective_batch_size": effective_batch,
            "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
            "velocity_mse_weight": args.velocity_mse_weight,
            "velocity_cosine_weight": args.velocity_cosine_weight,
            "output_l1_weight": args.output_l1_weight,
            "output_temporal_weight": args.output_temporal_weight,
            "learning_rate": args.learning_rate,
        }
        if rank == 0:
            write_json(run_dir / "run_config.json", run_config)
        dist.barrier()

        reae = ReAE(str(base_checkpoint / args.reae_filename))
        transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
            str(base_checkpoint),
            subfolder=args.transformer_subfolder,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        configure_train_scope(reae, transformer, "adapter")
        reae.to(device=device, dtype=dtype).eval()
        transformer.to(device=device, dtype=dtype)
        closure = SwiftVRVelocityDistillationForward(
            reae,
            transformer,
            attention_backend=args.attention_backend,
        )
        closure.train()
        closure.reae.eval()
        cast_trainable_parameters(closure, dtype=torch.float32)
        optimizer = build_fp32_adamw(
            closure,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.optimizer_eps,
        )
        scaler = build_grad_scaler(device, dtype)
        ddp_model = DistributedDataParallel(
            closure,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        writer = create_writer(args, run_dir, rank)

        train_log = run_dir / "train_log.jsonl"
        val_log = run_dir / "val_log.jsonl"
        best_relative_l2 = float("inf")
        global_step = 0
        epoch = 0
        last_record: dict[str, object] = {}
        started = time.perf_counter()

        if args.validate_at_start and val_loader is not None and val_cache is not None:
            dist.barrier()
            if rank == 0:
                baseline = validate_rank0(
                    closure, val_loader, val_cache, device=device, dtype=dtype, args=args
                )
                append_jsonl(val_log, {"global_step": 0, **baseline})
                best_relative_l2 = float(baseline["velocity_relative_l2"])
                print(
                    f"validation start rel_l2={baseline['velocity_relative_l2']:.6f} "
                    f"cos={baseline['velocity_cosine']:.6f} "
                    f"ref_psnr={baseline['student_teacher_psnr']:.4f}",
                    flush=True,
                )
            dist.barrier()

        autocast_enabled = dtype in (torch.float16, torch.bfloat16)
        while global_step < args.max_steps:
            loader = make_train_loader(
                train_dataset,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                args=args,
            )
            if len(loader) < args.gradient_accumulation_steps:
                raise RuntimeError("Per-rank epoch is shorter than gradient accumulation")
            iterator = iter(loader)
            while global_step < args.max_steps:
                optimizer.zero_grad(set_to_none=True)
                sums: dict[str, float] = {}
                step_started = time.perf_counter()
                consumed = 0
                for micro_index in range(args.gradient_accumulation_steps):
                    try:
                        batch_cpu = next(iterator)
                    except StopIteration:
                        break
                    consumed += 1
                    teacher_velocity = train_cache.load_batch(
                        batch_cpu, device=device, dtype=dtype
                    )
                    batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                    synchronize = micro_index + 1 == args.gradient_accumulation_steps
                    sync_context = nullcontext() if synchronize else ddp_model.no_sync()
                    with sync_context:
                        with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                            output = ddp_model(batch)
                            objective = objective_for_batch(
                                output, teacher_velocity, closure, args
                            )
                            scaled_loss = objective["loss"] / args.gradient_accumulation_steps
                        if not torch.isfinite(objective["loss"].detach()).item():
                            raise FloatingPointError("Non-finite distillation loss")
                        scaler.scale(scaled_loss).backward()
                    for key, value in objective.items():
                        sums[key] = sums.get(key, 0.0) + float(value.detach().float().item())
                if consumed != args.gradient_accumulation_steps:
                    optimizer.zero_grad(set_to_none=True)
                    break

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
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            [parameter for _, parameter in trainable_named_parameters(closure)],
                            max_norm=args.max_grad_norm,
                            error_if_nonfinite=True,
                        ).float().item()
                    )
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                overflow = torch.tensor(
                    [1 if scale_after < scale_before else 0],
                    device=device,
                    dtype=torch.int32,
                )
                dist.all_reduce(overflow, op=dist.ReduceOp.MAX)
                if int(overflow.item()) != 0:
                    raise FloatingPointError("At least one rank skipped a step")
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                keys = tuple(sums)
                packed = torch.tensor(
                    [sums[key] for key in keys] + [grad_norm, time.perf_counter() - step_started],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                denominator = args.gradient_accumulation_steps * world_size
                averages = {
                    key: float(packed[index].item()) / denominator
                    for index, key in enumerate(keys)
                }
                grad_norm_global = float(packed[-2].item()) / world_size
                step_seconds = float(packed[-1].item()) / world_size
                relative_l2 = math.sqrt(
                    max(averages["velocity_mse"], 0.0)
                    / max(averages["teacher_velocity_power"], 1e-12)
                )
                last_record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "world_size": world_size,
                    "global_effective_batch_size": effective_batch,
                    **averages,
                    "velocity_relative_l2": relative_l2,
                    "gradient_norm": grad_norm_global,
                    "grad_scaler_scale": scale_after,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "step_seconds": step_seconds,
                    "peak_allocated_gb_per_rank": torch.cuda.max_memory_allocated(device) / 1024**3,
                }
                if rank == 0:
                    append_jsonl(train_log, last_record)
                    if global_step % args.log_every == 0:
                        print(
                            f"step={global_step} loss={averages['loss']:.8f} "
                            f"rel_l2={relative_l2:.6f} "
                            f"cos={averages['velocity_cosine']:.6f} "
                            f"time={step_seconds:.3f}s",
                            flush=True,
                        )
                    if writer is not None:
                        for key, value in last_record.items():
                            if isinstance(value, (int, float)) and key != "global_step":
                                writer.add_scalar(f"train/teacher_distillation/{key}", value, global_step)
                        writer.flush()

                validation_due = (
                    val_loader is not None
                    and val_cache is not None
                    and args.validate_every > 0
                    and (global_step % args.validate_every == 0 or global_step == args.max_steps)
                )
                improved = False
                if validation_due:
                    dist.barrier()
                    if rank == 0:
                        validation = validate_rank0(
                            closure,
                            val_loader,
                            val_cache,
                            device=device,
                            dtype=dtype,
                            args=args,
                        )
                        append_jsonl(val_log, {"global_step": global_step, **validation})
                        improved = float(validation["velocity_relative_l2"]) < best_relative_l2
                        if improved:
                            best_relative_l2 = float(validation["velocity_relative_l2"])
                        print(
                            f"validation step={global_step} "
                            f"rel_l2={validation['velocity_relative_l2']:.6f} "
                            f"cos={validation['velocity_cosine']:.6f} "
                            f"ref_psnr={validation['student_teacher_psnr']:.4f} "
                            f"gt_psnr={validation['student_gt_psnr']:.4f}",
                            flush=True,
                        )
                        if writer is not None:
                            for key, value in validation.items():
                                if not isinstance(value, (int, float)):
                                    continue
                                if key.startswith("student_teacher_"):
                                    tag = "val/student_teacher/" + key.removeprefix("student_teacher_")
                                elif key.startswith("student_gt_"):
                                    tag = "val/student_gt/" + key.removeprefix("student_gt_")
                                else:
                                    tag = "val/teacher_distillation/" + key
                                writer.add_scalar(tag, value, global_step)
                            writer.flush()
                    marker = torch.tensor([1 if improved else 0], device=device)
                    dist.broadcast(marker, src=0)
                    improved = bool(marker.item())
                    dist.barrier()

                should_save = (
                    global_step % args.save_every == 0
                    or global_step == args.max_steps
                    or validation_due
                )
                if should_save:
                    dist.barrier()
                    if rank == 0:
                        state_summary = optimizer_state_summary(optimizer)
                        if int(state_summary["nonfinite_elements"]) != 0:
                            raise FloatingPointError("Optimizer state is non-finite")
                        checkpoint = save_checkpoint(
                            run_dir,
                            closure,
                            optimizer,
                            scaler,
                            step=global_step,
                            last_record=last_record,
                        )
                        if improved:
                            write_json(
                                run_dir / "best.json",
                                {
                                    "checkpoint": str(checkpoint.relative_to(run_dir)),
                                    "global_step": global_step,
                                    "velocity_relative_l2": best_relative_l2,
                                },
                            )
                        print(f"saved checkpoint: {checkpoint}", flush=True)
                    dist.barrier()
            epoch += 1

        if rank == 0:
            summary = {
                "status": "PASS",
                "global_step": global_step,
                "max_steps": args.max_steps,
                "elapsed_seconds": time.perf_counter() - started,
                "best_velocity_relative_l2": None if best_relative_l2 == float("inf") else best_relative_l2,
                "last_record": last_record,
                "run_dir": str(run_dir),
                "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
                "world_size": world_size,
            }
            write_json(run_dir / "summary.json", summary)
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
