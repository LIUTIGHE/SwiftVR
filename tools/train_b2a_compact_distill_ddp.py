#!/usr/bin/env python3
"""Train the B2-A Wan-1.3B-shaped compact DiT against Stage-A 200k velocity.

The training objective is decoder-free.  The compact DiT receives the same ReAE
LQ latent as the Stage-A no-time/no-prompt teacher and matches its cached endpoint
velocity using normalized MSE + cosine loss.  The original frozen ReAE decoder is
used only by rank-0 validation to render fixed teacher/student videos and frames;
RGB/GT metrics never contribute gradients or choose the best model.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import train_teacher_distillation_ddp as stage_a
from smoke_training_forward import (
    _CANONICAL_DTYPE_NAME,
    configure_train_scope,
    gradient_summary,
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training import (
    DistillationMetricAccumulator,
    TeacherVelocityCache,
    VideoMetricAccumulator,
    append_jsonl,
    build_fp32_adamw,
    build_grad_scaler,
    cast_trainable_parameters,
    decode_student_prediction,
    decode_teacher_prediction,
    seed_everything,
    trainable_named_parameters,
    velocity_distillation_objective,
    write_latest_checkpoint,
)
from swiftvr.training.b2a_width import (
    B2ACompactVelocityDistillationForward,
    B2AWidthSpec,
    transformer_width_shape,
)
from swiftvr.training.distillation_visuals import export_validation_visuals
from swiftvr.training.perceptual_review import parse_csv_ints
from swiftvr.training.reference import sha256_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--student-init", type=Path, required=True)
    p.add_argument("--teacher-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--val-teacher-cache", type=Path, default=None)
    p.add_argument("--val-manifest", type=Path, action="append", default=None)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--clip-length", type=int, default=13)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--val-crop-size", type=int, default=None)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--views-per-record", type=int, default=8)
    p.add_argument("--view-seed", type=int, default=20260805)
    p.add_argument("--val-views-per-record", type=int, default=1)
    p.add_argument("--val-view-seed", type=int, default=9000001)
    p.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    p.add_argument("--vertical-flip-probability", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--allow-dtype-mismatch", action="store_true")
    p.add_argument("--attention-backend", default="sdpa")

    p.add_argument("--velocity-mse-weight", type=float, default=1.0)
    p.add_argument("--velocity-cosine-weight", type=float, default=1.0)
    p.add_argument("--loss-epsilon", type=float, default=1e-8)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer-eps", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--expected-global-batch-size", type=int, default=None)
    p.add_argument("--lr-warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--validate-at-start", action="store_true")
    p.add_argument("--save-every", type=int, default=1000)

    p.add_argument("--visual-validation-samples", type=int, default=2)
    p.add_argument("--visual-frame-indices", default="0,6,12")
    p.add_argument("--visual-video-fps", type=float, default=8.0)
    p.add_argument("--visual-difference-scale", type=float, default=4.0)
    p.add_argument("--visualize-every", type=int, default=None)
    p.add_argument("--no-validation-visuals", action="store_true")

    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tensorboard-dir", type=Path, default=None)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")

    p.add_argument("--student-hidden-dim", type=int, default=1536)
    p.add_argument("--student-num-heads", type=int, default=12)
    p.add_argument("--student-head-dim", type=int, default=128)
    p.add_argument("--student-ffn-dim", type=int, default=8960)
    p.add_argument("--student-num-layers", type=int, default=30)
    p.add_argument("--student-adapter-dim", type=int, default=128)
    return p


def _spec(args: argparse.Namespace) -> B2AWidthSpec:
    return B2AWidthSpec(
        hidden_dim=args.student_hidden_dim,
        num_heads=args.student_num_heads,
        head_dim=args.student_head_dim,
        ffn_dim=args.student_ffn_dim,
        num_layers=args.student_num_layers,
        adapter_dim=args.student_adapter_dim,
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    positive = {
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "log_every": args.log_every,
        "save_every": args.save_every,
    }
    bad = [name for name, value in positive.items() if int(value) <= 0]
    if bad:
        raise ValueError(f"Arguments must be positive: {bad}")
    if args.num_workers != 0:
        raise ValueError("B2-A v1 keeps --num-workers 0 for deterministic diagnosis")
    if args.validate_every > 0 and (not args.val_manifest or not args.val_teacher_cache):
        raise ValueError("Validation requires --val-manifest and --val-teacher-cache")
    if args.lr_warmup_steps < 0 or args.lr_warmup_steps >= args.max_steps:
        raise ValueError("lr-warmup-steps must be in [0,max_steps)")
    if not 0.0 < args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio must be in (0,1]")
    if args.learning_rate <= 0 or args.optimizer_eps <= 0 or args.loss_epsilon <= 0:
        raise ValueError("learning-rate/eps values must be positive")
    if args.weight_decay < 0 or args.max_grad_norm < 0:
        raise ValueError("weight-decay/max-grad-norm must be non-negative")
    if args.velocity_mse_weight < 0 or args.velocity_cosine_weight < 0:
        raise ValueError("velocity loss weights must be non-negative")
    if args.velocity_mse_weight == 0 and args.velocity_cosine_weight == 0:
        raise ValueError("At least one velocity loss weight must be nonzero")
    if args.visual_validation_samples < 0:
        raise ValueError("visual-validation-samples must be non-negative")
    if args.visual_video_fps <= 0 or args.visual_difference_scale <= 0:
        raise ValueError("visual FPS/difference scale must be positive")
    if args.visualize_every is not None and args.visualize_every <= 0:
        raise ValueError("visualize-every must be positive")
    return parse_csv_ints(args.visual_frame_indices)


def _lr_for_step(args: argparse.Namespace, step: int) -> float:
    base = float(args.learning_rate)
    if args.lr_warmup_steps and step <= args.lr_warmup_steps:
        return base * step / args.lr_warmup_steps
    span = max(args.max_steps - args.lr_warmup_steps, 1)
    progress = min(max((step - args.lr_warmup_steps) / span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine)


def _batch_text(batch: Mapping[str, object], key: str, index: int) -> str:
    value = batch.get(key)
    if isinstance(value, (list, tuple)):
        return str(value[index])
    if isinstance(value, torch.Tensor):
        item = value[index]
        return str(item.item() if item.ndim == 0 else item.tolist())
    return str(value)


@torch.no_grad()
def validate_rank0(
    closure: B2ACompactVelocityDistillationForward,
    loader: DataLoader,
    cache: TeacherVelocityCache,
    *,
    device: torch.device,
    dtype: torch.dtype,
    visual_samples: int,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    """Full velocity validation plus decoder-only RGB diagnostics."""

    closure.eval()
    velocity_metrics = DistillationMetricAccumulator()
    latent_metrics = DistillationMetricAccumulator()
    student_teacher = VideoMetricAccumulator()
    student_gt = VideoMetricAccumulator()
    teacher_gt = VideoMetricAccumulator()
    visuals: list[dict[str, object]] = []
    batches = 0
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    try:
        for batch_cpu in loader:
            teacher_velocity = cache.load_batch(batch_cpu, device=device, dtype=dtype)
            batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
            with torch.autocast(
                "cuda", dtype=dtype,
                enabled=device.type == "cuda" and autocast_enabled,
            ):
                output = closure(batch)
                z_student = output["z_lq"] - output["velocity"]
                z_teacher = output["z_lq"] - teacher_velocity
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

            velocity_metrics.update(output["velocity"], teacher_velocity)
            latent_metrics.update(z_student, z_teacher)
            student_teacher.update(student_prediction, teacher_prediction, clamp=True)
            student_gt.update(student_prediction, output["target"], clamp=True)
            teacher_gt.update(teacher_prediction, output["target"], clamp=True)

            if len(visuals) < visual_samples:
                for local_index in range(int(student_prediction.shape[0])):
                    if len(visuals) >= visual_samples:
                        break
                    visuals.append(
                        {
                            "record_uid": _batch_text(batch_cpu, "record_uid", local_index),
                            "lq_input": output["lq_input"][local_index].clamp(0, 1).cpu(),
                            "target": output["target"][local_index].clamp(0, 1).cpu(),
                            "teacher_prediction": teacher_prediction[local_index].clamp(0, 1).cpu(),
                            "student_prediction": student_prediction[local_index].clamp(0, 1).cpu(),
                        }
                    )
            batches += 1
    finally:
        closure.train()
        closure.reae.eval()

    if batches == 0:
        raise RuntimeError("Validation produced no batches")
    result: dict[str, float | int] = {**velocity_metrics.compute(), "batches": batches}
    latent = latent_metrics.compute()
    rename = {
        "velocity_mse": "restored_latent_mse",
        "velocity_rmse": "restored_latent_rmse",
        "velocity_normalized_mse": "restored_latent_normalized_mse",
        "velocity_relative_l2": "restored_latent_relative_l2",
        "velocity_cosine": "restored_latent_cosine",
        "velocity_elements": "restored_latent_elements",
        "samples": "restored_latent_samples",
    }
    result.update({rename[key]: value for key, value in latent.items()})
    result.update({f"student_teacher_{key}": value for key, value in student_teacher.compute().items()})
    result.update({f"student_gt_{key}": value for key, value in student_gt.compute().items()})
    result.update({f"teacher_gt_{key}": value for key, value in teacher_gt.compute().items()})
    return result, visuals


def _write_validation_scalars(writer, step: int, validation: Mapping[str, object]) -> None:
    if writer is None:
        return
    for key, value in validation.items():
        if not isinstance(value, (int, float)):
            continue
        if key.startswith("student_teacher_"):
            tag = "val/rgb_student_teacher/" + key.removeprefix("student_teacher_")
        elif key.startswith("student_gt_"):
            tag = "val/rgb_student_gt/" + key.removeprefix("student_gt_")
        elif key.startswith("teacher_gt_"):
            tag = "val/rgb_teacher_gt/" + key.removeprefix("teacher_gt_")
        elif key.startswith("restored_latent_"):
            tag = "val/restored_latent/" + key.removeprefix("restored_latent_")
        else:
            tag = "val/velocity/" + key
        writer.add_scalar(tag, float(value), step)
    writer.flush()


def _save_snapshot(
    transformer: WanTransformer3DModelPromptFreeNoTime,
    checkpoint: Path,
    *,
    runtime_dtype: torch.dtype,
    transformer_subfolder: str,
    metadata: Mapping[str, object],
) -> None:
    """Save a reloadable compact Transformer only; no optimizer state in v1."""

    temp = checkpoint.with_name(checkpoint.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    transformer_dir = temp / transformer_subfolder
    transformer_dir.mkdir()
    transformer.save_config(str(transformer_dir))
    state = {
        name: value.detach().to(device="cpu", dtype=runtime_dtype).contiguous()
        for name, value in transformer.state_dict().items()
    }
    save_file(state, str(transformer_dir / "diffusion_pytorch_model.safetensors"))
    _write_json(temp / "metadata.json", metadata)
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    temp.replace(checkpoint)


def main() -> int:
    args = build_parser().parse_args()
    visual_frame_indices = _validate_args(args)
    spec = _spec(args)
    rank, local_rank, world_size, device = stage_a.init_distributed()
    writer = None
    try:
        effective_batch = world_size * args.batch_size * args.gradient_accumulation_steps
        if args.expected_global_batch_size is not None and effective_batch != args.expected_global_batch_size:
            raise ValueError(
                f"Global effective batch={effective_batch}, expected={args.expected_global_batch_size}"
            )

        run_dir = args.output_dir.expanduser().resolve()
        if rank == 0:
            if (run_dir / "train_log.jsonl").exists() or (run_dir / "latest.json").exists():
                raise FileExistsError("Output directory already contains a B2-A run")
            run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        base_root = args.base_checkpoint.expanduser().resolve()
        student_root = args.student_init.expanduser().resolve()
        folded_config = validate_folded_checkpoint(
            base_root,
            reae_filename=args.reae_filename,
            transformer_subfolder=args.transformer_subfolder,
        )
        dtype = resolve_runtime_dtype(
            args.dtype, folded_config, device,
            allow_mismatch=args.allow_dtype_mismatch,
        )
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")
        seed_everything(args.seed + rank)

        train_cache = TeacherVelocityCache(args.teacher_cache)
        if train_cache.metadata.get("kind") != "swiftvr_b2a_stage_a_teacher_velocity":
            raise ValueError(
                "B2-A requires Stage-A 200k no-time/no-prompt velocity cache; got "
                f"{train_cache.metadata.get('kind')!r}"
            )
        train_dataset = stage_a.build_cached_dataset(
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
            if val_cache.metadata.get("kind") != "swiftvr_b2a_stage_a_teacher_velocity":
                raise ValueError("Validation cache is not a B2-A Stage-A teacher cache")
            val_crop = args.crop_size if args.val_crop_size is None else args.val_crop_size
            val_dataset = stage_a.build_cached_dataset(
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

        reae = ReAE(str(base_root / args.reae_filename))
        transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
            str(student_root),
            subfolder=args.transformer_subfolder,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        shape = transformer_width_shape(transformer)
        expected_shape = {
            "hidden_dim": spec.hidden_dim,
            "num_heads": spec.num_heads,
            "head_dim": spec.head_dim,
            "ffn_dim": spec.ffn_dim,
            "num_layers": spec.num_layers,
            "adapter_dim": spec.adapter_dim,
        }
        if shape != expected_shape:
            raise ValueError(f"student-init shape mismatch: {shape} != {expected_shape}")

        train_scope = configure_train_scope(reae, transformer, "transformer")
        reae.to(device=device, dtype=dtype).eval()
        transformer.to(device=device, dtype=dtype)
        closure = B2ACompactVelocityDistillationForward(
            reae,
            transformer,
            attention_backend=args.attention_backend,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        )
        closure.train()
        closure.reae.eval()
        cast_summary = cast_trainable_parameters(closure, dtype=torch.float32)
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
        writer = stage_a.create_writer(args, run_dir, rank)

        visualize_every = args.validate_every if args.visualize_every is None else args.visualize_every
        visuals_enabled = (
            not args.no_validation_visuals
            and args.visual_validation_samples > 0
            and visualize_every > 0
        )
        run_config = {
            "trainer": "b2a_wan13_velocity_distill_ddp_v1",
            "base_checkpoint": str(base_root),
            "student_init": str(student_root),
            "student_shape": shape,
            "student_parameters": sum(parameter.numel() for parameter in transformer.parameters()),
            "train_scope": train_scope,
            "cast_trainable_parameters": cast_summary,
            "teacher_cache": str(args.teacher_cache.expanduser().resolve()),
            "teacher_cache_metadata_sha256": sha256_file(args.teacher_cache.expanduser().resolve() / "metadata.json"),
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
            "optimizer_master_dtype": "float32",
            "velocity_mse_weight": args.velocity_mse_weight,
            "velocity_cosine_weight": args.velocity_cosine_weight,
            "learning_rate": args.learning_rate,
            "lr_warmup_steps": args.lr_warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "decoder_in_training_loss": False,
            "validation_decoder": "original_frozen_reae",
            "visuals_enabled": visuals_enabled,
            "visualize_every": visualize_every,
            "visual_validation_samples": args.visual_validation_samples,
            "visual_frame_indices": list(visual_frame_indices),
            "checkpoint_format": "runtime-dtype transformer snapshot; optimizer state omitted",
        }
        if rank == 0:
            _write_json(run_dir / "run_config.json", run_config)
        dist.barrier()

        train_log = run_dir / "train_log.jsonl"
        val_log = run_dir / "val_log.jsonl"
        best_relative_l2 = float("inf")
        best_step: int | None = None
        global_step = 0
        epoch = 0
        last_record: dict[str, object] = {}
        started = time.perf_counter()

        def run_validation(step: int) -> dict[str, float | int] | None:
            nonlocal best_relative_l2, best_step
            if val_loader is None or val_cache is None:
                return None
            visual_due = visuals_enabled and (
                step == 0 or step % visualize_every == 0 or step == args.max_steps
            )
            validation, visual_samples = validate_rank0(
                closure,
                val_loader,
                val_cache,
                device=device,
                dtype=dtype,
                visual_samples=args.visual_validation_samples if visual_due else 0,
            )
            append_jsonl(val_log, {"global_step": step, **validation})
            _write_validation_scalars(writer, step, validation)
            if visual_due:
                report = export_validation_visuals(
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
                        "validation visual MP4 warnings: " + json.dumps(report["video_errors"]),
                        flush=True,
                    )
            value = float(validation["velocity_relative_l2"])
            if value < best_relative_l2:
                best_relative_l2 = value
                best_step = step
                _write_json(
                    run_dir / "best.json",
                    {
                        "global_step": step,
                        "velocity_relative_l2": value,
                        "velocity_cosine": float(validation["velocity_cosine"]),
                        "note": "Best is selected only by velocity rel-L2; RGB is diagnostic only.",
                    },
                )
            return validation

        validation_configured = bool(args.val_manifest and args.val_teacher_cache)
        if args.validate_at_start and validation_configured:
            dist.barrier()
            if rank == 0:
                baseline = run_validation(0)
                assert baseline is not None
                print(
                    f"B2-A init rel_l2={baseline['velocity_relative_l2']:.6f} "
                    f"cos={baseline['velocity_cosine']:.6f} "
                    f"latent_rel_l2={baseline['restored_latent_relative_l2']:.6f} "
                    f"rgb_teacher_psnr={baseline['student_teacher_psnr']:.4f}",
                    flush=True,
                )
            dist.barrier()

        autocast_enabled = dtype in (torch.float16, torch.bfloat16)
        while global_step < args.max_steps:
            loader = stage_a.make_train_loader(
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
                next_step = global_step + 1
                current_lr = _lr_for_step(args, next_step)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

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
                        with torch.autocast(
                            "cuda", dtype=dtype,
                            enabled=device.type == "cuda" and autocast_enabled,
                        ):
                            output = ddp_model(batch)
                        objective = velocity_distillation_objective(
                            output["velocity"],
                            teacher_velocity,
                            velocity_mse_weight=args.velocity_mse_weight,
                            velocity_cosine_weight=args.velocity_cosine_weight,
                            output_l1_weight=0.0,
                            output_temporal_weight=0.0,
                            gt_loss_mode="none",
                            gt_pixel_weight=0.0,
                            gt_temporal_weight=0.0,
                            epsilon=args.loss_epsilon,
                        )
                        scaled_loss = objective["loss"] / args.gradient_accumulation_steps
                        if not torch.isfinite(objective["loss"].detach()).item():
                            raise FloatingPointError("Non-finite B2-A distillation loss")
                        if scaler.is_enabled():
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                    for key in (
                        "loss",
                        "velocity_mse",
                        "velocity_normalized_mse",
                        "velocity_cosine",
                        "velocity_cosine_loss",
                        "teacher_velocity_power",
                    ):
                        sums[key] = sums.get(key, 0.0) + float(
                            objective[key].detach().float().item()
                        )

                if consumed != args.gradient_accumulation_steps:
                    optimizer.zero_grad(set_to_none=True)
                    break

                if scaler.is_enabled():
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
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scale_after = float(scaler.get_scale())
                overflow = torch.tensor(
                    [1 if scaler.is_enabled() and scale_after < scale_before else 0],
                    device=device,
                    dtype=torch.int32,
                )
                dist.all_reduce(overflow, op=dist.ReduceOp.MAX)
                if int(overflow.item()) != 0:
                    raise FloatingPointError("At least one rank skipped an optimizer step")
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                keys = tuple(sums)
                packed = torch.tensor(
                    [sums[key] for key in keys]
                    + [grad_norm, time.perf_counter() - step_started],
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
                    "learning_rate": current_lr,
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
                            f"lr={current_lr:.3e} grad={grad_norm_global:.4f} "
                            f"time={step_seconds:.3f}s "
                            f"peak={last_record['peak_allocated_gb_per_rank']:.2f}GB",
                            flush=True,
                        )
                    if writer is not None:
                        for key, value in last_record.items():
                            if isinstance(value, (int, float)) and key != "global_step":
                                writer.add_scalar("train/" + key, float(value), global_step)
                        writer.flush()

                validation_due = (
                    validation_configured
                    and args.validate_every > 0
                    and (
                        global_step % args.validate_every == 0
                        or global_step == args.max_steps
                    )
                )
                if validation_due:
                    dist.barrier()
                    if rank == 0:
                        validation = run_validation(global_step)
                        assert validation is not None
                        print(
                            f"validation step={global_step} "
                            f"rel_l2={validation['velocity_relative_l2']:.6f} "
                            f"cos={validation['velocity_cosine']:.6f} "
                            f"latent_rel_l2={validation['restored_latent_relative_l2']:.6f} "
                            f"rgb_teacher_psnr={validation['student_teacher_psnr']:.4f} "
                            f"rgb_gt_psnr={validation['student_gt_psnr']:.4f}",
                            flush=True,
                        )
                    dist.barrier()

                save_due = global_step % args.save_every == 0 or global_step == args.max_steps
                if save_due:
                    dist.barrier()
                    if rank == 0:
                        checkpoint = run_dir / "checkpoints" / f"step_{global_step:08d}"
                        _save_snapshot(
                            transformer,
                            checkpoint,
                            runtime_dtype=dtype,
                            transformer_subfolder=args.transformer_subfolder,
                            metadata={
                                "trainer": "b2a_wan13_velocity_distill_ddp_v1",
                                "global_step": global_step,
                                "source_student_init": str(student_root),
                                "velocity_relative_l2_train": relative_l2,
                                "best_validation_velocity_relative_l2": (
                                    None if best_relative_l2 == float("inf") else best_relative_l2
                                ),
                                "best_validation_step": best_step,
                                "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
                                "optimizer_state_saved": False,
                            },
                        )
                        write_latest_checkpoint(run_dir, checkpoint)
                        if best_step == global_step and (run_dir / "best.json").is_file():
                            best_info = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
                            best_info["checkpoint"] = str(checkpoint.relative_to(run_dir))
                            _write_json(run_dir / "best.json", best_info)
                        print(f"saved compact snapshot: {checkpoint}", flush=True)
                    dist.barrier()
            epoch += 1

        if rank == 0:
            summary = {
                "status": "PASS",
                "global_step": global_step,
                "max_steps": args.max_steps,
                "elapsed_seconds": time.perf_counter() - started,
                "best_velocity_relative_l2": (
                    None if best_relative_l2 == float("inf") else best_relative_l2
                ),
                "best_step": best_step,
                "last_record": last_record,
                "run_dir": str(run_dir),
                "student_shape": shape,
                "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
                "world_size": world_size,
            }
            _write_json(run_dir / "summary.json", summary)
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
