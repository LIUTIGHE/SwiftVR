#!/usr/bin/env python3
"""B2B-1B short joint recovery gate for tiny DiT + extreme decoder.

The frozen Stage-A ReAE encoder supplies z_LQ.  A D768/F4080 prompt-free/no-time
DiT and the [96,48,24,16] extreme decoder are optimized jointly against the
Stage-A 200k teacher-velocity cache plus online frozen full-ReAE teacher RGB.
The v1 objective is intentionally minimal:

  0.05 normalized velocity MSE + 0.05 velocity cosine
  + 1.0 teacher-RGB L1 + 0.5 GT-RGB L1

No GAN, LPIPS, feature KD, or temporal loss is used in this recovery gate.
DiT and decoder use separate optimizer groups and separate gradient clipping so
the decoder's much larger raw gradient norm cannot suppress DiT adaptation.
BF16/FP32 only: this gate deliberately avoids FP16 loss-scaling confounders.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
for search_root in (ROOT, TOOLS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from tools import train_teacher_distillation_ddp as stage_a
from tools.diagnose_b2b_extreme_train_val_gap import _resolve_student_root
from tools.smoke_training_forward import move_video_batch
from swiftvr.models import ReAE
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training import (
    DistillationMetricAccumulator,
    TeacherVelocityCache,
    VideoMetricAccumulator,
    append_jsonl,
    cast_trainable_parameters,
    decode_teacher_prediction,
    seed_everything,
    write_latest_checkpoint,
)
from swiftvr.training.b2a_width import transformer_width_shape
from swiftvr.training.b2b_joint import (
    B2B_EXTREME_DECODER_CHANNELS,
    B2B_TINY_SPEC,
    B2BJointForward,
    b2b_compute_budget,
    b2b_joint_objective,
)


DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--student-init", type=Path, required=True)
    p.add_argument("--decoder-init", type=Path, required=True)
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
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--attention-backend", default="sdpa")

    p.add_argument("--representation-mse-weight", type=float, default=0.05)
    p.add_argument("--representation-cosine-weight", type=float, default=0.05)
    p.add_argument("--teacher-rgb-l1-weight", type=float, default=1.0)
    p.add_argument("--gt-rgb-l1-weight", type=float, default=0.5)
    p.add_argument("--loss-epsilon", type=float, default=1e-8)

    p.add_argument("--dit-learning-rate", type=float, default=2e-5)
    p.add_argument("--decoder-learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer-eps", type=float, default=1e-8)
    p.add_argument("--dit-max-grad-norm", type=float, default=1.0)
    p.add_argument("--decoder-max-grad-norm", type=float, default=1.0)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--expected-global-batch-size", type=int, default=None)
    p.add_argument("--lr-warmup-steps", type=int, default=50)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--no-gradient-checkpointing", action="store_true")

    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--validate-at-start", action="store_true")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    return p


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "log_every",
        "save_every",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers != 0:
        raise ValueError("B2B-1B keeps --num-workers 0 for deterministic diagnosis")
    if args.validate_every > 0 and (not args.val_manifest or not args.val_teacher_cache):
        raise ValueError("Validation requires --val-manifest and --val-teacher-cache")
    if args.lr_warmup_steps < 0 or args.lr_warmup_steps >= args.max_steps:
        raise ValueError("lr-warmup-steps must be in [0,max_steps)")
    if not 0.0 < args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio must be in (0,1]")
    if args.dit_learning_rate <= 0 or args.decoder_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.optimizer_eps <= 0 or args.loss_epsilon <= 0:
        raise ValueError("eps values must be positive")
    if args.weight_decay < 0 or args.dit_max_grad_norm < 0 or args.decoder_max_grad_norm < 0:
        raise ValueError("weight-decay/grad norms must be non-negative")
    loss_weights = (
        args.representation_mse_weight,
        args.representation_cosine_weight,
        args.teacher_rgb_l1_weight,
        args.gt_rgb_l1_weight,
    )
    if any(float(value) < 0 for value in loss_weights) or not any(float(value) > 0 for value in loss_weights):
        raise ValueError("loss weights must be non-negative with at least one nonzero")


def _lr_scale(args: argparse.Namespace, step: int) -> float:
    if args.lr_warmup_steps and step <= args.lr_warmup_steps:
        return step / args.lr_warmup_steps
    span = max(args.max_steps - args.lr_warmup_steps, 1)
    progress = min(max((step - args.lr_warmup_steps) / span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


def _gradient_report(module: torch.nn.Module) -> dict[str, float | int | bool]:
    sq = 0.0
    tensors = 0
    elements = 0
    nonfinite = 0
    nonzero = 0
    max_abs = 0.0
    for parameter in module.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        value = grad.detach().float()
        tensors += 1
        elements += value.numel()
        bad = ~torch.isfinite(value)
        nonfinite += int(bad.sum().item())
        finite = value.masked_fill(bad, 0.0)
        sq += float(finite.square().sum().item())
        nonzero += int((finite != 0).sum().item())
        if finite.numel():
            max_abs = max(max_abs, float(finite.abs().max().item()))
    return {
        "gradient_tensors": tensors,
        "gradient_elements": elements,
        "nonfinite_elements": nonfinite,
        "nonzero_elements": nonzero,
        "l2": math.sqrt(sq),
        "max_abs": max_abs,
        "finite": nonfinite == 0,
        "has_nonzero": nonzero > 0,
    }


def _allreduce_bool(value: bool, device: torch.device) -> bool:
    flag = torch.tensor([1 if value else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _save_snapshot(
    checkpoint: Path,
    *,
    transformer: WanTransformer3DModelPromptFreeNoTime,
    decoder: SlimReAEDecoder,
    runtime_dtype: torch.dtype,
    transformer_subfolder: str,
    metadata: Mapping[str, object],
) -> None:
    temp = checkpoint.with_name(checkpoint.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    transformer_dir = temp / transformer_subfolder
    transformer.save_config(str(transformer_dir))
    state = {
        name: tensor.detach().to(device="cpu", dtype=runtime_dtype).contiguous()
        for name, tensor in transformer.state_dict().items()
    }
    save_file(state, str(transformer_dir / "diffusion_pytorch_model.safetensors"))
    decoder.save_pretrained(temp / "tiny_decoder")
    _write_json(temp / "metadata.json", metadata)
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    temp.replace(checkpoint)


def _validate_rank0(
    closure: B2BJointForward,
    loader: DataLoader,
    cache: TeacherVelocityCache,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float | int]:
    closure.eval()
    velocity = DistillationMetricAccumulator()
    student_teacher = VideoMetricAccumulator()
    student_gt = VideoMetricAccumulator()
    teacher_gt = VideoMetricAccumulator()
    batches = 0
    autocast_enabled = device.type == "cuda" and dtype == torch.bfloat16
    try:
        with torch.inference_mode():
            for batch_cpu in loader:
                teacher_velocity = cache.load_batch(batch_cpu, device=device, dtype=dtype)
                batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype if autocast_enabled else torch.float32,
                    enabled=autocast_enabled,
                ):
                    output = closure(batch)
                    teacher_prediction = decode_teacher_prediction(
                        reae=closure.reae,
                        z_lq=output["z_lq"],
                        teacher_velocity=teacher_velocity,
                        output_frames=int(output["target"].shape[1]),
                    )
                velocity.update(output["velocity"], teacher_velocity)
                student_teacher.update(output["prediction"], teacher_prediction, clamp=True)
                student_gt.update(output["prediction"], output["target"], clamp=True)
                teacher_gt.update(teacher_prediction, output["target"], clamp=True)
                batches += 1
    finally:
        closure.train()
        closure.reae.eval()

    if batches == 0:
        raise RuntimeError("Validation produced no batches")
    result: dict[str, float | int] = {**velocity.compute(), "batches": batches}
    result.update({f"student_teacher_{key}": value for key, value in student_teacher.compute().items()})
    result.update({f"student_gt_{key}": value for key, value in student_gt.compute().items()})
    result.update({f"teacher_gt_{key}": value for key, value in teacher_gt.compute().items()})
    return result


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    rank, local_rank, world_size, device = stage_a.init_distributed()
    try:
        effective_batch = world_size * args.batch_size * args.gradient_accumulation_steps
        if args.expected_global_batch_size is not None and effective_batch != args.expected_global_batch_size:
            raise ValueError(
                f"Global effective batch={effective_batch}, expected={args.expected_global_batch_size}"
            )

        dtype = DTYPES[args.dtype]
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")
        seed_everything(args.seed + rank)

        run_dir = args.output_dir.expanduser().resolve()
        if rank == 0:
            if (run_dir / "train_log.jsonl").exists() or (run_dir / "latest.json").exists():
                raise FileExistsError("Output directory already contains a B2B-1B run")
            run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        train_cache = TeacherVelocityCache(args.teacher_cache)
        if train_cache.metadata.get("kind") != "swiftvr_b2a_stage_a_teacher_velocity":
            raise ValueError("B2B-1B requires Stage-A 200k teacher velocity cache")
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
                raise ValueError("Validation cache is not Stage-A teacher velocity")
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

        base_root = args.base_checkpoint.expanduser().resolve()
        student_root = args.student_init.expanduser().resolve()
        decoder_root, decoder_resolution = _resolve_student_root(args.decoder_init)
        reae = ReAE(str(base_root / args.reae_filename)).to(device=device, dtype=dtype).eval()
        transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
            str(student_root),
            subfolder=args.transformer_subfolder,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device=device, dtype=dtype)
        expected_shape = {
            "hidden_dim": B2B_TINY_SPEC.hidden_dim,
            "num_heads": B2B_TINY_SPEC.num_heads,
            "head_dim": B2B_TINY_SPEC.head_dim,
            "ffn_dim": B2B_TINY_SPEC.ffn_dim,
            "num_layers": B2B_TINY_SPEC.num_layers,
            "adapter_dim": B2B_TINY_SPEC.adapter_dim,
        }
        actual_shape = transformer_width_shape(transformer)
        if actual_shape != expected_shape:
            raise ValueError(f"B2B DiT shape mismatch: {actual_shape} != {expected_shape}")
        decoder = SlimReAEDecoder.from_pretrained(decoder_root, device=device, dtype=dtype)
        if tuple(decoder.channels) != B2B_EXTREME_DECODER_CHANNELS:
            raise ValueError("decoder-init is not the B2B extreme decoder")

        closure = B2BJointForward(
            reae,
            transformer,
            decoder,
            attention_backend=args.attention_backend,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        ).to(device=device)
        closure.train()
        cast_summary = cast_trainable_parameters(closure, dtype=torch.float32)

        dit_parameters = [parameter for parameter in closure.transformer.parameters() if parameter.requires_grad]
        decoder_parameters = [parameter for parameter in closure.decoder.parameters() if parameter.requires_grad]
        if not dit_parameters or not decoder_parameters:
            raise RuntimeError("B2B joint optimizer requires trainable DiT and decoder parameters")
        optimizer = torch.optim.AdamW(
            [
                {"params": dit_parameters, "lr": args.dit_learning_rate, "name": "dit"},
                {"params": decoder_parameters, "lr": args.decoder_learning_rate, "name": "decoder"},
            ],
            weight_decay=args.weight_decay,
            eps=args.optimizer_eps,
            foreach=False,
        )
        ddp_model = DistributedDataParallel(
            closure,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

        run_config = {
            "trainer": "swiftvr_b2b_joint_recovery_ddp_v1",
            "student_shape": actual_shape,
            "decoder_channels": list(decoder.channels),
            "decoder_resolution": decoder_resolution,
            "compute": b2b_compute_budget(),
            "world_size": world_size,
            "local_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_effective_batch_size": effective_batch,
            "dtype": args.dtype,
            "cast_trainable_parameters": cast_summary,
            "loss_weights": {
                "representation_mse": args.representation_mse_weight,
                "representation_cosine": args.representation_cosine_weight,
                "teacher_rgb_l1": args.teacher_rgb_l1_weight,
                "gt_rgb_l1": args.gt_rgb_l1_weight,
            },
            "dit_learning_rate": args.dit_learning_rate,
            "decoder_learning_rate": args.decoder_learning_rate,
            "dit_max_grad_norm": args.dit_max_grad_norm,
            "decoder_max_grad_norm": args.decoder_max_grad_norm,
            "lr_warmup_steps": args.lr_warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "max_steps": args.max_steps,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "checkpoint_contains": ["transformer", "tiny_decoder"],
        }
        if rank == 0:
            _write_json(run_dir / "run_config.json", run_config)
        dist.barrier()

        train_log = run_dir / "train_log.jsonl"
        val_log = run_dir / "val_log.jsonl"
        global_step = 0
        epoch = 0
        best_teacher_psnr = -float("inf")
        best_step: int | None = None

        def run_validation(step: int) -> dict[str, float | int] | None:
            nonlocal best_teacher_psnr, best_step
            if val_loader is None or val_cache is None:
                return None
            validation = _validate_rank0(
                closure,
                val_loader,
                val_cache,
                device=device,
                dtype=dtype,
            )
            append_jsonl(val_log, {"global_step": step, **validation})
            teacher_psnr = float(validation["student_teacher_psnr"])
            if teacher_psnr > best_teacher_psnr:
                best_teacher_psnr = teacher_psnr
                best_step = step
                _write_json(
                    run_dir / "best.json",
                    {
                        "global_step": step,
                        "student_teacher_psnr": teacher_psnr,
                        "student_gt_psnr": float(validation["student_gt_psnr"]),
                        "velocity_relative_l2": float(validation["velocity_relative_l2"]),
                        "note": "B2B-1B recovery best selected by Student->Teacher RGB PSNR.",
                    },
                )
            return validation

        # This flag must be identical on every rank.  val_loader is intentionally
        # rank-0-only, so deriving the flag from val_loader would make rank 0 enter
        # validation barriers while the other ranks enter DDP gradient all-reduce.
        validation_configured = bool(args.val_manifest and args.val_teacher_cache)
        if args.validate_at_start and validation_configured:
            dist.barrier()
            if rank == 0:
                baseline = run_validation(0)
                assert baseline is not None
                print(
                    f"B2B init teacher_psnr={baseline['student_teacher_psnr']:.4f} "
                    f"gt_psnr={baseline['student_gt_psnr']:.4f} "
                    f"rel_l2={baseline['velocity_relative_l2']:.6f} "
                    f"cos={baseline['velocity_cosine']:.6f}",
                    flush=True,
                )
            dist.barrier()

        autocast_enabled = device.type == "cuda" and dtype == torch.bfloat16
        started = time.perf_counter()
        while global_step < args.max_steps:
            loader = stage_a.make_train_loader(
                train_dataset,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                args=args,
            )
            iterator = iter(loader)
            while global_step < args.max_steps:
                micro_batches: list[Mapping[str, object]] = []
                for _ in range(args.gradient_accumulation_steps):
                    try:
                        micro_batches.append(next(iterator))
                    except StopIteration:
                        break
                if len(micro_batches) != args.gradient_accumulation_steps:
                    break

                next_step = global_step + 1
                lr_scale = _lr_scale(args, next_step)
                optimizer.param_groups[0]["lr"] = args.dit_learning_rate * lr_scale
                optimizer.param_groups[1]["lr"] = args.decoder_learning_rate * lr_scale
                optimizer.zero_grad(set_to_none=True)
                sums: dict[str, float] = {}
                step_started = time.perf_counter()

                for micro_index, batch_cpu in enumerate(micro_batches):
                    teacher_velocity = train_cache.load_batch(batch_cpu, device=device, dtype=dtype)
                    batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                    sync_context = (
                        nullcontext()
                        if micro_index == args.gradient_accumulation_steps - 1
                        else ddp_model.no_sync()
                    )
                    with sync_context:
                        with torch.autocast(
                            device_type=device.type,
                            dtype=dtype if autocast_enabled else torch.float32,
                            enabled=autocast_enabled,
                        ):
                            output = ddp_model(batch)
                            with torch.no_grad():
                                teacher_prediction = decode_teacher_prediction(
                                    reae=closure.reae,
                                    z_lq=output["z_lq"],
                                    teacher_velocity=teacher_velocity,
                                    output_frames=int(output["target"].shape[1]),
                                )
                        objective = b2b_joint_objective(
                            output["velocity"],
                            teacher_velocity,
                            output["prediction"],
                            teacher_prediction,
                            output["target"],
                            representation_mse_weight=args.representation_mse_weight,
                            representation_cosine_weight=args.representation_cosine_weight,
                            teacher_rgb_l1_weight=args.teacher_rgb_l1_weight,
                            gt_rgb_l1_weight=args.gt_rgb_l1_weight,
                            epsilon=args.loss_epsilon,
                        )
                        (objective["loss"] / args.gradient_accumulation_steps).backward()
                    for key in (
                        "loss",
                        "velocity_mse",
                        "velocity_normalized_mse",
                        "velocity_cosine",
                        "velocity_cosine_loss",
                        "teacher_velocity_power",
                        "output_l1",
                        "gt_student_pixel_l1",
                    ):
                        sums[key] = sums.get(key, 0.0) + float(objective[key].detach().item())

                dit_grad = _gradient_report(closure.transformer)
                decoder_grad = _gradient_report(closure.decoder)
                global_bad = _allreduce_bool(
                    (not bool(dit_grad["finite"])) or (not bool(decoder_grad["finite"])),
                    device,
                )
                if global_bad:
                    raise FloatingPointError(
                        f"B2B-1B non-finite gradient: dit={dit_grad}, decoder={decoder_grad}"
                    )
                if not dit_grad["has_nonzero"] or not decoder_grad["has_nonzero"]:
                    raise RuntimeError(
                        f"B2B-1B missing joint gradient branch: dit={dit_grad}, decoder={decoder_grad}"
                    )

                dit_preclip = float(dit_grad["l2"])
                decoder_preclip = float(decoder_grad["l2"])
                if args.dit_max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        dit_parameters,
                        max_norm=args.dit_max_grad_norm,
                        error_if_nonfinite=True,
                    )
                if args.decoder_max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        decoder_parameters,
                        max_norm=args.decoder_max_grad_norm,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                keys = tuple(sums)
                packed = torch.tensor(
                    [sums[key] for key in keys]
                    + [dit_preclip, decoder_preclip, time.perf_counter() - step_started],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                denominator = args.gradient_accumulation_steps * world_size
                averages = {
                    key: float(packed[index].item()) / denominator
                    for index, key in enumerate(keys)
                }
                record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    **averages,
                    "velocity_relative_l2": math.sqrt(
                        max(averages["velocity_mse"], 0.0)
                        / max(averages["teacher_velocity_power"], 1e-12)
                    ),
                    "dit_grad_l2_preclip": float(packed[-3].item()) / world_size,
                    "decoder_grad_l2_preclip": float(packed[-2].item()) / world_size,
                    "dit_learning_rate": optimizer.param_groups[0]["lr"],
                    "decoder_learning_rate": optimizer.param_groups[1]["lr"],
                    "step_seconds": float(packed[-1].item()) / world_size,
                    "peak_allocated_gb_per_rank": torch.cuda.max_memory_allocated(device) / 1024**3,
                }
                if rank == 0:
                    append_jsonl(train_log, record)
                    if global_step % args.log_every == 0:
                        print(
                            f"step={global_step} loss={record['loss']:.6f} "
                            f"rel_l2={record['velocity_relative_l2']:.6f} "
                            f"cos={record['velocity_cosine']:.6f} "
                            f"rgb_l1={record['output_l1']:.6f} "
                            f"gt_l1={record['gt_student_pixel_l1']:.6f} "
                            f"grad_dit={record['dit_grad_l2_preclip']:.3f} "
                            f"grad_dec={record['decoder_grad_l2_preclip']:.3f} "
                            f"time={record['step_seconds']:.2f}s",
                            flush=True,
                        )

                validation_due = validation_configured and args.validate_every > 0 and (
                    global_step % args.validate_every == 0 or global_step == args.max_steps
                )
                if validation_due:
                    dist.barrier()
                    if rank == 0:
                        validation = run_validation(global_step)
                        assert validation is not None
                        print(
                            f"validation step={global_step} "
                            f"teacher_psnr={validation['student_teacher_psnr']:.4f} "
                            f"gt_psnr={validation['student_gt_psnr']:.4f} "
                            f"rel_l2={validation['velocity_relative_l2']:.6f} "
                            f"cos={validation['velocity_cosine']:.6f}",
                            flush=True,
                        )
                    dist.barrier()

                save_due = global_step % args.save_every == 0 or global_step == args.max_steps
                if save_due:
                    dist.barrier()
                    if rank == 0:
                        checkpoint = run_dir / "checkpoints" / f"step_{global_step:08d}"
                        _save_snapshot(
                            checkpoint,
                            transformer=closure.transformer,
                            decoder=closure.decoder,
                            runtime_dtype=dtype,
                            transformer_subfolder=args.transformer_subfolder,
                            metadata={
                                "trainer": "swiftvr_b2b_joint_recovery_ddp_v1",
                                "global_step": global_step,
                                "best_student_teacher_psnr": best_teacher_psnr,
                                "best_validation_step": best_step,
                                "runtime_dtype": args.dtype,
                            },
                        )
                        write_latest_checkpoint(run_dir, checkpoint)
                        if best_step == global_step and (run_dir / "best.json").is_file():
                            best = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
                            best["checkpoint"] = str(checkpoint.relative_to(run_dir))
                            _write_json(run_dir / "best.json", best)
                        print(f"saved B2B joint snapshot: {checkpoint}", flush=True)
                    dist.barrier()
            epoch += 1

        if rank == 0:
            summary = {
                "status": "PASS",
                "global_step": global_step,
                "best_student_teacher_psnr": best_teacher_psnr,
                "best_validation_step": best_step,
                "elapsed_seconds": time.perf_counter() - started,
                "compute": b2b_compute_budget(),
            }
            _write_json(run_dir / "summary.json", summary)
            print("================ B2B-1B short recovery complete ================")
            print(f"Steps                         : {global_step}")
            print(f"Best Student->Teacher PSNR   : {best_teacher_psnr:.4f} dB @ {best_step}")
            print(f"Saved                         : {run_dir}")
            print("=================================================================")
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
