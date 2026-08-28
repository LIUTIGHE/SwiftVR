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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    p.add_argument("--val-teacher-cache", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--clip-length", type=int, default=13)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--val-crop-size", type=int, default=128)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--views-per-record", type=int, default=8)
    p.add_argument("--view-seed", type=int, default=20260805)
    p.add_argument("--val-views-per-record", type=int, default=1)
    p.add_argument("--val-view-seed", type=int, default=9000001)
    p.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    p.add_argument("--vertical-flip-probability", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--expected-global-batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--no-gradient-checkpointing", action="store_true")

    p.add_argument("--representation-mse-weight", type=float, default=0.05)
    p.add_argument("--representation-cosine-weight", type=float, default=0.05)
    p.add_argument("--teacher-rgb-l1-weight", type=float, default=1.0)
    p.add_argument("--gt-rgb-l1-weight", type=float, default=0.5)

    p.add_argument("--dit-learning-rate", type=float, default=2e-5)
    p.add_argument("--decoder-learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer-eps", type=float, default=1e-8)
    p.add_argument("--dit-max-grad-norm", type=float, default=1.0)
    p.add_argument("--decoder-max-grad-norm", type=float, default=1.0)
    p.add_argument("--lr-warmup-steps", type=int, default=50)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--validate-at-start", action="store_true")
    p.add_argument("--save-every", type=int, default=100)

    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    return p


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "log_every",
        "validate_every",
        "save_every",
    )
    bad = [name for name in positive_ints if int(getattr(args, name)) <= 0]
    if bad:
        raise ValueError(f"Arguments must be positive: {bad}")
    if args.num_workers != 0:
        raise ValueError("B2B-1B keeps --num-workers 0 for deterministic diagnosis")
    if args.lr_warmup_steps < 0 or args.lr_warmup_steps >= args.max_steps:
        raise ValueError("lr-warmup-steps must be in [0,max-steps)")
    if not 0.0 < args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio must be in (0,1]")
    for name in (
        "representation_mse_weight",
        "representation_cosine_weight",
        "teacher_rgb_l1_weight",
        "gt_rgb_l1_weight",
        "weight_decay",
        "dit_max_grad_norm",
        "decoder_max_grad_norm",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_','-')} must be non-negative")
    if args.dit_learning_rate <= 0 or args.decoder_learning_rate <= 0 or args.optimizer_eps <= 0:
        raise ValueError("learning rates and optimizer-eps must be positive")
    if (
        args.representation_mse_weight
        + args.representation_cosine_weight
        + args.teacher_rgb_l1_weight
        + args.gt_rgb_l1_weight
        <= 0
    ):
        raise ValueError("At least one B2B loss weight must be non-zero")


def _lr_multiplier(args: argparse.Namespace, step: int) -> float:
    if args.lr_warmup_steps and step <= args.lr_warmup_steps:
        return float(step) / float(args.lr_warmup_steps)
    span = max(args.max_steps - args.lr_warmup_steps, 1)
    progress = min(max((step - args.lr_warmup_steps) / span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(args.min_lr_ratio) + (1.0 - float(args.min_lr_ratio)) * cosine


def _gradient_report(parameters) -> dict[str, float | int | bool]:
    sq = 0.0
    max_abs = 0.0
    tensors = 0
    elements = 0
    nonfinite = 0
    nonzero = 0
    for parameter in parameters:
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
        if finite.numel():
            max_abs = max(max_abs, float(finite.abs().max().item()))
        nonzero += int((finite != 0).sum().item())
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


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _save_snapshot(
    closure: B2BJointForward,
    checkpoint: Path,
    *,
    runtime_dtype: torch.dtype,
    transformer_subfolder: str,
    metadata: Mapping[str, object],
) -> None:
    temp = checkpoint.with_name(checkpoint.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    transformer_dir = temp / transformer_subfolder
    transformer_dir.mkdir()
    closure.transformer.save_config(str(transformer_dir))
    transformer_state = {
        name: value.detach().to(device="cpu", dtype=runtime_dtype).contiguous()
        for name, value in closure.transformer.state_dict().items()
    }
    save_file(
        transformer_state,
        str(transformer_dir / "diffusion_pytorch_model.safetensors"),
    )
    closure.decoder.save_pretrained(temp / "tiny_decoder")
    _write_json(temp / "metadata.json", metadata)
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    temp.replace(checkpoint)


@torch.no_grad()
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
        raise RuntimeError("B2B validation produced no batches")
    result: dict[str, float | int] = {**velocity.compute(), "batches": batches}
    result.update({f"student_teacher_{k}": v for k, v in student_teacher.compute().items()})
    result.update({f"student_gt_{k}": v for k, v in student_gt.compute().items()})
    result.update({f"teacher_gt_{k}": v for k, v in teacher_gt.compute().items()})
    return result


def _build_optimizer(
    closure: B2BJointForward,
    *,
    dit_lr: float,
    decoder_lr: float,
    weight_decay: float,
    eps: float,
) -> torch.optim.AdamW:
    dit = [p for p in closure.transformer.parameters() if p.requires_grad]
    decoder = [p for p in closure.decoder.parameters() if p.requires_grad]
    bad = sorted({str(p.dtype) for p in dit + decoder if p.dtype != torch.float32})
    if bad:
        raise RuntimeError(f"B2B optimizer expects FP32 master parameters, found {bad}")
    return torch.optim.AdamW(
        [
            {"params": dit, "lr": float(dit_lr), "name": "dit"},
            {"params": decoder, "lr": float(decoder_lr), "name": "decoder"},
        ],
        weight_decay=float(weight_decay),
        eps=float(eps),
        foreach=False,
    )


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
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(f"Output directory is not empty: {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        train_cache = TeacherVelocityCache(args.teacher_cache)
        val_cache = TeacherVelocityCache(args.val_teacher_cache)
        for label, cache in (("train", train_cache), ("val", val_cache)):
            if cache.metadata.get("kind") != "swiftvr_b2a_stage_a_teacher_velocity":
                raise ValueError(f"{label} cache is not Stage-A B2-A teacher velocity")

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
        val_dataset = stage_a.build_cached_dataset(
            args.val_manifest,
            val_cache,
            split=args.val_split,
            path_root=args.path_root,
            clip_length=args.clip_length,
            crop_size=args.val_crop_size,
            scale=args.scale,
            views_per_record=args.val_views_per_record,
            view_seed=args.val_view_seed,
            hflip=0.0,
            vflip=0.0,
            verify_paths=args.verify_paths,
        )
        val_loader = None
        if rank == 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
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
        decoder = SlimReAEDecoder.from_pretrained(decoder_root, device=device, dtype=dtype)

        expected_shape = {
            "hidden_dim": B2B_TINY_SPEC.hidden_dim,
            "num_heads": B2B_TINY_SPEC.num_heads,
            "head_dim": B2B_TINY_SPEC.head_dim,
            "ffn_dim": B2B_TINY_SPEC.ffn_dim,
            "num_layers": B2B_TINY_SPEC.num_layers,
            "adapter_dim": B2B_TINY_SPEC.adapter_dim,
        }
        shape = transformer_width_shape(transformer)
        if shape != expected_shape:
            raise ValueError(f"B2B Transformer shape mismatch: {shape} != {expected_shape}")
        if tuple(decoder.channels) != B2B_EXTREME_DECODER_CHANNELS:
            raise ValueError(
                f"B2B decoder channels {decoder.channels} != {B2B_EXTREME_DECODER_CHANNELS}"
            )

        closure = B2BJointForward(
            reae,
            transformer,
            decoder,
            attention_backend=args.attention_backend,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        ).to(device=device)
        closure.train()
        cast_summary = cast_trainable_parameters(closure, dtype=torch.float32)
        optimizer = _build_optimizer(
            closure,
            dit_lr=args.dit_learning_rate,
            decoder_lr=args.decoder_learning_rate,
            weight_decay=args.weight_decay,
            eps=args.optimizer_eps,
        )
        ddp = DistributedDataParallel(
            closure,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

        run_config = {
            "trainer": "swiftvr_b2b_joint_recovery_ddp_v1",
            "base_checkpoint": str(base_root),
            "student_init": str(student_root),
            "decoder_init": decoder_resolution,
            "teacher_cache": str(args.teacher_cache.expanduser().resolve()),
            "val_teacher_cache": str(args.val_teacher_cache.expanduser().resolve()),
            "student_shape": shape,
            "decoder_channels": list(decoder.channels),
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
            "separate_gradient_clipping": True,
            "dit_max_grad_norm": args.dit_max_grad_norm,
            "decoder_max_grad_norm": args.decoder_max_grad_norm,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "best_selection": "max validation student_gt_psnr",
        }
        if rank == 0:
            _write_json(run_dir / "run_config.json", run_config)
        dist.barrier()

        train_log = run_dir / "train_log.jsonl"
        val_log = run_dir / "val_log.jsonl"
        best_gt_psnr = -float("inf")
        best_step = None
        global_step = 0
        epoch = 0
        started = time.perf_counter()

        def run_validation(step: int) -> dict[str, float | int]:
            nonlocal best_gt_psnr, best_step
            assert val_loader is not None
            validation = _validate_rank0(
                closure,
                val_loader,
                val_cache,
                device=device,
                dtype=dtype,
            )
            append_jsonl(val_log, {"global_step": step, **validation})
            gt_psnr = float(validation["student_gt_psnr"])
            if gt_psnr > best_gt_psnr:
                best_gt_psnr = gt_psnr
                best_step = step
                _write_json(
                    run_dir / "best.json",
                    {
                        "global_step": step,
                        "student_gt_psnr": gt_psnr,
                        "student_teacher_psnr": float(validation["student_teacher_psnr"]),
                        "velocity_relative_l2": float(validation["velocity_relative_l2"]),
                    },
                )
            return validation

        if args.validate_at_start:
            dist.barrier()
            if rank == 0:
                initial = run_validation(0)
                print(
                    f"B2B init teacher_psnr={initial['student_teacher_psnr']:.4f} "
                    f"gt_psnr={initial['student_gt_psnr']:.4f} "
                    f"rel_l2={initial['velocity_relative_l2']:.6f}",
                    flush=True,
                )
            dist.barrier()

        autocast_enabled = dtype == torch.bfloat16
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
                next_step = global_step + 1
                step_started = time.perf_counter()
                micro_batches = []
                for _ in range(args.gradient_accumulation_steps):
                    try:
                        micro_batches.append(next(iterator))
                    except StopIteration:
                        break
                if len(micro_batches) != args.gradient_accumulation_steps:
                    optimizer.zero_grad(set_to_none=True)
                    break

                lr_mult = _lr_multiplier(args, next_step)
                optimizer.param_groups[0]["lr"] = args.dit_learning_rate * lr_mult
                optimizer.param_groups[1]["lr"] = args.decoder_learning_rate * lr_mult
                optimizer.zero_grad(set_to_none=True)
                sums: dict[str, float] = {}

                for micro_index, batch_cpu in enumerate(micro_batches):
                    teacher_velocity = train_cache.load_batch(batch_cpu, device=device, dtype=dtype)
                    batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                    sync_context = (
                        ddp.no_sync()
                        if micro_index + 1 < args.gradient_accumulation_steps
                        else nullcontext()
                    )
                    with sync_context:
                        with torch.autocast(
                            device_type="cuda",
                            dtype=dtype if autocast_enabled else torch.float32,
                            enabled=autocast_enabled,
                        ):
                            output = ddp(batch)
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
                        sums[key] = sums.get(key, 0.0) + float(objective[key].detach().float().item())

                dit_params = [p for p in closure.transformer.parameters() if p.requires_grad]
                decoder_params = [p for p in closure.decoder.parameters() if p.requires_grad]
                dit_grad_before = _gradient_report(dit_params)
                dec_grad_before = _gradient_report(decoder_params)
                local_bad = int(not dit_grad_before["finite"] or not dec_grad_before["finite"])
                bad_flag = torch.tensor([local_bad], device=device, dtype=torch.int32)
                dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
                if int(bad_flag.item()):
                    raise FloatingPointError(
                        f"B2B non-finite gradients at step={next_step}: "
                        f"dit={dit_grad_before}, decoder={dec_grad_before}"
                    )
                if not dit_grad_before["has_nonzero"] or not dec_grad_before["has_nonzero"]:
                    raise RuntimeError(
                        f"B2B missing gradient branch at step={next_step}: "
                        f"dit={dit_grad_before}, decoder={dec_grad_before}"
                    )

                dit_clip_return = float(dit_grad_before["l2"])
                dec_clip_return = float(dec_grad_before["l2"])
                if args.dit_max_grad_norm > 0:
                    dit_clip_return = float(
                        torch.nn.utils.clip_grad_norm_(
                            dit_params,
                            max_norm=args.dit_max_grad_norm,
                            error_if_nonfinite=True,
                        ).float().item()
                    )
                if args.decoder_max_grad_norm > 0:
                    dec_clip_return = float(
                        torch.nn.utils.clip_grad_norm_(
                            decoder_params,
                            max_norm=args.decoder_max_grad_norm,
                            error_if_nonfinite=True,
                        ).float().item()
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step = next_step

                keys = tuple(sums)
                packed = torch.tensor(
                    [sums[key] for key in keys]
                    + [dit_clip_return, dec_clip_return, time.perf_counter() - step_started],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                denominator = args.gradient_accumulation_steps * world_size
                averages = {
                    key: float(packed[index].item()) / denominator
                    for index, key in enumerate(keys)
                }
                dit_grad_norm = float(packed[-3].item()) / world_size
                decoder_grad_norm = float(packed[-2].item()) / world_size
                step_seconds = float(packed[-1].item()) / world_size
                relative_l2 = math.sqrt(
                    max(averages["velocity_mse"], 0.0)
                    / max(averages["teacher_velocity_power"], 1e-12)
                )
                record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "global_effective_batch_size": effective_batch,
                    **averages,
                    "velocity_relative_l2": relative_l2,
                    "dit_grad_norm_preclip": dit_grad_norm,
                    "decoder_grad_norm_preclip": decoder_grad_norm,
                    "dit_learning_rate": optimizer.param_groups[0]["lr"],
                    "decoder_learning_rate": optimizer.param_groups[1]["lr"],
                    "step_seconds": step_seconds,
                    "peak_allocated_gb_per_rank": torch.cuda.max_memory_allocated(device) / 1024**3,
                }
                if rank == 0:
                    append_jsonl(train_log, record)
                    if global_step % args.log_every == 0:
                        print(
                            f"step={global_step} loss={averages['loss']:.6f} "
                            f"rel_l2={relative_l2:.4f} cos={averages['velocity_cosine']:.4f} "
                            f"teacher_l1={averages['output_l1']:.5f} "
                            f"gt_l1={averages['gt_student_pixel_l1']:.5f} "
                            f"grad_dit={dit_grad_norm:.3f} grad_dec={decoder_grad_norm:.3f} "
                            f"time={step_seconds:.2f}s",
                            flush=True,
                        )

                validation_due = global_step % args.validate_every == 0 or global_step == args.max_steps
                if validation_due:
                    dist.barrier()
                    if rank == 0:
                        validation = run_validation(global_step)
                        print(
                            f"validation step={global_step} "
                            f"teacher_psnr={validation['student_teacher_psnr']:.4f} "
                            f"gt_psnr={validation['student_gt_psnr']:.4f} "
                            f"teacher_gt={validation['teacher_gt_psnr']:.4f} "
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
                            closure,
                            checkpoint,
                            runtime_dtype=dtype,
                            transformer_subfolder=args.transformer_subfolder,
                            metadata={
                                "trainer": "swiftvr_b2b_joint_recovery_ddp_v1",
                                "global_step": global_step,
                                "runtime_dtype": args.dtype,
                                "student_shape": shape,
                                "decoder_channels": list(decoder.channels),
                                "compute": b2b_compute_budget(),
                                "best_validation_student_gt_psnr": best_gt_psnr,
                                "best_validation_step": best_step,
                            },
                        )
                        write_latest_checkpoint(run_dir, checkpoint)
                        if best_step == global_step and (run_dir / "best.json").is_file():
                            best = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
                            best["checkpoint"] = str(checkpoint.relative_to(run_dir))
                            _write_json(run_dir / "best.json", best)
                        print(f"saved B2B snapshot: {checkpoint}", flush=True)
                    dist.barrier()
            epoch += 1

        if rank == 0:
            summary = {
                "status": "PASS",
                "global_step": global_step,
                "best_validation_student_gt_psnr": best_gt_psnr,
                "best_validation_step": best_step,
                "elapsed_seconds": time.perf_counter() - started,
                "compute": b2b_compute_budget(),
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
