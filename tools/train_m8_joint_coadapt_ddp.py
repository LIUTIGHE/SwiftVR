#!/usr/bin/env python3
"""M8-C V2 decoder-aware joint co-adaptation for D1024/L20 MoE + Decoder76.

The frozen ReAE encoder supplies z_LQ and Stage-A D3072 cached velocity remains
the final-behavior teacher. V2 explicitly separates gradient responsibilities:

* Transformer recovery: velocity/latent anchors plus RGB losses through the
  frozen ORIGINAL ReAE decoder, comparing OriginalDecoder(z_M8) against
  OriginalDecoder(z_StageA). Gradients flow through z_M8 into the Transformer,
  never into Decoder76.
* Decoder76 recovery: Decoder76(z_M8.detach()) matches
  OriginalDecoder(z_M8.detach()) with the validated M8-B same-latent
  10x-L2 + LPIPS + temporal recipe. These gradients update Decoder76 only.

This prevents Decoder76 from being forced to compensate for the remaining M8
Transformer latent gap. GT is diagnostic only and never contributes gradients or
checkpoint selection.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools import train_teacher_distillation_ddp as stage_a
from tools.smoke_training_forward import move_video_batch, resolve_runtime_dtype, validate_folded_checkpoint
from swiftvr.models import ReAE, WanTransformer3DModelPromptFreeNoTimeMoE
from swiftvr.models import transformer as transformer_ops
from swiftvr.models.reae_slim_decoder import M8_DECODER76_CHANNELS, SlimReAEDecoder
from swiftvr.training import (
    DistillationMetricAccumulator,
    TeacherVelocityCache,
    VideoMetricAccumulator,
    append_jsonl,
    build_grad_scaler,
    cast_trainable_parameters,
    decode_reae_clip,
    decode_teacher_prediction,
    seed_everything,
    write_latest_checkpoint,
)
from swiftvr.training.b2b_moe import B2BMoESpec, expected_moe_shape, transformer_moe_shape
from swiftvr.training.b2b_moe_training import router_summary
from swiftvr.training.m8_joint import (
    M8JointDecoupledLossWeights,
    M8JointForward,
    m8_joint_decoupled_objective,
)
from swiftvr.training.tiny_decoder import LPIPSAlexLoss


STAGE_A_CACHE_KIND = "swiftvr_b2a_stage_a_teacher_velocity"
M8_SPEC = B2BMoESpec(num_layers=20)
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--transformer-init", type=Path, required=True)
    p.add_argument(
        "--decoder-init",
        type=Path,
        required=True,
        help="Decoder76 directory, typically warm-up checkpoint/tiny_decoder",
    )
    p.add_argument(
        "--teacher-cache",
        type=Path,
        required=True,
        help="Stage-A D3072 training velocity cache",
    )
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
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--dtype", choices=tuple(DTYPES), default="float16")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--no-gradient-checkpointing", action="store_true")

    p.add_argument("--tail-full-blocks", type=int, default=6)
    p.add_argument("--transformer-light-learning-rate", type=float, default=1e-6)
    p.add_argument("--transformer-tail-learning-rate", type=float, default=2e-6)
    p.add_argument("--decoder-learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer-eps", type=float, default=1e-8)
    p.add_argument("--transformer-max-grad-norm", type=float, default=1.0)
    p.add_argument("--decoder-max-grad-norm", type=float, default=1.0)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--expected-global-batch-size", type=int, default=64)
    p.add_argument("--lr-warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)

    # Transformer/representation anchors.
    p.add_argument("--velocity-nmse-weight", type=float, default=0.25)
    p.add_argument("--velocity-cosine-weight", type=float, default=0.25)
    p.add_argument("--latent-spatial-weight", type=float, default=0.5)
    p.add_argument("--latent-temporal-weight", type=float, default=0.5)

    # Frozen-original-decoder RGB system path -> Transformer only.
    p.add_argument("--teacher-rgb-l1-weight", type=float, default=1.0)
    p.add_argument("--teacher-lpips-weight", type=float, default=0.1)
    p.add_argument("--teacher-rgb-temporal-weight", type=float, default=1.0)

    # Same-latent compact-decoder recovery -> Decoder76 only.
    p.add_argument("--decoder-teacher-l2-weight", type=float, default=10.0)
    p.add_argument("--decoder-teacher-lpips-weight", type=float, default=0.1)
    p.add_argument("--decoder-teacher-temporal-weight", type=float, default=1.0)

    p.add_argument("--router-balance-weight", type=float, default=0.01)
    p.add_argument("--lpips-microbatch-frames", type=int, default=16)
    p.add_argument("--loss-epsilon", type=float, default=1e-8)

    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--validate-at-start", action="store_true")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    return p


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "log_every",
        "save_every",
        "transformer_light_learning_rate",
        "transformer_tail_learning_rate",
        "decoder_learning_rate",
        "optimizer_eps",
        "lpips_microbatch_frames",
        "loss_epsilon",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not 0 < args.tail_full_blocks <= M8_SPEC.num_layers:
        raise ValueError("--tail-full-blocks must lie in [1,20]")
    if args.lr_warmup_steps < 0 or args.lr_warmup_steps >= args.max_steps:
        raise ValueError("--lr-warmup-steps must lie in [0,max-steps)")
    if not 0 < args.min_lr_ratio <= 1:
        raise ValueError("--min-lr-ratio must lie in (0,1]")
    for name in (
        "velocity_nmse_weight",
        "velocity_cosine_weight",
        "latent_spatial_weight",
        "latent_temporal_weight",
        "teacher_rgb_l1_weight",
        "teacher_lpips_weight",
        "teacher_rgb_temporal_weight",
        "decoder_teacher_l2_weight",
        "decoder_teacher_lpips_weight",
        "decoder_teacher_temporal_weight",
        "router_balance_weight",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_','-')} must be non-negative")


def _weights(args: argparse.Namespace) -> M8JointDecoupledLossWeights:
    return M8JointDecoupledLossWeights(
        velocity_nmse=args.velocity_nmse_weight,
        velocity_cosine=args.velocity_cosine_weight,
        latent_spatial=args.latent_spatial_weight,
        latent_temporal=args.latent_temporal_weight,
        system_rgb_l1=args.teacher_rgb_l1_weight,
        system_lpips=args.teacher_lpips_weight,
        system_rgb_temporal=args.teacher_rgb_temporal_weight,
        decoder_teacher_l2=args.decoder_teacher_l2_weight,
        decoder_teacher_lpips=args.decoder_teacher_lpips_weight,
        decoder_teacher_temporal=args.decoder_teacher_temporal_weight,
        router_balance=args.router_balance_weight,
    )


def _configure_trainable_scope(transformer, decoder, tail_full_blocks: int):
    """Freeze the trunk, then open early adapter/router and a fully trainable tail."""
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    for parameter in decoder.parameters():
        parameter.requires_grad_(True)

    tail_start = len(transformer.blocks) - int(tail_full_blocks)
    light: list[torch.nn.Parameter] = []
    tail: list[torch.nn.Parameter] = []
    light_names: list[str] = []
    tail_names: list[str] = []
    for block_index, block in enumerate(transformer.blocks):
        for name, parameter in block.named_parameters():
            full_name = f"blocks.{block_index}.{name}"
            if block_index >= tail_start:
                parameter.requires_grad_(True)
                tail.append(parameter)
                tail_names.append(full_name)
            elif "prompt_free_adapter" in name or name.startswith("ffn.router"):
                parameter.requires_grad_(True)
                light.append(parameter)
                light_names.append(full_name)

    decoder_params = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    groups = {
        "transformer_light": light,
        "transformer_tail": tail,
        "decoder": decoder_params,
    }
    ids = [{id(parameter) for parameter in values} for values in groups.values()]
    if any(ids[i] & ids[j] for i in range(len(ids)) for j in range(i + 1, len(ids))):
        raise RuntimeError("M8 optimizer groups overlap")

    all_transformer_trainable = {
        id(parameter) for parameter in transformer.parameters() if parameter.requires_grad
    }
    expected_transformer_trainable = ids[0] | ids[1]
    if all_transformer_trainable != expected_transformer_trainable:
        raise RuntimeError("M8 transformer trainable scope is not covered exactly by optimizer groups")

    return groups, {
        "tail_start_block": tail_start,
        "tail_full_blocks": int(tail_full_blocks),
        "transformer_light_parameter_elements": sum(p.numel() for p in light),
        "transformer_tail_parameter_elements": sum(p.numel() for p in tail),
        "decoder_parameter_elements": sum(p.numel() for p in decoder_params),
        "transformer_light_names": light_names,
        "transformer_tail_names": tail_names,
    }


def _build_optimizer(groups, args):
    rates = {
        "transformer_light": float(args.transformer_light_learning_rate),
        "transformer_tail": float(args.transformer_tail_learning_rate),
        "decoder": float(args.decoder_learning_rate),
    }
    param_groups = []
    for name, parameters in groups.items():
        if not parameters:
            raise RuntimeError(f"empty optimizer group {name}")
        bad = {str(p.dtype) for p in parameters if p.dtype != torch.float32}
        if bad:
            raise RuntimeError(f"optimizer group {name} is not FP32: {sorted(bad)}")
        param_groups.append(
            {
                "params": parameters,
                "lr": rates[name],
                "base_lr": rates[name],
                "group_name": name,
            }
        )
    return torch.optim.AdamW(
        param_groups,
        weight_decay=float(args.weight_decay),
        eps=float(args.optimizer_eps),
        foreach=False,
    )


def _lr_scale(args, step: int) -> float:
    if args.lr_warmup_steps and step <= args.lr_warmup_steps:
        return step / args.lr_warmup_steps
    span = max(args.max_steps - args.lr_warmup_steps, 1)
    progress = min(max((step - args.lr_warmup_steps) / span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _clear_window_caches() -> None:
    transformer_ops._WindowIndexCache.clear()
    transformer_ops._WindowRuntimeMetaCache.clear()


def _save_snapshot(
    root: Path,
    *,
    transformer,
    decoder,
    metadata: Mapping[str, object],
    subfolder: str,
) -> None:
    temporary = root.with_name(root.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    transformer.save_pretrained(str(temporary / subfolder), safe_serialization=True)
    decoder.save_pretrained(temporary / "tiny_decoder")
    _write_json(temporary / "metadata.json", metadata)
    if root.exists():
        shutil.rmtree(root)
    temporary.replace(root)


def _decode_full_student(reae, z_student: torch.Tensor, output_frames: int) -> torch.Tensor:
    """Decode non-detached [B,C,F,H,W] student latent with frozen full ReAE."""
    return decode_reae_clip(
        reae,
        z_student.permute(0, 2, 1, 3, 4).contiguous(),
        output_frames=int(output_frames),
        clamp=False,
    )


def _validate_rank0(closure, loader, cache, perceptual, weights, *, device, dtype, args):
    _clear_window_caches()
    closure.eval()
    closure.reae.eval()
    velocity_accumulator = DistillationMetricAccumulator()
    student_teacher = VideoMetricAccumulator()
    full_student_teacher = VideoMetricAccumulator()
    decoder_same_latent = VideoMetricAccumulator()
    student_gt = VideoMetricAccumulator()
    teacher_gt = VideoMetricAccumulator()
    sums: dict[str, float] = {}
    samples = 0
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    try:
        with torch.no_grad():
            for batch_cpu in loader:
                teacher_velocity = cache.load_batch(batch_cpu, device=device, dtype=dtype)
                batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                    output = closure(batch)
                    full_student_prediction = _decode_full_student(
                        closure.reae,
                        output["z_student"],
                        int(output["target"].shape[1]),
                    )
                    teacher_prediction = decode_teacher_prediction(
                        reae=closure.reae,
                        z_lq=output["z_lq"],
                        teacher_velocity=teacher_velocity,
                        output_frames=int(output["target"].shape[1]),
                    )
                    objective = m8_joint_decoupled_objective(
                        student_velocity=output["velocity"],
                        teacher_velocity=teacher_velocity,
                        z_lq=output["z_lq"],
                        compact_prediction=output["prediction"],
                        full_student_prediction=full_student_prediction,
                        teacher_prediction=teacher_prediction,
                        router_balance_loss=output["router_balance_loss"],
                        perceptual=perceptual,
                        weights=weights,
                        lpips_microbatch_frames=args.lpips_microbatch_frames,
                        epsilon=args.loss_epsilon,
                    )
                batch_size = int(output["target"].shape[0])
                samples += batch_size
                for key, value in objective.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().float().item()) * batch_size
                velocity_accumulator.update(output["velocity"], teacher_velocity)
                student_teacher.update(output["prediction"], teacher_prediction, clamp=True)
                full_student_teacher.update(full_student_prediction, teacher_prediction, clamp=True)
                decoder_same_latent.update(output["prediction"], full_student_prediction, clamp=True)
                student_gt.update(output["prediction"], output["target"], clamp=True)
                teacher_gt.update(teacher_prediction, output["target"], clamp=True)
    finally:
        closure.train()
        closure.reae.eval()
        _clear_window_caches()

    result = {key: value / max(samples, 1) for key, value in sums.items()}
    result.update(velocity_accumulator.compute())
    for prefix, accumulator in (
        ("student_teacher", student_teacher),
        ("full_student_teacher", full_student_teacher),
        ("decoder_same_latent", decoder_same_latent),
        ("student_gt", student_gt),
        ("teacher_gt", teacher_gt),
    ):
        result.update({f"{prefix}_{key}": value for key, value in accumulator.compute().items()})
    result["samples"] = samples
    return result


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    rank, local_rank, world_size, device = stage_a.init_distributed()
    try:
        effective_batch = world_size * args.batch_size * args.gradient_accumulation_steps
        if (
            args.expected_global_batch_size is not None
            and effective_batch != args.expected_global_batch_size
        ):
            raise ValueError(
                f"global effective batch={effective_batch}, expected={args.expected_global_batch_size}"
            )
        dtype = DTYPES[args.dtype]
        seed_everything(args.seed + rank)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

        run_dir = args.output_dir.expanduser().resolve()
        if rank == 0:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        base_root = args.base_checkpoint.expanduser().resolve()
        transformer_root = args.transformer_init.expanduser().resolve()
        decoder_root = args.decoder_init.expanduser().resolve()
        folded_config = validate_folded_checkpoint(
            base_root,
            reae_filename=args.reae_filename,
            transformer_subfolder=args.transformer_subfolder,
        )
        dtype = resolve_runtime_dtype(
            args.dtype,
            folded_config,
            device,
            allow_mismatch=True,
        )

        train_cache = TeacherVelocityCache(args.teacher_cache)
        val_cache = TeacherVelocityCache(args.val_teacher_cache)
        if (
            train_cache.metadata.get("kind") != STAGE_A_CACHE_KIND
            or val_cache.metadata.get("kind") != STAGE_A_CACHE_KIND
        ):
            raise ValueError(
                "M8-C requires Stage-A D3072 velocity caches for both training and validation"
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
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=args.pin_memory,
            )
            if rank == 0
            else None
        )

        reae = ReAE(str(base_root / args.reae_filename))
        transformer = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
            str(transformer_root),
            subfolder=args.transformer_subfolder,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        shape = transformer_moe_shape(transformer)
        expected_shape = expected_moe_shape(M8_SPEC)
        if shape != expected_shape:
            raise ValueError(f"M8 transformer shape mismatch: {shape} != {expected_shape}")
        decoder = SlimReAEDecoder.from_pretrained(
            decoder_root,
            device="cpu",
            dtype=dtype,
        )
        if tuple(decoder.channels) != tuple(M8_DECODER76_CHANNELS):
            raise ValueError(
                f"M8 decoder must be {M8_DECODER76_CHANNELS}, got {decoder.channels}"
            )

        reae.to(device=device, dtype=dtype).eval()
        transformer.to(device=device, dtype=dtype)
        decoder.to(device=device, dtype=dtype)
        groups, scope_report = _configure_trainable_scope(
            transformer,
            decoder,
            args.tail_full_blocks,
        )
        closure = M8JointForward(
            reae,
            transformer,
            decoder,
            attention_backend=args.attention_backend,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        ).to(device=device)
        cast_report = cast_trainable_parameters(closure, dtype=torch.float32)
        groups, scope_report = _configure_trainable_scope(
            transformer,
            decoder,
            args.tail_full_blocks,
        )
        optimizer = _build_optimizer(groups, args)
        scaler = build_grad_scaler(device, dtype)
        perceptual = (
            LPIPSAlexLoss().to(device=device).eval()
            if args.teacher_lpips_weight > 0 or args.decoder_teacher_lpips_weight > 0
            else None
        )
        weights = _weights(args)
        ddp = DDP(
            closure,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

        run_config = {
            "trainer": "swiftvr_m8c_decoder_aware_joint_coadapt_ddp_v2_decoupled",
            "architecture": "m8-d1024-l20+decoder76",
            "gradient_roles": {
                "transformer": "stage_a_velocity_latent_plus_frozen_original_decoder_rgb",
                "decoder76": "same_latent_original_decoder_teacher_only",
            },
            "transformer_init": str(transformer_root),
            "decoder_init": str(decoder_root),
            "transformer_shape": shape,
            "decoder_channels": list(decoder.channels),
            "training_teacher": "stage_a_d3072_reference",
            "teacher_cache": str(args.teacher_cache.expanduser().resolve()),
            "validation_teacher_cache": str(args.val_teacher_cache.expanduser().resolve()),
            "gt_role": "diagnostic_only",
            "checkpoint_selection": "lowest_joint_teacher_validation_loss_with_safe_velocity_postfilter",
            "loss_weights": vars(weights),
            "trainable_scope": scope_report,
            "trainable_cast": cast_report,
            "world_size": world_size,
            "local_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_effective_batch_size": effective_batch,
            "max_steps": args.max_steps,
        }
        if rank == 0:
            _write_json(run_dir / "run_config.json", run_config)
        dist.barrier()

        best_loss = float("inf")
        best_step = None
        global_step = 0
        epoch = 0
        train_log = run_dir / "train_log.jsonl"
        val_log = run_dir / "val_log.jsonl"

        def validate(step: int):
            nonlocal best_loss, best_step
            if rank != 0 or val_loader is None:
                return None
            validation = _validate_rank0(
                closure,
                val_loader,
                val_cache,
                perceptual,
                weights,
                device=device,
                dtype=dtype,
                args=args,
            )
            append_jsonl(val_log, {"global_step": step, **validation})
            value = float(validation["loss"])
            if value < best_loss:
                best_loss = value
                best_step = step
                _write_json(
                    run_dir / "best.json",
                    {
                        "global_step": step,
                        "joint_teacher_validation_loss": value,
                        "velocity_relative_l2": validation["velocity_relative_l2"],
                        "student_teacher_psnr": validation["student_teacher_psnr"],
                        "student_teacher_ssim": validation["student_teacher_ssim"],
                        "full_student_teacher_psnr": validation["full_student_teacher_psnr"],
                        "decoder_same_latent_psnr": validation["decoder_same_latent_psnr"],
                        "note": "GT metrics are diagnostic only.",
                    },
                )
            print(
                json.dumps(
                    {"phase": "m8_joint_val_v2", "global_step": step, **validation},
                    sort_keys=True,
                ),
                flush=True,
            )
            return validation

        if args.validate_at_start:
            dist.barrier()
            validate(0)
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
                raise RuntimeError("per-rank epoch is shorter than gradient accumulation")
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
                learning_rate_scale = _lr_scale(args, next_step)
                for group in optimizer.param_groups:
                    group["lr"] = float(group["base_lr"]) * learning_rate_scale
                optimizer.zero_grad(set_to_none=True)
                sums: dict[str, float] = {}
                started = time.perf_counter()

                for micro_index, batch_cpu in enumerate(micro_batches):
                    teacher_velocity = train_cache.load_batch(
                        batch_cpu,
                        device=device,
                        dtype=dtype,
                    )
                    batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                    synchronize = micro_index + 1 == len(micro_batches)
                    synchronization_context = nullcontext() if synchronize else ddp.no_sync()
                    with synchronization_context:
                        with torch.autocast(
                            "cuda",
                            dtype=dtype,
                            enabled=autocast_enabled,
                        ):
                            output = ddp(batch)

                            # Frozen full decoder on non-detached M8 latent: this
                            # path gives decoder-aware RGB gradients to Transformer.
                            full_student_prediction = _decode_full_student(
                                reae,
                                output["z_student"],
                                int(output["target"].shape[1]),
                            )

                            # Stage-A final-behavior RGB target is detached.
                            with torch.no_grad():
                                teacher_prediction = decode_teacher_prediction(
                                    reae=reae,
                                    z_lq=output["z_lq"],
                                    teacher_velocity=teacher_velocity,
                                    output_frames=int(output["target"].shape[1]),
                                )

                            objective = m8_joint_decoupled_objective(
                                student_velocity=output["velocity"],
                                teacher_velocity=teacher_velocity,
                                z_lq=output["z_lq"],
                                compact_prediction=output["prediction"],
                                full_student_prediction=full_student_prediction,
                                teacher_prediction=teacher_prediction,
                                router_balance_loss=output["router_balance_loss"],
                                perceptual=perceptual,
                                weights=weights,
                                lpips_microbatch_frames=args.lpips_microbatch_frames,
                                epsilon=args.loss_epsilon,
                            )
                            scaled_loss = objective["loss"] / args.gradient_accumulation_steps
                        if not torch.isfinite(scaled_loss.detach()).item():
                            raise FloatingPointError("non-finite M8 joint V2 loss")
                        if scaler.is_enabled():
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()

                    for key, value in objective.items():
                        sums[key] = (
                            sums.get(key, 0.0)
                            + float(value.detach().float().item()) / len(micro_batches)
                        )

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                transformer_parameters = (
                    groups["transformer_light"] + groups["transformer_tail"]
                )
                decoder_parameters = groups["decoder"]
                transformer_grad_norm = torch.nn.utils.clip_grad_norm_(
                    transformer_parameters,
                    args.transformer_max_grad_norm,
                )
                decoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                    decoder_parameters,
                    args.decoder_max_grad_norm,
                )
                if not torch.isfinite(transformer_grad_norm) or not torch.isfinite(decoder_grad_norm):
                    raise FloatingPointError(
                        "non-finite M8 joint V2 grad norm: "
                        f"transformer={transformer_grad_norm}, decoder={decoder_grad_norm}"
                    )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                global_step = next_step

                if rank == 0 and (global_step % args.log_every == 0 or global_step == 1):
                    route = router_summary(transformer)
                    record = {
                        "global_step": global_step,
                        "seconds": time.perf_counter() - started,
                        **sums,
                        "transformer_grad_norm": float(transformer_grad_norm.detach().item()),
                        "decoder_grad_norm": float(decoder_grad_norm.detach().item()),
                        "router_normalized_entropy": route["normalized_entropy"],
                        "router_load_cv": route["load_cv"],
                        "lr_transformer_light": optimizer.param_groups[0]["lr"],
                        "lr_transformer_tail": optimizer.param_groups[1]["lr"],
                        "lr_decoder": optimizer.param_groups[2]["lr"],
                    }
                    append_jsonl(train_log, record)
                    print(json.dumps(record, sort_keys=True), flush=True)

                if args.validate_every > 0 and global_step % args.validate_every == 0:
                    dist.barrier()
                    validate(global_step)
                    dist.barrier()

                if global_step % args.save_every == 0 or global_step == args.max_steps:
                    dist.barrier()
                    if rank == 0:
                        checkpoint = run_dir / "checkpoints" / f"step_{global_step:08d}"
                        _save_snapshot(
                            checkpoint,
                            transformer=transformer,
                            decoder=decoder,
                            subfolder=args.transformer_subfolder,
                            metadata={
                                "global_step": global_step,
                                "architecture": "m8-d1024-l20+decoder76",
                                "trainer": "m8c_v2_decoupled",
                                "best_step_so_far": best_step,
                                "best_joint_val_loss": best_loss,
                            },
                        )
                        write_latest_checkpoint(run_dir, checkpoint)
                    dist.barrier()
            epoch += 1

        if rank == 0:
            _write_json(
                run_dir / "summary.json",
                {
                    "status": "PASS",
                    "trainer": "m8c_v2_decoupled",
                    "global_step": global_step,
                    "best_step": best_step,
                    "best_joint_val_loss": best_loss,
                },
            )
        return 0
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
