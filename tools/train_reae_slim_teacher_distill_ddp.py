#!/usr/bin/env python3
"""Teacher-only distillation for structurally slimmed ReAE decoders.

Two initial targets are supported:

* slim100:    (256,128,64,64), analytically ~98.223 GMAC/output-frame;
* aggressive: (256,128,64,32), analytically ~86.792 GMAC/output-frame.

The original ReAE topology is retained exactly.  On a fresh run rank 0 measures
activation RMS on cached Stage-A z_SR latents, selects the top-k channels for each
ReAE stage, broadcasts the selected indices, and every rank materializes the same
structured teacher subnetwork.  Training then uses only frozen-ReAE behavior as
the optimization target; GT is reported as a secondary diagnostic but never enters
the loss.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_tiny_decoder_formal_ddp as formal
from swiftvr.models import ReAE
from swiftvr.models.reae_slim_decoder import (
    STAGE_SCORE_LAYER_INDICES,
    VARIANT_CHANNELS,
    SlimReAEDecoder,
    topk_stage_indices,
)
from swiftvr.training import build_grad_scaler, cast_trainable_parameters
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.reference import sha256_file
from swiftvr.training.stage3 import VideoMetricAccumulator, temporal_difference_mse
from swiftvr.training.tiny_decoder import LPIPSAlexLoss
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


TRAINER_ID = "swiftvr_stage_b1_reae_slim_teacher_distill_ddp_v1"
VARIANT_GMAC = {"slim100": 98.2228992, "aggressive": 86.79211008}


def build_parser() -> argparse.ArgumentParser:
    parser = formal.build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "init_decoder":
            action.required = False
            action.default = None
            action.help = argparse.SUPPRESS
    parser.add_argument("--variant", choices=tuple(VARIANT_CHANNELS), required=True)
    parser.add_argument("--prune-calibration-samples", type=int, default=64)
    parser.add_argument("--teacher-l2-weight", type=float, default=10.0)
    parser.add_argument("--teacher-lpips-weight", type=float, default=0.1)
    parser.add_argument("--teacher-temporal-weight", type=float, default=1.0)
    parser.set_defaults(learning_rate=3e-5)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    formal._validate_args(args)
    if args.variant not in VARIANT_CHANNELS:
        raise ValueError(f"variant must be one of {sorted(VARIANT_CHANNELS)}")
    if int(args.prune_calibration_samples) <= 0:
        raise ValueError("prune-calibration-samples must be positive")
    for name in ("teacher_l2_weight", "teacher_lpips_weight", "teacher_temporal_weight"):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"{name.replace('_','-')} must be non-negative")


def _teacher_objective(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    *,
    perceptual: LPIPSAlexLoss | None,
    l2_weight: float,
    lpips_weight: float,
    temporal_weight: float,
    lpips_microbatch_frames: int,
) -> dict[str, torch.Tensor]:
    pred_f = prediction.float()
    teacher_f = teacher.detach().float()
    teacher_l2 = F.mse_loss(pred_f, teacher_f)
    teacher_temporal = temporal_difference_mse(pred_f, teacher_f)
    teacher_lpips = teacher_l2.new_zeros(())
    if lpips_weight > 0:
        if perceptual is None:
            raise ValueError("positive teacher LPIPS weight requires LPIPSAlexLoss")
        teacher_lpips = perceptual.forward_video(
            prediction,
            teacher,
            microbatch_frames=lpips_microbatch_frames,
        )
    loss = (
        float(l2_weight) * teacher_l2
        + float(lpips_weight) * teacher_lpips
        + float(temporal_weight) * teacher_temporal
    )
    return {
        "loss": loss,
        "teacher_l2": teacher_l2,
        "teacher_lpips": teacher_lpips,
        "teacher_temporal_mse": teacher_temporal,
    }


def _calibrate_activation_scores(
    teacher: ReAE,
    cache: TinyDecoderLatentCache,
    dataset,
    *,
    samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    samples = min(int(samples), len(dataset))
    loader = DataLoader(
        Subset(dataset, list(range(samples))),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    sums: list[torch.Tensor | None] = [None, None, None, None]
    counts = [0, 0, 0, 0]
    handles = []

    def hook_for(stage: int):
        def _hook(_module, _inputs, output):
            value = output.detach().float()
            if value.ndim != 4:
                raise RuntimeError(f"stage{stage} calibration output must be NCHW")
            score = value.square().sum(dim=(0, 2, 3)).cpu()
            sums[stage] = score if sums[stage] is None else sums[stage] + score
            counts[stage] += int(value.shape[0] * value.shape[2] * value.shape[3])
        return _hook

    # Aggregate all MemBlock outputs in the first three stages and the final
    # stage-3 ReLU.  This avoids selecting channels from only one block state.
    for stage, layer_indices in STAGE_SCORE_LAYER_INDICES.items():
        for layer_index in layer_indices:
            handles.append(teacher.decoder[layer_index].register_forward_hook(hook_for(stage)))

    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    try:
        with torch.no_grad():
            for batch_cpu in loader:
                target_value = batch_cpu.get("hr")
                if not isinstance(target_value, torch.Tensor):
                    raise TypeError("calibration batch is missing hr")
                output_frames = int(target_value.shape[1])
                z_sr = cache.load_batch(batch_cpu, device=device, dtype=dtype)
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype if autocast_enabled else torch.float32,
                    enabled=autocast_enabled,
                ):
                    decode_reae_clip(
                        teacher,
                        z_sr,
                        output_frames=output_frames,
                        clamp=False,
                    )
    finally:
        for handle in handles:
            handle.remove()

    rms: list[torch.Tensor] = []
    report: dict[str, object] = {"samples": samples, "stages": []}
    for stage in range(4):
        if sums[stage] is None or counts[stage] <= 0:
            raise RuntimeError(f"no activation statistics collected for stage{stage}")
        values = (sums[stage] / float(counts[stage])).sqrt()
        rms.append(values)
        report["stages"].append(
            {
                "stage": stage,
                "channels": int(values.numel()),
                "min_rms": float(values.min().item()),
                "mean_rms": float(values.mean().item()),
                "max_rms": float(values.max().item()),
            }
        )
    return rms, report


def _fingerprint(
    args: argparse.Namespace,
    *,
    world_size: int,
    train_cache: TinyDecoderLatentCache,
    val_cache: TinyDecoderLatentCache,
    stage_indices: Sequence[Sequence[int]],
) -> dict[str, object]:
    base = args.base_checkpoint.expanduser().resolve()
    return {
        "trainer": TRAINER_ID,
        "variant": args.variant,
        "student_channels": list(VARIANT_CHANNELS[args.variant]),
        "estimated_gmac_per_output_frame_1920x1088": VARIANT_GMAC[args.variant],
        "pruning_scheme": "activation_rms_topk_structured_stage_subset_v1",
        "stage_indices": [list(int(v) for v in values) for values in stage_indices],
        "prune_calibration_samples": int(args.prune_calibration_samples),
        "world_size": int(world_size),
        "base_checkpoint": str(base),
        "reae_sha256": sha256_file(base / args.reae_filename),
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
        "teacher_l2_weight": float(args.teacher_l2_weight),
        "teacher_lpips_weight": float(args.teacher_lpips_weight),
        "teacher_temporal_weight": float(args.teacher_temporal_weight),
        "lpips_microbatch_frames": int(args.lpips_microbatch_frames),
        "seed": int(args.seed),
        "gt_optimization_weight": 0.0,
    }


def _validate(
    student: SlimReAEDecoder,
    teacher: ReAE,
    cache: TinyDecoderLatentCache,
    loader: DataLoader,
    perceptual: LPIPSAlexLoss | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    student.eval()
    teacher.eval()
    sums = {"loss": 0.0, "teacher_l2": 0.0, "teacher_lpips": 0.0, "teacher_temporal_mse": 0.0}
    count = 0
    student_teacher = VideoMetricAccumulator()
    student_gt = VideoMetricAccumulator()
    teacher_gt = VideoMetricAccumulator()
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    for batch_cpu in loader:
        moved = formal._move_pixels(batch_cpu, device, dtype)
        prepared = prepare_training_batch(moved)
        target = prepared["target"]
        if not isinstance(target, torch.Tensor):
            raise TypeError("validation batch is missing target")
        z_sr = cache.load_batch(batch_cpu, device=device, dtype=dtype)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=dtype if autocast_enabled else torch.float32,
            enabled=autocast_enabled,
        ):
            teacher_rgb = decode_reae_clip(
                teacher, z_sr, output_frames=int(target.shape[1]), clamp=False
            )
            prediction = student(z_sr, output_frames=int(target.shape[1]), clamp=False)
        with torch.no_grad():
            objective = _teacher_objective(
                prediction,
                teacher_rgb,
                perceptual=perceptual,
                l2_weight=args.teacher_l2_weight,
                lpips_weight=args.teacher_lpips_weight,
                temporal_weight=args.teacher_temporal_weight,
                lpips_microbatch_frames=args.lpips_microbatch_frames,
            )
        batch_size = int(target.shape[0])
        count += batch_size
        for key in sums:
            sums[key] += float(objective[key].item()) * batch_size
        student_teacher.update(prediction, teacher_rgb, clamp=True)
        student_gt.update(prediction, target, clamp=True)
        teacher_gt.update(teacher_rgb, target, clamp=True)

    if count <= 0:
        raise RuntimeError("validation loader is empty")
    result: dict[str, float | int] = {key: value / count for key, value in sums.items()}
    result["samples"] = count
    for prefix, accumulator in (
        ("student_teacher", student_teacher),
        ("student_gt", student_gt),
        ("reae_teacher_gt", teacher_gt),
    ):
        for key, value in accumulator.compute().items():
            result[f"{prefix}_{key}"] = value
    return result


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    rank, local_rank, world_size, device = formal._init_distributed(args.ddp_timeout_seconds)
    try:
        formal._seed(args.seed + rank)
        dtype = formal.DTYPES[args.dtype]
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

        run_dir = args.output_dir.expanduser().resolve()
        path_root = args.path_root.expanduser().resolve()
        base = args.base_checkpoint.expanduser().resolve()
        train_cache = TinyDecoderLatentCache(args.train_cache)
        val_cache = TinyDecoderLatentCache(args.val_cache)
        formal._validate_cache_pair(train_cache, val_cache)

        train_dataset = formal._cache_subset(
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
        val_dataset = formal._cache_subset(
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
        train_loader = formal._train_loader(train_dataset, sampler, args)
        val_loader = formal._val_loader(val_dataset, args) if rank == 0 else None

        teacher = ReAE(str(base / args.reae_filename)).to(device=device, dtype=dtype).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        resume_checkpoint: Path | None = None
        calibration_report: dict[str, object] | None = None
        if args.resume is not None:
            resume_checkpoint = formal._resolve_resume(run_dir, args.resume) if rank == 0 else None
            payload = [str(resume_checkpoint) if rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            resume_checkpoint = Path(str(payload[0])).resolve()
            student = SlimReAEDecoder.from_pretrained(
                resume_checkpoint / "tiny_decoder", device=device, dtype=dtype
            )
            metadata = student.pruning_metadata
            stage_indices = tuple(tuple(int(v) for v in values) for values in metadata["stage_indices"])
        else:
            if rank == 0:
                scores, calibration_report = _calibrate_activation_scores(
                    teacher,
                    train_cache,
                    train_dataset,
                    samples=args.prune_calibration_samples,
                    device=device,
                    dtype=dtype,
                )
                selected = topk_stage_indices(scores, VARIANT_CHANNELS[args.variant])
                payload = [[list(values) for values in selected]]
            else:
                payload = [None]
            dist.broadcast_object_list(payload, src=0)
            stage_indices = tuple(tuple(int(v) for v in values) for values in payload[0])
            student = SlimReAEDecoder(
                channels=VARIANT_CHANNELS[args.variant],
                latent_channels=48,
                patch_size=2,
                frames_to_trim=3,
            ).to(device=device, dtype=dtype)
            student.initialize_from_reae(
                teacher, stage_indices, score_method="activation_rms"
            )

        if tuple(student.channels) != tuple(VARIANT_CHANNELS[args.variant]):
            raise ValueError(
                f"checkpoint channels={student.channels} do not match variant {args.variant}"
            )

        cast_report = cast_trainable_parameters(student, dtype=torch.float32)
        student.train()
        optimizer = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            eps=float(args.optimizer_eps),
            foreach=False,
        )
        scaler = build_grad_scaler(device, dtype)
        perceptual = (
            LPIPSAlexLoss().to(device=device).eval()
            if args.teacher_lpips_weight > 0
            else None
        )

        fingerprint = _fingerprint(
            args,
            world_size=world_size,
            train_cache=train_cache,
            val_cache=val_cache,
            stage_indices=stage_indices,
        )
        start_epoch = 0
        global_step = 0
        best_val_loss = math.inf
        if resume_checkpoint is not None:
            state = formal._load_training_state(resume_checkpoint)
            saved = state.get("fingerprint")
            if not isinstance(saved, Mapping):
                raise TypeError("resume checkpoint has no valid fingerprint")
            formal._assert_fingerprint(saved, fingerprint)
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_epoch = int(state["completed_epoch"])
            global_step = int(state["global_step"])
            best_val_loss = float(state.get("best_val_loss", math.inf))

        ddp = DDP(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        autocast_enabled = dtype in (torch.float16, torch.bfloat16)

        if rank == 0:
            if args.resume is None:
                if run_dir.exists() and any(run_dir.iterdir()):
                    raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
                run_dir.mkdir(parents=True, exist_ok=True)
                formal._write_json(
                    run_dir / formal.RUN_CONFIG_FILENAME,
                    {
                        **fingerprint,
                        "epochs": int(args.epochs),
                        "train_samples": len(train_dataset),
                        "val_samples": len(val_dataset),
                        "steps_per_epoch_per_rank": len(train_loader),
                        "student_parameters": sum(p.numel() for p in student.parameters()),
                        "trainable_cast": cast_report,
                        "calibration": calibration_report,
                    },
                )
            elif not run_dir.is_dir():
                raise FileNotFoundError(run_dir)
        dist.barrier()

        if args.resume is None:
            if rank == 0:
                assert val_loader is not None
                initial_val = _validate(
                    student,
                    teacher,
                    val_cache,
                    val_loader,
                    perceptual,
                    device=device,
                    dtype=dtype,
                    args=args,
                )
                formal._write_json(run_dir / "validation_epoch_000.json", initial_val)
                formal._append_jsonl(
                    run_dir / "val_log.jsonl",
                    {"completed_epoch": 0, "global_step": 0, **initial_val},
                )
                print(
                    json.dumps(
                        {"phase": "initial_reae_slim_validation", "variant": args.variant, **initial_val},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            dist.barrier()

        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            ddp.train()
            interval = {"loss": 0.0, "teacher_l2": 0.0, "teacher_lpips": 0.0, "teacher_temporal_mse": 0.0}
            interval_count = 0
            interval_grad = 0.0
            started = time.perf_counter()

            for batch_index, batch_cpu in enumerate(train_loader, start=1):
                moved = formal._move_pixels(batch_cpu, device, dtype)
                prepared = prepare_training_batch(moved)
                target = prepared["target"]
                if not isinstance(target, torch.Tensor):
                    raise TypeError("training batch is missing target")
                z_sr = train_cache.load_batch(batch_cpu, device=device, dtype=dtype)

                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=dtype, enabled=autocast_enabled
                ):
                    teacher_rgb = decode_reae_clip(
                        teacher, z_sr, output_frames=int(target.shape[1]), clamp=False
                    )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                    prediction = ddp(z_sr, output_frames=int(target.shape[1]), clamp=False)
                objective = _teacher_objective(
                    prediction,
                    teacher_rgb,
                    perceptual=perceptual,
                    l2_weight=args.teacher_l2_weight,
                    lpips_weight=args.teacher_lpips_weight,
                    temporal_weight=args.teacher_temporal_weight,
                    lpips_microbatch_frames=args.lpips_microbatch_frames,
                )
                scaler.scale(objective["loss"]).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), args.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"non-finite grad norm epoch={epoch+1} batch={batch_index}: {grad_norm}"
                    )
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

                batch_size = int(target.shape[0])
                interval_count += batch_size
                interval_grad += float(grad_norm.detach().item()) * batch_size
                for key in interval:
                    interval[key] += float(objective[key].detach().item()) * batch_size

                if batch_index % args.log_every == 0 or batch_index == len(train_loader):
                    elapsed = time.perf_counter() - started
                    reduced = formal._allreduce_interval(
                        interval,
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
                            "variant": args.variant,
                            "global_batch_size": args.batch_size * world_size,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            **reduced,
                        }
                        formal._append_jsonl(run_dir / "train_log.jsonl", record)
                        print(json.dumps(record, sort_keys=True), flush=True)
                    interval = {key: 0.0 for key in interval}
                    interval_count = 0
                    interval_grad = 0.0
                    started = time.perf_counter()

            dist.barrier()
            if rank == 0:
                assert val_loader is not None
                validation = _validate(
                    student,
                    teacher,
                    val_cache,
                    val_loader,
                    perceptual,
                    device=device,
                    dtype=dtype,
                    args=args,
                )
                completed_epoch = epoch + 1
                record = {"completed_epoch": completed_epoch, "global_step": global_step, **validation}
                formal._append_jsonl(run_dir / "val_log.jsonl", record)
                formal._write_json(
                    run_dir / f"validation_epoch_{completed_epoch:03d}.json", validation
                )
                current_loss = float(validation["loss"])
                is_best = current_loss < best_val_loss
                if is_best:
                    best_val_loss = current_loss
                checkpoint = formal._save_checkpoint(
                    run_dir,
                    model=student,
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
                            "variant": args.variant,
                            "checkpoint": str(checkpoint),
                            "is_best": is_best,
                            **record,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            payload = [best_val_loss if rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            best_val_loss = float(payload[0])
            dist.barrier()

        if rank == 0:
            best = json.loads((run_dir / formal.BEST_FILENAME).read_text(encoding="utf-8"))
            summary = {
                "status": "PASS",
                "variant": args.variant,
                "student_channels": list(VARIANT_CHANNELS[args.variant]),
                "estimated_gmac_per_output_frame_1920x1088": VARIANT_GMAC[args.variant],
                "completed_epochs": int(args.epochs),
                "global_step": int(global_step),
                "world_size": world_size,
                "local_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "teacher_l2_weight": float(args.teacher_l2_weight),
                "teacher_lpips_weight": float(args.teacher_lpips_weight),
                "teacher_temporal_weight": float(args.teacher_temporal_weight),
                "best": best,
            }
            formal._write_json(run_dir / "summary.json", summary)
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        dist.barrier()
        return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
