#!/usr/bin/env python3
"""Stage-B1 resize-conv tail recovery with an explicit residual p8 phase loss.

This experiment is intentionally isolated from the verified tail-recovery trainer.
It uses exactly the same trainable scope and optimizer groups as
``train_tiny_decoder_resize_conv_tail_recovery_ddp.py``:

    decoder stage2 blocks
    stage2 -> stage3 transition
    decoder stage3 blocks
    resize-conv RGB head

The only optimization change is an additional penalty on the visible
Tiny-minus-ReAE residual.  The penalty measures the per-frame luma period-8 phase
map, removes its nested period-4 parent, and minimizes the remaining p8 RMS.  It
therefore targets the large-period artifact diagnosed after PixelShuffle removal
without suppressing all image high frequencies.

If --phase-loss-weight is omitted, one deterministic training batch is used before
DDP wrapping to calibrate lambda so the RMS rank-gradient norm of lambda*L_phase is
--phase-gradient-target-ratio (default 0.1) times the base MSE+LPIPS gradient norm.
The resolved lambda is then frozen for the entire run and stored in the fingerprint.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_tiny_decoder_formal_ddp as formal
from tools import train_tiny_decoder_resize_conv_tail_recovery_ddp as tail
from swiftvr.models import ReAE
from swiftvr.models.tiny_conditional_decoder_resize_conv import (
    ResizeConvTinyConditionalDecoder,
)
from swiftvr.training import build_grad_scaler, cast_trainable_parameters
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.tiny_decoder import LPIPSAlexLoss, tiny_decoder_objective
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache
from swiftvr.training.tiny_decoder_phase import (
    calibrated_phase_weight,
    residual_new_p8_phase_loss,
)


TRAINER_ID = "swiftvr_stage_b1_tiny_decoder_resize_conv_phase_recovery_ddp_v1"
PHASE_LOSS_ID = "tiny_minus_reae_luma_new_p8_beyond_p4_per_frame_rms"


def build_parser() -> argparse.ArgumentParser:
    parser = tail.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--phase-loss-weight",
        type=float,
        default=None,
        help=(
            "Fixed lambda for L_phase8. If omitted, calibrate lambda once from the "
            "first distributed training batch before optimization."
        ),
    )
    parser.add_argument(
        "--phase-gradient-target-ratio",
        type=float,
        default=0.1,
        help=(
            "Auto-calibration target for ||lambda*g_phase|| / ||g_base||. "
            "Ignored when --phase-loss-weight is supplied."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    tail._validate_args(args)
    if args.phase_loss_weight is not None and float(args.phase_loss_weight) <= 0:
        raise ValueError("phase-loss-weight must be positive")
    if float(args.phase_gradient_target_ratio) <= 0:
        raise ValueError("phase-gradient-target-ratio must be positive")


def _fingerprint(
    args: argparse.Namespace,
    *,
    world_size: int,
    train_cache: TinyDecoderLatentCache,
    val_cache: TinyDecoderLatentCache,
    init_decoder: Path,
    phase_loss_weight: float,
    phase_loss_weight_mode: str,
    phase_gradient_target_ratio: float,
) -> dict[str, object]:
    fingerprint = tail._fingerprint(
        args,
        world_size=world_size,
        train_cache=train_cache,
        val_cache=val_cache,
        init_decoder=init_decoder,
    )
    fingerprint.update(
        {
            "trainer": TRAINER_ID,
            "explicit_phase_loss": True,
            "phase_loss": PHASE_LOSS_ID,
            "phase_loss_weight": float(phase_loss_weight),
            "phase_loss_weight_mode": str(phase_loss_weight_mode),
            "phase_gradient_target_ratio": float(phase_gradient_target_ratio),
        }
    )
    return fingerprint


def _grad_squared_norm(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )
    result = loss.new_zeros((), dtype=torch.float64)
    for gradient in gradients:
        result = result + gradient.detach().double().square().sum()
    return result


def _calibrate_phase_weight(
    model: ResizeConvTinyConditionalDecoder,
    reae: ReAE,
    cache: TinyDecoderLatentCache,
    batch_cpu: Mapping[str, object],
    perceptual: LPIPSAlexLoss | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    world_size: int,
) -> tuple[float, dict[str, float | str]]:
    """Calibrate lambda from one batch per rank without modifying parameters."""
    model.train()
    moved = formal._move_pixels(batch_cpu, device, dtype)
    prepared = prepare_training_batch(moved)
    lq_input = prepared["lq_input"]
    target = prepared["target"]
    if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("Calibration batch is missing lq_input/target")
    z_sr = cache.load_batch(batch_cpu, device=device, dtype=dtype)
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)

    with torch.no_grad(), torch.autocast(
        "cuda", dtype=dtype, enabled=autocast_enabled
    ):
        teacher = decode_reae_clip(
            reae,
            z_sr,
            output_frames=int(target.shape[1]),
            clamp=False,
        )

    with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        prediction = model(
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
    phase8 = residual_new_p8_phase_loss(prediction, teacher)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise RuntimeError("No trainable parameters available for phase calibration")

    base_sq = _grad_squared_norm(objective["loss"], parameters, retain_graph=True)
    phase_sq = _grad_squared_norm(phase8, parameters, retain_graph=False)
    packed = torch.stack(
        (
            base_sq,
            phase_sq,
            objective["loss"].detach().double(),
            phase8.detach().double(),
        )
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed /= float(world_size)

    base_grad_norm = math.sqrt(max(float(packed[0].item()), 0.0))
    phase_grad_norm = math.sqrt(max(float(packed[1].item()), 0.0))
    target_ratio = float(args.phase_gradient_target_ratio)
    weight = calibrated_phase_weight(
        base_grad_norm,
        phase_grad_norm,
        target_ratio=target_ratio,
    )
    achieved_ratio = weight * phase_grad_norm / max(base_grad_norm, 1e-30)
    report: dict[str, float | str] = {
        "mode": "auto_one_batch_per_rank_rms_grad_norm",
        "base_loss": float(packed[2].item()),
        "phase8_loss": float(packed[3].item()),
        "base_grad_norm": base_grad_norm,
        "phase_grad_norm": phase_grad_norm,
        "target_gradient_ratio": target_ratio,
        "resolved_phase_loss_weight": weight,
        "resolved_gradient_ratio": achieved_ratio,
    }
    return weight, report


@torch.no_grad()
def _validate_phase8(
    model: ResizeConvTinyConditionalDecoder,
    reae: ReAE,
    cache: TinyDecoderLatentCache,
    loader,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    model.eval()
    reae.eval()
    total = 0.0
    sample_count = 0
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    for batch_cpu in loader:
        moved = formal._move_pixels(batch_cpu, device, dtype)
        prepared = prepare_training_batch(moved)
        lq_input = prepared["lq_input"]
        target = prepared["target"]
        if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("Validation batch is missing lq_input/target")
        z_sr = cache.load_batch(batch_cpu, device=device, dtype=dtype)
        with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
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
        phase8 = residual_new_p8_phase_loss(prediction, teacher)
        batch_size = int(target.shape[0])
        total += float(phase8.item()) * batch_size
        sample_count += batch_size
    if sample_count <= 0:
        raise RuntimeError("Validation loader is empty")
    return total / sample_count


def _validate_with_phase(
    model: ResizeConvTinyConditionalDecoder,
    reae: ReAE,
    cache: TinyDecoderLatentCache,
    loader,
    perceptual: LPIPSAlexLoss | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    phase_loss_weight: float,
) -> dict[str, float | int]:
    validation = formal._validate(
        model,
        reae,
        cache,
        loader,
        perceptual,
        device=device,
        dtype=dtype,
        args=args,
    )
    phase8 = _validate_phase8(
        model,
        reae,
        cache,
        loader,
        device=device,
        dtype=dtype,
    )
    base_loss = float(validation["loss"])
    weighted_phase8 = float(phase_loss_weight) * phase8
    validation["base_loss"] = base_loss
    validation["phase8_residual_rms"] = phase8
    validation["weighted_phase8"] = weighted_phase8
    validation["regularized_loss"] = base_loss + weighted_phase8
    return validation


def _annotate_best(run_dir: Path, selection_loss: float) -> None:
    path = run_dir / formal.BEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection_metric"] = "regularized_loss"
    payload["selection_loss"] = float(selection_loss)
    formal._write_json(path, payload)


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    rank, local_rank, world_size, device = formal._init_distributed(
        args.ddp_timeout_seconds
    )
    try:
        formal._seed(args.seed + rank)
        dtype = formal.DTYPES[args.dtype]
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"{torch.cuda.get_device_name(device)} does not support BF16"
            )

        run_dir = args.output_dir.expanduser().resolve()
        path_root = args.path_root.expanduser().resolve()
        base = args.base_checkpoint.expanduser().resolve()
        init_decoder = args.init_decoder.expanduser().resolve()
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
            raise ValueError(
                f"Formal B1 train cache must contain 15896 views, got {len(train_dataset)}"
            )
        if len(val_dataset) != 13:
            raise ValueError(
                f"Primary formal B1 validation must contain 13 views, got {len(val_dataset)}"
            )

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

        reae = ReAE(str(base / args.reae_filename)).to(device=device, dtype=dtype).eval()
        for parameter in reae.parameters():
            parameter.requires_grad_(False)

        resume_checkpoint: Path | None = None
        saved_state: Mapping[str, object] | None = None
        saved_fingerprint: Mapping[str, object] | None = None
        if args.resume is not None:
            resume_checkpoint = (
                formal._resolve_resume(run_dir, args.resume) if rank == 0 else None
            )
            payload = [str(resume_checkpoint) if rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            resume_checkpoint = Path(str(payload[0])).resolve()
            tiny = ResizeConvTinyConditionalDecoder.from_pretrained(
                resume_checkpoint / "tiny_decoder",
                device=device,
                dtype=dtype,
            )
            saved_state = formal._load_training_state(resume_checkpoint)
            candidate = saved_state.get("fingerprint")
            if not isinstance(candidate, Mapping):
                raise TypeError("Resume checkpoint has no valid fingerprint")
            saved_fingerprint = candidate
        else:
            tiny = ResizeConvTinyConditionalDecoder.from_pretrained(
                init_decoder,
                device=device,
                dtype=dtype,
            )

        if tiny.block_mode != "compact":
            raise ValueError(
                f"Formal B1 requires materialized compact decoder, got {tiny.block_mode!r}"
            )
        if tuple(tiny.block_internal_channels or ()) != (80, 48, 24, 16):
            raise ValueError(
                "Formal B1 topology is frozen to keep_040 internal widths "
                f"(80,48,24,16), got {tiny.block_internal_channels}"
            )
        if tiny.resize_mode != args.resize_mode:
            raise ValueError(
                f"Checkpoint resize_mode={tiny.resize_mode!r} != requested {args.resize_mode!r}"
            )

        trainable_report, groups = tail._set_tail_trainable(tiny)
        cast_report = cast_trainable_parameters(tiny, dtype=torch.float32)
        tiny.train()
        optimizer = tail._build_grouped_adamw(groups, args)
        scaler = build_grad_scaler(device, dtype)
        perceptual = (
            LPIPSAlexLoss().to(device=device).eval()
            if args.lpips_weight > 0
            else None
        )

        if saved_fingerprint is not None:
            saved_weight = float(saved_fingerprint["phase_loss_weight"])
            if args.phase_loss_weight is not None and not math.isclose(
                float(args.phase_loss_weight), saved_weight, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "Resume phase-loss-weight does not match checkpoint: "
                    f"requested={args.phase_loss_weight} saved={saved_weight}"
                )
            phase_loss_weight = saved_weight
            phase_loss_weight_mode = str(saved_fingerprint["phase_loss_weight_mode"])
            phase_gradient_target_ratio = float(
                saved_fingerprint["phase_gradient_target_ratio"]
            )
            calibration_report: dict[str, float | str] = {
                "mode": "resume_reuse_saved_weight",
                "resolved_phase_loss_weight": phase_loss_weight,
            }
        elif args.phase_loss_weight is not None:
            phase_loss_weight = float(args.phase_loss_weight)
            phase_loss_weight_mode = "fixed"
            phase_gradient_target_ratio = float(args.phase_gradient_target_ratio)
            calibration_report = {
                "mode": "fixed",
                "resolved_phase_loss_weight": phase_loss_weight,
            }
        else:
            sampler.set_epoch(0)
            try:
                calibration_batch = next(iter(train_loader))
            except StopIteration as exc:
                raise RuntimeError("Training loader is empty during phase calibration") from exc
            phase_loss_weight, calibration_report = _calibrate_phase_weight(
                tiny,
                reae,
                train_cache,
                calibration_batch,
                perceptual,
                device=device,
                dtype=dtype,
                args=args,
                world_size=world_size,
            )
            phase_loss_weight_mode = "auto"
            phase_gradient_target_ratio = float(args.phase_gradient_target_ratio)
            sampler.set_epoch(0)

        fingerprint = _fingerprint(
            args,
            world_size=world_size,
            train_cache=train_cache,
            val_cache=val_cache,
            init_decoder=init_decoder,
            phase_loss_weight=phase_loss_weight,
            phase_loss_weight_mode=phase_loss_weight_mode,
            phase_gradient_target_ratio=phase_gradient_target_ratio,
        )

        start_epoch = 0
        global_step = 0
        best_val_loss = math.inf
        if saved_state is not None:
            assert saved_fingerprint is not None
            formal._assert_fingerprint(saved_fingerprint, fingerprint)
            optimizer.load_state_dict(saved_state["optimizer"])
            scaler.load_state_dict(saved_state["scaler"])
            start_epoch = int(saved_state["completed_epoch"])
            global_step = int(saved_state["global_step"])
            best_val_loss = float(saved_state.get("best_val_loss", math.inf))
            if start_epoch >= args.epochs:
                raise ValueError(
                    f"Checkpoint already completed {start_epoch} epochs; requested epochs={args.epochs}"
                )

        ddp = DDP(
            tiny,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        autocast_enabled = dtype in (torch.float16, torch.bfloat16)

        if rank == 0:
            if args.resume is None:
                if run_dir.exists() and any(run_dir.iterdir()):
                    raise FileExistsError(
                        f"Refusing to overwrite non-empty run directory: {run_dir}"
                    )
                run_dir.mkdir(parents=True, exist_ok=True)
                formal._write_json(
                    run_dir / formal.RUN_CONFIG_FILENAME,
                    {
                        **fingerprint,
                        "epochs": int(args.epochs),
                        "train_samples": len(train_dataset),
                        "val_samples": len(val_dataset),
                        "steps_per_epoch_per_rank": len(train_loader),
                        "decoder_block_internal_channels": list(
                            tiny.block_internal_channels or ()
                        ),
                        "decoder_parameters": sum(p.numel() for p in tiny.parameters()),
                        "trainable": trainable_report,
                        "trainable_cast": cast_report,
                        "phase_calibration": calibration_report,
                    },
                )
                formal._write_json(run_dir / "phase_calibration.json", calibration_report)
            elif not run_dir.is_dir():
                raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
            print(
                json.dumps(
                    {
                        "phase": "phase_loss_configuration",
                        "phase_loss": PHASE_LOSS_ID,
                        "phase_loss_weight": phase_loss_weight,
                        "phase_loss_weight_mode": phase_loss_weight_mode,
                        **calibration_report,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.barrier()

        if args.resume is None:
            if rank == 0:
                assert val_loader is not None
                initial_val = _validate_with_phase(
                    tiny,
                    reae,
                    val_cache,
                    val_loader,
                    perceptual,
                    device=device,
                    dtype=dtype,
                    args=args,
                    phase_loss_weight=phase_loss_weight,
                )
                formal._write_json(run_dir / "validation_epoch_000.json", initial_val)
                formal._append_jsonl(
                    run_dir / "val_log.jsonl",
                    {"completed_epoch": 0, "global_step": 0, **initial_val},
                )
                print(
                    json.dumps(
                        {
                            "phase": "initial_phase_recovery_validation",
                            "trainable_scope": tail.TRAINABLE_SCOPE,
                            **initial_val,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            dist.barrier()

        base_keys = (
            "gt_l2",
            "teacher_l2",
            "gt_lpips",
            "teacher_lpips",
            "gt_temporal_mse",
            "teacher_temporal_mse",
        )
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            ddp.train()
            interval_sums = {
                "loss": 0.0,
                "base_loss": 0.0,
                "phase8_residual_rms": 0.0,
                "weighted_phase8": 0.0,
                **{key: 0.0 for key in base_keys},
            }
            interval_count = 0
            interval_grad = 0.0
            interval_started = time.perf_counter()

            for batch_index, batch_cpu in enumerate(train_loader, start=1):
                moved = formal._move_pixels(batch_cpu, device, dtype)
                prepared = prepare_training_batch(moved)
                lq_input = prepared["lq_input"]
                target = prepared["target"]
                if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
                    raise TypeError("Training batch is missing lq_input/target")
                z_sr = train_cache.load_batch(batch_cpu, device=device, dtype=dtype)

                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=dtype, enabled=autocast_enabled
                ):
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
                phase8 = residual_new_p8_phase_loss(prediction, teacher)
                weighted_phase8 = phase8 * phase_loss_weight
                loss = objective["loss"] + weighted_phase8

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in ddp.parameters() if parameter.requires_grad),
                    args.max_grad_norm,
                )
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
                interval_sums["loss"] += float(loss.detach().item()) * batch_size
                interval_sums["base_loss"] += float(objective["loss"].detach().item()) * batch_size
                interval_sums["phase8_residual_rms"] += float(phase8.detach().item()) * batch_size
                interval_sums["weighted_phase8"] += float(weighted_phase8.detach().item()) * batch_size
                for key in base_keys:
                    interval_sums[key] += float(objective[key].detach().item()) * batch_size

                should_log = batch_index % args.log_every == 0 or batch_index == len(train_loader)
                if should_log:
                    elapsed = time.perf_counter() - interval_started
                    reduced = formal._allreduce_interval(
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
                            "trainable_scope": tail.TRAINABLE_SCOPE,
                            "phase_loss_weight": phase_loss_weight,
                            "learning_rates": {
                                group["group_name"]: float(group["lr"])
                                for group in optimizer.param_groups
                            },
                            **reduced,
                        }
                        formal._append_jsonl(run_dir / "train_log.jsonl", record)
                        print(json.dumps(record, sort_keys=True), flush=True)
                    interval_sums = {key: 0.0 for key in interval_sums}
                    interval_count = 0
                    interval_grad = 0.0
                    interval_started = time.perf_counter()

            dist.barrier()
            if rank == 0:
                assert val_loader is not None
                validation = _validate_with_phase(
                    tiny,
                    reae,
                    val_cache,
                    val_loader,
                    perceptual,
                    device=device,
                    dtype=dtype,
                    args=args,
                    phase_loss_weight=phase_loss_weight,
                )
                completed_epoch = epoch + 1
                record = {
                    "completed_epoch": completed_epoch,
                    "global_step": global_step,
                    **validation,
                }
                formal._append_jsonl(run_dir / "val_log.jsonl", record)
                formal._write_json(
                    run_dir / f"validation_epoch_{completed_epoch:03d}.json",
                    validation,
                )
                current_val_loss = float(validation["regularized_loss"])
                is_best = current_val_loss < best_val_loss
                if is_best:
                    best_val_loss = current_val_loss
                checkpoint = formal._save_checkpoint(
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
                if is_best:
                    _annotate_best(run_dir, current_val_loss)
                print(
                    json.dumps(
                        {
                            "phase": "epoch_validation",
                            "checkpoint": str(checkpoint),
                            "is_best": is_best,
                            "selection_metric": "regularized_loss",
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
            best = json.loads((run_dir / formal.BEST_FILENAME).read_text(encoding="utf-8"))
            summary = {
                "status": "PASS",
                "completed_epochs": int(args.epochs),
                "global_step": int(global_step),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "world_size": world_size,
                "local_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "resize_mode": args.resize_mode,
                "trainable_scope": tail.TRAINABLE_SCOPE,
                "trainable_parameters": int(trainable_report["parameter_elements"]),
                "parameter_group_learning_rates": {
                    group["group_name"]: float(group["lr"])
                    for group in optimizer.param_groups
                },
                "phase_loss": PHASE_LOSS_ID,
                "phase_loss_weight": phase_loss_weight,
                "phase_loss_weight_mode": phase_loss_weight_mode,
                "phase_calibration": calibration_report,
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
