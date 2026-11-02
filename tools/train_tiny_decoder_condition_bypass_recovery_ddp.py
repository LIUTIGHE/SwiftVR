#!/usr/bin/env python3
"""F1b: exact-init rich packed-LQ bypass recovery for Stage-B1.

The supplied ResizeConv checkpoint remains an exact frozen baseline path.  A new
zero-initialized 3072->C0 3x3 branch reads the packed RGB condition directly and
is added immediately after the trained decoder input convolution.  Only this
bypass weight is optimized; the 3072->32 projection, latent fusion, compact
trunk, transitions, and ResizeConv RGB head all remain frozen.

This cleanly tests whether the 32-channel condition bottleneck is limiting the
compact decoder without destroying the already-trained R4 function at epoch 0.
No explicit phase loss is used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_tiny_decoder_formal_ddp as formal
from swiftvr.models import ReAE
from swiftvr.models.tiny_conditional_decoder_condition_bypass_resize_conv import (
    CONDITION_BYPASS_MODE,
    CONDITION_BYPASS_SCHEME,
    ConditionBypassResizeConvTinyConditionalDecoder,
)
from swiftvr.models.tiny_conditional_decoder_resize_conv import RESIZE_MODES
from swiftvr.training import build_grad_scaler, cast_trainable_parameters, trainable_named_parameters
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.tiny_decoder import LPIPSAlexLoss, tiny_decoder_objective
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


TRAINER_ID = "swiftvr_stage_b1_tiny_decoder_condition_bypass_recovery_ddp_v1"
TRAINABLE_SCOPE = "direct_condition_bypass_only"


def build_parser() -> argparse.ArgumentParser:
    parser = formal.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--resize-mode",
        choices=tuple(sorted(RESIZE_MODES)),
        default="nearest",
    )
    parser.add_argument(
        "--bypass-learning-rate",
        type=float,
        default=1e-5,
        help="LR for the zero-init packed-condition bypass convolution.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    formal._validate_args(args)
    if args.resize_mode not in RESIZE_MODES:
        raise ValueError(f"resize-mode must be one of {sorted(RESIZE_MODES)}")
    if float(args.bypass_learning_rate) <= 0:
        raise ValueError("bypass-learning-rate must be positive")


def _set_bypass_trainable(
    model: ConditionBypassResizeConvTinyConditionalDecoder,
) -> tuple[dict[str, object], list[torch.nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.direct_condition_bypass.parameters():
        parameter.requires_grad_(True)

    named = trainable_named_parameters(model)
    actual = {name for name, _ in named}
    expected = {"direct_condition_bypass.weight"}
    if actual != expected:
        raise RuntimeError(
            f"Condition-bypass trainable scope mismatch: expected={sorted(expected)} actual={sorted(actual)}"
        )
    parameters = [parameter for _, parameter in named]
    report = {
        "scope": TRAINABLE_SCOPE,
        "parameter_tensors": len(named),
        "parameter_elements": sum(parameter.numel() for parameter in parameters),
        "parameter_names": [name for name, _ in named],
        "packed_condition_channels": int(model.packed_condition_channels),
        "output_channels": int(model.channels[0]),
        "source_condition_projection_preserved": hasattr(model, "condition_projection"),
    }
    return report, parameters


def _build_optimizer(
    parameters: list[torch.nn.Parameter], args: argparse.Namespace
) -> torch.optim.AdamW:
    if not parameters:
        raise RuntimeError("No trainable condition-bypass parameters")
    bad = sorted({str(p.dtype) for p in parameters if p.dtype != torch.float32})
    if bad:
        raise RuntimeError(f"Bypass optimizer requires FP32 parameters, found {bad}")
    return torch.optim.AdamW(
        [{
            "params": parameters,
            "lr": float(args.bypass_learning_rate),
            "group_name": "condition_bypass",
        }],
        lr=float(args.bypass_learning_rate),
        weight_decay=float(args.weight_decay),
        eps=float(args.optimizer_eps),
        foreach=False,
    )


def _fingerprint(
    args: argparse.Namespace,
    *,
    world_size: int,
    train_cache: TinyDecoderLatentCache,
    val_cache: TinyDecoderLatentCache,
    init_decoder: Path,
) -> dict[str, object]:
    fingerprint = formal._fingerprint(
        args,
        world_size=world_size,
        train_cache=train_cache,
        val_cache=val_cache,
        init_decoder=init_decoder,
    )
    fingerprint.update(
        {
            "trainer": TRAINER_ID,
            "output_head": "resize_conv",
            "resize_mode": args.resize_mode,
            "condition_injection": CONDITION_BYPASS_MODE,
            "condition_bypass_scheme": CONDITION_BYPASS_SCHEME,
            "packed_condition_channels": 3 * 4 * 16 * 16,
            "source_condition_projection": "preserved_frozen",
            "trainable_scope": TRAINABLE_SCOPE,
            "bypass_learning_rate": float(args.bypass_learning_rate),
            "condition_dropout_probability": 0.0,
            "explicit_phase_loss": False,
        }
    )
    return fingerprint


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

        reae = ReAE(str(base / args.reae_filename)).to(device=device, dtype=dtype).eval()
        for parameter in reae.parameters():
            parameter.requires_grad_(False)

        resume_checkpoint: Path | None = None
        initialization_report: dict[str, object] | None = None
        if args.resume is not None:
            resume_checkpoint = formal._resolve_resume(run_dir, args.resume) if rank == 0 else None
            payload = [str(resume_checkpoint) if rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            resume_checkpoint = Path(str(payload[0])).resolve()
            tiny = ConditionBypassResizeConvTinyConditionalDecoder.from_pretrained(
                resume_checkpoint / "tiny_decoder", device=device, dtype=dtype
            )
        else:
            tiny, initialization_report = (
                ConditionBypassResizeConvTinyConditionalDecoder.from_resizeconv_pretrained(
                    init_decoder, device=device, dtype=dtype
                )
            )

        if tiny.block_mode != "compact":
            raise ValueError(f"Formal B1 requires compact decoder, got {tiny.block_mode!r}")
        if tuple(tiny.block_internal_channels or ()) != (80, 48, 24, 16):
            raise ValueError(
                "Formal B1 topology is frozen to keep_040 internal widths "
                f"(80,48,24,16), got {tiny.block_internal_channels}"
            )
        if tiny.resize_mode != args.resize_mode:
            raise ValueError(
                f"Checkpoint resize_mode={tiny.resize_mode!r} != requested {args.resize_mode!r}"
            )
        if tiny.packed_condition_channels != 3072:
            raise ValueError(
                f"F1b requires 3072 packed RGB channels, got {tiny.packed_condition_channels}"
            )
        if not hasattr(tiny, "condition_projection"):
            raise RuntimeError("F1b must preserve the trained source condition projection")

        trainable_report, trainable_parameters = _set_bypass_trainable(tiny)
        cast_report = cast_trainable_parameters(tiny, dtype=torch.float32)
        tiny.train()
        optimizer = _build_optimizer(trainable_parameters, args)
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
            state = formal._load_training_state(resume_checkpoint)
            saved_fingerprint = state.get("fingerprint")
            if not isinstance(saved_fingerprint, Mapping):
                raise TypeError("Resume checkpoint has no valid fingerprint")
            formal._assert_fingerprint(saved_fingerprint, fingerprint)
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_epoch = int(state["completed_epoch"])
            global_step = int(state["global_step"])
            best_val_loss = float(state.get("best_val_loss", math.inf))
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
                        "decoder_block_internal_channels": list(tiny.block_internal_channels or ()),
                        "decoder_parameters": sum(p.numel() for p in tiny.parameters()),
                        "trainable": trainable_report,
                        "trainable_cast": cast_report,
                        "initialization": initialization_report,
                    },
                )
            elif not run_dir.is_dir():
                raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
        dist.barrier()

        # Epoch 0 must reproduce R4 because the only new path is exactly zero.
        if args.resume is None:
            if rank == 0:
                assert val_loader is not None
                initial_val = formal._validate(
                    tiny, reae, val_cache, val_loader, perceptual,
                    device=device, dtype=dtype, args=args,
                )
                formal._write_json(run_dir / "validation_epoch_000.json", initial_val)
                formal._append_jsonl(
                    run_dir / "val_log.jsonl",
                    {"completed_epoch": 0, "global_step": 0, **initial_val},
                )
                print(json.dumps({
                    "phase": "initial_condition_bypass_validation",
                    "trainable_scope": TRAINABLE_SCOPE,
                    "condition_injection": CONDITION_BYPASS_MODE,
                    **initial_val,
                }, sort_keys=True), flush=True)
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
                moved = formal._move_pixels(batch_cpu, device, dtype)
                prepared = prepare_training_batch(moved)
                lq_input = prepared["lq_input"]
                target = prepared["target"]
                if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
                    raise TypeError("Training batch is missing lq_input/target")
                z_sr = train_cache.load_batch(batch_cpu, device=device, dtype=dtype)

                with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                    teacher = decode_reae_clip(
                        reae, z_sr, output_frames=int(target.shape[1]), clamp=False
                    )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                    prediction = ddp(
                        z_sr, lq_input, output_frames=int(target.shape[1]), clamp=False
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
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in ddp.parameters() if p.requires_grad), args.max_grad_norm
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
                for key in interval_sums:
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
                            "trainable_scope": TRAINABLE_SCOPE,
                            "condition_injection": CONDITION_BYPASS_MODE,
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
                validation = formal._validate(
                    tiny, reae, val_cache, val_loader, perceptual,
                    device=device, dtype=dtype, args=args,
                )
                completed_epoch = epoch + 1
                record = {"completed_epoch": completed_epoch, "global_step": global_step, **validation}
                formal._append_jsonl(run_dir / "val_log.jsonl", record)
                formal._write_json(
                    run_dir / f"validation_epoch_{completed_epoch:03d}.json", validation
                )
                current_val_loss = float(validation["loss"])
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
                print(json.dumps({
                    "phase": "epoch_validation",
                    "checkpoint": str(checkpoint),
                    "is_best": is_best,
                    **record,
                }, sort_keys=True), flush=True)
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
                "condition_injection": CONDITION_BYPASS_MODE,
                "packed_condition_channels": int(tiny.packed_condition_channels),
                "trainable_scope": TRAINABLE_SCOPE,
                "trainable_parameters": int(trainable_report["parameter_elements"]),
                "bypass_learning_rate": float(args.bypass_learning_rate),
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
