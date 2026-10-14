#!/usr/bin/env python3
"""Stage-B1 Tiny Decoder refinement with projected-condition dropout.

This trainer is intentionally isolated from the frozen formal B1 trainer. It
reuses the same deterministic datasets, cached z_SR latents, ReAE teacher,
objective, validation, logging, and checkpoint helpers, but during *training
only* it can zero the projected LQ-condition feature for a deterministic subset
of samples. Validation and inference always use the full aligned condition.

The purpose is to regularize excessive deterministic RGB-condition injection
without changing the TinyConditionalDecoder checkpoint format or its normal
inference path. Condition-dropout masks are deterministic functions of
(seed, epoch, batch index, rank), so epoch-boundary resume remains reproducible
for a fixed world size and training recipe.
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_tiny_decoder_formal_ddp as formal
from swiftvr.models.tiny_conditional_decoder import (
    TinyConditionalDecoder,
    _apply_video_stack,
    _validate_video,
)
from swiftvr.training import build_fp32_adamw, build_grad_scaler, cast_trainable_parameters
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.tiny_decoder import LPIPSAlexLoss, tiny_decoder_objective
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache
from swiftvr.models import ReAE


TRAINER_ID = "swiftvr_stage_b1_tiny_decoder_condition_dropout_ddp_v1"
MASK_SCHEME = "seed_epoch_batch_rank_per_sample_v1"


class ConditionDropoutTinyDecoder(TinyConditionalDecoder):
    """Tiny decoder with an optional projected-condition keep mask.

    No parameters or persistent buffers are added. When ``condition_keep_mask``
    is omitted, execution is identical to TinyConditionalDecoder.forward.
    """

    @property
    def config_dict(self) -> dict[str, object]:
        # Keep checkpoints loadable as ordinary TinyConditionalDecoder models.
        config = dict(super().config_dict)
        config["class_name"] = "TinyConditionalDecoder"
        return config

    def forward(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor,
        *,
        output_frames: int | None = None,
        clamp: bool = False,
        condition_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_video("latents", latents, channels=self.latent_channels)
        _validate_video("condition", condition, channels=3)
        if int(latents.shape[0]) != int(condition.shape[0]):
            raise ValueError("latents and condition must share batch size")

        condition_latent = self.project_condition(condition)
        if tuple(condition_latent.shape[:2]) != tuple(latents.shape[:2]) or tuple(
            condition_latent.shape[-2:]
        ) != tuple(latents.shape[-2:]):
            raise ValueError(
                "Packed condition does not match latent grid: "
                f"condition={tuple(condition_latent.shape)}, latent={tuple(latents.shape)}"
            )

        if condition_keep_mask is not None:
            if not isinstance(condition_keep_mask, torch.Tensor):
                raise TypeError("condition_keep_mask must be a tensor or None")
            expected = (int(latents.shape[0]), 1, 1, 1, 1)
            if tuple(condition_keep_mask.shape) != expected:
                raise ValueError(
                    "condition_keep_mask must have shape "
                    f"{expected}, got {tuple(condition_keep_mask.shape)}"
                )
            condition_latent = condition_latent * condition_keep_mask.to(
                device=condition_latent.device,
                dtype=condition_latent.dtype,
            )

        hidden = torch.cat([latents, condition_latent], dim=2)
        pixels = _apply_video_stack(self.decoder, hidden)

        batch, frames, channels, height, width = pixels.shape
        flat = F.pixel_shuffle(
            pixels.reshape(batch * frames, channels, height, width),
            self.patch_size,
        )
        pixels = flat.reshape(batch, frames, *flat.shape[1:])

        if self.frames_to_trim:
            pixels = pixels[:, self.frames_to_trim :]
        if output_frames is not None:
            output_frames = int(output_frames)
            if output_frames <= 0:
                raise ValueError("output_frames must be positive")
            if pixels.shape[1] < output_frames:
                raise RuntimeError(
                    f"Tiny decoder emitted {pixels.shape[1]} valid frames; "
                    f"requested {output_frames}"
                )
            pixels = pixels[:, :output_frames]
        return pixels.clamp(0.0, 1.0) if clamp else pixels


def build_parser() -> argparse.ArgumentParser:
    parser = formal.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--condition-dropout-probability",
        type=float,
        default=0.0,
        help=(
            "Training-only probability of zeroing the projected LQ-condition "
            "feature for each sample. Validation/inference always use the full "
            "condition. Default 0 preserves the formal B1 behavior."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    formal._validate_args(args)
    probability = float(args.condition_dropout_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("condition-dropout-probability must be in [0,1]")


def _mask_seed(*, seed: int, epoch: int, batch_index: int, rank: int) -> int:
    """Deterministic 63-bit seed independent of prior RNG consumption."""
    value = (
        int(seed) * 1_000_003
        + int(epoch) * 1_000_000_007
        + int(batch_index) * 10_000_019
        + int(rank) * 100_003
        + 0x5A17B1
    )
    return value % (2**63 - 1)


def _condition_keep_mask(
    batch_size: int,
    probability: float,
    *,
    seed: int,
    epoch: int,
    batch_index: int,
    rank: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return [B,1,1,1,1] keep mask, or None when dropout is disabled."""
    probability = float(probability)
    if probability <= 0.0:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("condition dropout probability must be in [0,1]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        _mask_seed(
            seed=seed,
            epoch=epoch,
            batch_index=batch_index,
            rank=rank,
        )
    )
    keep = torch.rand((int(batch_size),), generator=generator) >= probability
    return keep.to(device=device).reshape(int(batch_size), 1, 1, 1, 1)


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
    fingerprint["trainer"] = TRAINER_ID
    fingerprint["condition_dropout_probability"] = float(
        args.condition_dropout_probability
    )
    fingerprint["condition_dropout_mask_scheme"] = MASK_SCHEME
    return fingerprint


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

        reae = ReAE(str(base / args.reae_filename)).to(
            device=device, dtype=dtype
        ).eval()
        for parameter in reae.parameters():
            parameter.requires_grad_(False)

        resume_checkpoint: Path | None = None
        if args.resume is not None:
            resume_checkpoint = (
                formal._resolve_resume(run_dir, args.resume) if rank == 0 else None
            )
            payload = [str(resume_checkpoint) if rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            resume_checkpoint = Path(str(payload[0])).resolve()
            tiny_root = resume_checkpoint / "tiny_decoder"
        else:
            tiny_root = init_decoder

        tiny = ConditionDropoutTinyDecoder.from_pretrained(
            tiny_root, device=device, dtype=None
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
        cast_trainable_parameters(tiny, dtype=torch.float32)
        tiny.train()
        optimizer = build_fp32_adamw(
            tiny,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.optimizer_eps,
        )
        scaler = build_grad_scaler(device, dtype)
        perceptual = (
            LPIPSAlexLoss().to(device=device).eval()
            if args.lpips_weight > 0
            else None
        )

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
                    f"Checkpoint already completed {start_epoch} epochs; "
                    f"requested epochs={args.epochs}"
                )

        ddp = DDP(
            tiny,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        autocast_enabled = dtype in (torch.float16, torch.bfloat16)

        if rank == 0:
            if args.resume is None and run_dir.exists() and any(run_dir.iterdir()):
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
                },
            )
        dist.barrier()

        # Fresh runs establish a full-condition validation baseline.
        if args.resume is None:
            dist.barrier()
            if rank == 0:
                assert val_loader is not None
                initial_val = formal._validate(
                    tiny,
                    reae,
                    val_cache,
                    val_loader,
                    perceptual,
                    device=device,
                    dtype=dtype,
                    args=args,
                )
                formal._write_json(
                    run_dir / "validation_epoch_000.json", initial_val
                )
                formal._append_jsonl(
                    run_dir / "val_log.jsonl",
                    {
                        "completed_epoch": 0,
                        "global_step": 0,
                        **initial_val,
                    },
                )
                print(
                    json.dumps(
                        {"phase": "initial_validation", **initial_val},
                        sort_keys=True,
                    ),
                    flush=True,
                )
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
            interval_condition_kept = 0.0
            interval_started = time.perf_counter()

            for batch_index, batch_cpu in enumerate(train_loader, start=1):
                moved = formal._move_pixels(batch_cpu, device, dtype)
                prepared = prepare_training_batch(moved)
                lq_input = prepared["lq_input"]
                target = prepared["target"]
                if not isinstance(lq_input, torch.Tensor) or not isinstance(
                    target, torch.Tensor
                ):
                    raise TypeError("Training batch is missing lq_input/target")
                z_sr = train_cache.load_batch(
                    batch_cpu, device=device, dtype=dtype
                )
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=dtype, enabled=autocast_enabled
                ):
                    teacher = decode_reae_clip(
                        reae,
                        z_sr,
                        output_frames=int(target.shape[1]),
                        clamp=False,
                    )

                batch_size = int(target.shape[0])
                keep_mask = _condition_keep_mask(
                    batch_size,
                    args.condition_dropout_probability,
                    seed=args.seed,
                    epoch=epoch,
                    batch_index=batch_index,
                    rank=rank,
                    device=device,
                )
                kept_count = (
                    float(batch_size)
                    if keep_mask is None
                    else float(keep_mask.float().sum().item())
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    "cuda", dtype=dtype, enabled=autocast_enabled
                ):
                    prediction = ddp(
                        z_sr,
                        lq_input,
                        output_frames=int(target.shape[1]),
                        clamp=False,
                        condition_keep_mask=keep_mask,
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
                    ddp.parameters(), args.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        "Non-finite grad norm "
                        f"epoch={epoch + 1} batch={batch_index}: {grad_norm}"
                    )
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

                interval_count += batch_size
                interval_condition_kept += kept_count
                interval_grad += float(grad_norm.detach().item()) * batch_size
                for key in interval_sums:
                    interval_sums[key] += (
                        float(objective[key].detach().item()) * batch_size
                    )

                should_log = (
                    batch_index % args.log_every == 0
                    or batch_index == len(train_loader)
                )
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
                    keep_tensor = torch.tensor(
                        [interval_condition_kept, float(interval_count)],
                        device=device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(keep_tensor, op=dist.ReduceOp.SUM)
                    condition_keep_fraction = float(
                        keep_tensor[0].item()
                        / max(keep_tensor[1].item(), 1.0)
                    )
                    if rank == 0:
                        record = {
                            "epoch": epoch + 1,
                            "batch": batch_index,
                            "batches_per_epoch": len(train_loader),
                            "global_step": global_step,
                            "global_batch_size": args.batch_size * world_size,
                            "condition_keep_fraction": condition_keep_fraction,
                            "condition_dropout_probability": float(
                                args.condition_dropout_probability
                            ),
                            **reduced,
                        }
                        formal._append_jsonl(
                            run_dir / "train_log.jsonl", record
                        )
                        print(json.dumps(record, sort_keys=True), flush=True)
                    interval_sums = {key: 0.0 for key in interval_sums}
                    interval_count = 0
                    interval_grad = 0.0
                    interval_condition_kept = 0.0
                    interval_started = time.perf_counter()

            dist.barrier()
            validation: dict[str, float | int] | None = None
            if rank == 0:
                assert val_loader is not None
                validation = formal._validate(
                    tiny,
                    reae,
                    val_cache,
                    val_loader,
                    perceptual,
                    device=device,
                    dtype=dtype,
                    args=args,
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
                print(
                    json.dumps(
                        {
                            "phase": "epoch_validation",
                            "checkpoint": str(checkpoint),
                            "is_best": is_best,
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
            best = json.loads(
                (run_dir / formal.BEST_FILENAME).read_text(encoding="utf-8")
            )
            summary = {
                "status": "PASS",
                "completed_epochs": int(args.epochs),
                "global_step": int(global_step),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "world_size": world_size,
                "local_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "condition_dropout_probability": float(
                    args.condition_dropout_probability
                ),
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
