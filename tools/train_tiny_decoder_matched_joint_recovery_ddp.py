#!/usr/bin/env python3
"""Matched full-decoder recovery for R4 and rich-condition-bypass Stage-B1 variants.

This trainer is the controlled architecture comparison after the R4 tail-recovery
and direct-condition diagnostics.  Two runs use the *same* code path and differ
only by ``--variant``:

* ``r4``: the supplied ResizeConv R4 decoder, jointly optimized end to end;
* ``condition-bypass``: the same R4 decoder plus an exact-zero-initialized direct
  packed-RGB -> C0 3x3 condition bypass, jointly optimized end to end.

Both variants start from the same R4 checkpoint, use the same cached z_SR, data,
objective, validation, optimizer groups and old-decoder learning rates.  The
condition-bypass run adds only one extra optimizer group for the new bypass.
No explicit phase loss or condition dropout is used.  Epoch-0 validation is always
written before optimization; for the bypass variant it should match R4 because the
new branch is identically zero at initialization.
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
from tools.train_tiny_decoder_resize_conv_tail_recovery_ddp import _decoder_layout
from swiftvr.models import ReAE
from swiftvr.models.tiny_conditional_decoder_condition_bypass_resize_conv import (
    CONDITION_BYPASS_MODE,
    ConditionBypassResizeConvTinyConditionalDecoder,
)
from swiftvr.models.tiny_conditional_decoder_resize_conv import (
    RESIZE_MODES,
    ResizeConvTinyConditionalDecoder,
)
from swiftvr.training import (
    build_grad_scaler,
    cast_trainable_parameters,
    trainable_named_parameters,
)
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.tiny_decoder import LPIPSAlexLoss, tiny_decoder_objective
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


TRAINER_ID = "swiftvr_stage_b1_tiny_decoder_matched_joint_recovery_ddp_v1"
VARIANTS = ("r4", "condition-bypass")
TRAINABLE_SCOPE = "full_decoder_joint"


def build_parser() -> argparse.ArgumentParser:
    parser = formal.build_parser()
    parser.description = __doc__
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--resize-mode", choices=tuple(sorted(RESIZE_MODES)), default="nearest"
    )
    parser.add_argument(
        "--condition-input-learning-rate",
        type=float,
        default=1e-5,
        help="LR for the learned 3072->32 condition projection and decoder input conv.",
    )
    parser.add_argument(
        "--early-learning-rate",
        type=float,
        default=5e-6,
        help="LR for stage0, transition01, stage1, and transition12.",
    )
    parser.add_argument(
        "--stage2-learning-rate", type=float, default=1e-5
    )
    parser.add_argument(
        "--transition23-learning-rate", type=float, default=1e-5
    )
    parser.add_argument(
        "--stage3-learning-rate", type=float, default=2e-5
    )
    parser.add_argument(
        "--head-learning-rate", type=float, default=3e-5
    )
    parser.add_argument(
        "--bypass-learning-rate",
        type=float,
        default=3e-5,
        help="LR for the new direct packed-RGB bypass; used only by condition-bypass.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    formal._validate_args(args)
    if args.variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if args.resize_mode not in RESIZE_MODES:
        raise ValueError(f"resize-mode must be one of {sorted(RESIZE_MODES)}")
    for name in (
        "condition_input_learning_rate",
        "early_learning_rate",
        "stage2_learning_rate",
        "transition23_learning_rate",
        "stage3_learning_rate",
        "head_learning_rate",
        "bypass_learning_rate",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")


def _append_module_parameters(
    destination: list[torch.nn.Parameter], module: torch.nn.Module
) -> None:
    destination.extend(list(module.parameters(recurse=True)))


def _set_joint_trainable(
    model: ResizeConvTinyConditionalDecoder,
    *,
    variant: str,
) -> tuple[dict[str, object], dict[str, list[torch.nn.Parameter]]]:
    """Jointly train the entire decoder with matched semantic optimizer groups."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layout = _decoder_layout(model)
    groups: dict[str, list[torch.nn.Parameter]] = {
        "condition_input": [],
        "early": [],
        "stage2": [],
        "transition23": [],
        "stage3": [],
        "output_head": [],
    }

    _append_module_parameters(groups["condition_input"], model.condition_projection)
    _append_module_parameters(groups["condition_input"], model.decoder[1])

    for layout_name in ("stage0", "transition01", "stage1", "transition12"):
        indices = layout[layout_name]
        assert isinstance(indices, tuple)
        for index in indices:
            _append_module_parameters(groups["early"], model.decoder[index])

    for group_name, layout_name in (
        ("stage2", "stage2"),
        ("transition23", "transition23"),
        ("stage3", "stage3"),
    ):
        indices = layout[layout_name]
        assert isinstance(indices, tuple)
        for index in indices:
            _append_module_parameters(groups[group_name], model.decoder[index])

    _append_module_parameters(groups["output_head"], model.output_head)

    if variant == "condition-bypass":
        if not isinstance(model, ConditionBypassResizeConvTinyConditionalDecoder):
            raise TypeError("condition-bypass variant requires ConditionBypass decoder")
        groups["condition_bypass"] = list(
            model.direct_condition_bypass.parameters(recurse=True)
        )
    elif isinstance(model, ConditionBypassResizeConvTinyConditionalDecoder):
        raise TypeError("r4 variant must not use the condition-bypass decoder class")

    # Every parameter in the selected architecture should be covered exactly once.
    ids_by_group = {name: {id(p) for p in values} for name, values in groups.items()}
    names = tuple(ids_by_group)
    for i, lhs in enumerate(names):
        for rhs in names[i + 1 :]:
            overlap = ids_by_group[lhs] & ids_by_group[rhs]
            if overlap:
                raise RuntimeError(f"Optimizer groups {lhs}/{rhs} overlap")

    all_grouped = set().union(*(values for values in ids_by_group.values()))
    all_model = {id(parameter) for parameter in model.parameters()}
    if all_grouped != all_model:
        name_by_id = {id(p): name for name, p in model.named_parameters()}
        missing = sorted(name_by_id[item] for item in all_model - all_grouped)
        extra = len(all_grouped - all_model)
        raise RuntimeError(
            f"Joint optimizer groups do not cover model exactly; missing={missing}, extra={extra}"
        )

    for parameters in groups.values():
        for parameter in parameters:
            parameter.requires_grad_(True)

    named = trainable_named_parameters(model)
    trainable_ids = {id(parameter) for _, parameter in named}
    if trainable_ids != all_model:
        raise RuntimeError("Full-joint trainable scope does not cover every model parameter")

    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    group_report: dict[str, object] = {}
    for group_name, parameters in groups.items():
        group_report[group_name] = {
            "parameter_tensors": len(parameters),
            "parameter_elements": sum(p.numel() for p in parameters),
            "parameter_names": [name_by_id[id(p)] for p in parameters],
        }

    report = {
        "scope": TRAINABLE_SCOPE,
        "variant": variant,
        "parameter_tensors": len(named),
        "parameter_elements": sum(p.numel() for _, p in named),
        "parameter_names": [name for name, _ in named],
        "layout": {
            key: list(value) if isinstance(value, tuple) else int(value)
            for key, value in layout.items()
        },
        "groups": group_report,
    }
    return report, groups


def _learning_rates(args: argparse.Namespace, variant: str) -> dict[str, float]:
    rates = {
        "condition_input": float(args.condition_input_learning_rate),
        "early": float(args.early_learning_rate),
        "stage2": float(args.stage2_learning_rate),
        "transition23": float(args.transition23_learning_rate),
        "stage3": float(args.stage3_learning_rate),
        "output_head": float(args.head_learning_rate),
    }
    if variant == "condition-bypass":
        rates["condition_bypass"] = float(args.bypass_learning_rate)
    return rates


def _build_grouped_adamw(
    groups: dict[str, list[torch.nn.Parameter]],
    args: argparse.Namespace,
    *,
    variant: str,
) -> torch.optim.AdamW:
    rates = _learning_rates(args, variant)
    param_groups: list[dict[str, object]] = []
    for name in rates:
        parameters = groups.get(name, [])
        if not parameters:
            raise RuntimeError(f"Optimizer group {name!r} has no parameters")
        bad = sorted({str(p.dtype) for p in parameters if p.dtype != torch.float32})
        if bad:
            raise RuntimeError(f"Optimizer group {name} must be FP32, found {bad}")
        param_groups.append(
            {"params": parameters, "lr": rates[name], "group_name": name}
        )
    return torch.optim.AdamW(
        param_groups,
        lr=float(args.condition_input_learning_rate),
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
            "variant": args.variant,
            "output_head": "resize_conv",
            "resize_mode": args.resize_mode,
            "trainable_scope": TRAINABLE_SCOPE,
            "parameter_group_learning_rates": _learning_rates(args, args.variant),
            "condition_injection": (
                CONDITION_BYPASS_MODE if args.variant == "condition-bypass" else "projected_rgb_32"
            ),
            "condition_dropout_probability": 0.0,
            "explicit_phase_loss": False,
        }
    )
    return fingerprint


def _load_initial_model(
    init_decoder: Path,
    *,
    variant: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[ResizeConvTinyConditionalDecoder, dict[str, object]]:
    if variant == "r4":
        model = ResizeConvTinyConditionalDecoder.from_pretrained(
            init_decoder, device=device, dtype=dtype
        )
        report = {
            "variant": "r4",
            "source_function_exact_at_initialization": True,
            "scheme": "identity_load_r4",
            "new_architecture_parameters": 0,
        }
        return model, report
    model, report = ConditionBypassResizeConvTinyConditionalDecoder.from_resizeconv_pretrained(
        init_decoder, device=device, dtype=dtype
    )
    return model, {"variant": "condition-bypass", **report}


def _load_resume_model(
    root: Path,
    *,
    variant: str,
    device: torch.device,
    dtype: torch.dtype,
) -> ResizeConvTinyConditionalDecoder:
    if variant == "r4":
        return ResizeConvTinyConditionalDecoder.from_pretrained(
            root, device=device, dtype=dtype
        )
    return ConditionBypassResizeConvTinyConditionalDecoder.from_pretrained(
        root, device=device, dtype=dtype
    )


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
            tiny = _load_resume_model(
                resume_checkpoint / "tiny_decoder",
                variant=args.variant,
                device=device,
                dtype=dtype,
            )
        else:
            tiny, initialization_report = _load_initial_model(
                init_decoder,
                variant=args.variant,
                device=device,
                dtype=dtype,
            )

        if tiny.block_mode != "compact":
            raise ValueError(f"Matched joint B1 requires compact decoder, got {tiny.block_mode!r}")
        if tuple(tiny.block_internal_channels or ()) != (80, 48, 24, 16):
            raise ValueError(
                "Matched joint B1 is frozen to keep_040 internal widths (80,48,24,16), "
                f"got {tiny.block_internal_channels}"
            )
        if tiny.resize_mode != args.resize_mode:
            raise ValueError(
                f"Checkpoint resize_mode={tiny.resize_mode!r} != requested {args.resize_mode!r}"
            )

        trainable_report, groups = _set_joint_trainable(tiny, variant=args.variant)
        cast_report = cast_trainable_parameters(tiny, dtype=torch.float32)
        tiny.train()
        optimizer = _build_grouped_adamw(groups, args, variant=args.variant)
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

        if args.resume is None:
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
                formal._write_json(run_dir / "validation_epoch_000.json", initial_val)
                formal._append_jsonl(
                    run_dir / "val_log.jsonl",
                    {"completed_epoch": 0, "global_step": 0, **initial_val},
                )
                print(
                    json.dumps(
                        {
                            "phase": "initial_matched_joint_validation",
                            "variant": args.variant,
                            **initial_val,
                        },
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
                            "variant": args.variant,
                            "trainable_scope": TRAINABLE_SCOPE,
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
            best_payload = [best_val_loss if rank == 0 else None]
            dist.broadcast_object_list(best_payload, src=0)
            best_val_loss = float(best_payload[0])
            dist.barrier()

        if rank == 0:
            best = json.loads((run_dir / formal.BEST_FILENAME).read_text(encoding="utf-8"))
            summary = {
                "status": "PASS",
                "variant": args.variant,
                "completed_epochs": int(args.epochs),
                "global_step": int(global_step),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "world_size": world_size,
                "local_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "resize_mode": args.resize_mode,
                "trainable_scope": TRAINABLE_SCOPE,
                "trainable_parameters": int(trainable_report["parameter_elements"]),
                "parameter_group_learning_rates": _learning_rates(args, args.variant),
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
