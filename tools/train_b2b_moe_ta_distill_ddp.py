#!/usr/bin/env python3
"""Train the D1024 1S12E2A student from the D1536 TA; validate vs Stage-A.

This is the compute-matched architecture race against the dense D768 student.
Training uses only cached B2-A D1536 endpoint velocity plus a small differentiable
router load-balance term.  Validation always uses the original Stage-A D3072
velocity cache and original frozen ReAE decoder.  GT remains diagnostic only.
"""

from __future__ import annotations

import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
for search_root in (ROOT, TOOLS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from tools import train_b2a_compact_distill_ddp as base
from tools import train_teacher_distillation_ddp as stage_a
from tools.smoke_training_forward import (
    _CANONICAL_DTYPE_NAME,
    configure_train_scope,
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.models import ReAE, WanTransformer3DModelPromptFreeNoTimeMoE
from swiftvr.training import (
    TeacherVelocityCache,
    append_jsonl,
    build_fp32_adamw,
    build_grad_scaler,
    cast_trainable_parameters,
    seed_everything,
    trainable_named_parameters,
    velocity_distillation_objective,
    write_latest_checkpoint,
)
from swiftvr.training.b2b_moe import B2BMoESpec, expected_moe_shape, transformer_moe_shape
from swiftvr.training.b2b_moe_training import B2BMoEVelocityDistillationForward, router_summary
from swiftvr.training.reference import sha256_file

TA_CACHE_KIND = "swiftvr_b2b_d1536_ta_velocity"
STAGE_A_CACHE_KIND = "swiftvr_b2a_stage_a_teacher_velocity"
LOCKED_SPEC = B2BMoESpec()


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--router-balance-weight",
        type=float,
        default=0.01,
        help="Weight on mean per-block Switch/Dense2MoE load-balance loss.",
    )
    parser.set_defaults(
        student_hidden_dim=LOCKED_SPEC.hidden_dim,
        student_num_heads=LOCKED_SPEC.num_heads,
        student_head_dim=LOCKED_SPEC.head_dim,
        student_ffn_dim=LOCKED_SPEC.active_ffn_dim,
        student_num_layers=LOCKED_SPEC.num_layers,
        student_adapter_dim=LOCKED_SPEC.adapter_dim,
    )
    return parser


def _validate_args(args) -> tuple[int, ...]:
    frame_indices = base._validate_args(args)
    if args.router_balance_weight < 0:
        raise ValueError("--router-balance-weight must be non-negative")
    locked = {
        "student_hidden_dim": LOCKED_SPEC.hidden_dim,
        "student_num_heads": LOCKED_SPEC.num_heads,
        "student_head_dim": LOCKED_SPEC.head_dim,
        "student_ffn_dim": LOCKED_SPEC.active_ffn_dim,
        "student_num_layers": LOCKED_SPEC.num_layers,
        "student_adapter_dim": LOCKED_SPEC.adapter_dim,
    }
    changed = {name: getattr(args, name) for name, expected in locked.items() if int(getattr(args, name)) != expected}
    if changed:
        raise ValueError(f"D1024-MoE architecture is locked for the race; changed args: {changed}")
    return frame_indices


def _gradient_summary_allow_sparse_experts(module) -> dict[str, object]:
    total_sq = 0.0
    gradient_tensors = 0
    nonfinite_elements = 0
    forbidden_missing: list[str] = []
    allowed_missing: list[str] = []
    for name, parameter in trainable_named_parameters(module):
        grad = parameter.grad
        if grad is None:
            if ".ffn.experts." in name:
                allowed_missing.append(name)
            else:
                forbidden_missing.append(name)
            continue
        gradient_tensors += 1
        value = grad.detach().float()
        finite = torch.isfinite(value)
        nonfinite_elements += int((~finite).sum().item())
        if bool(finite.any()):
            total_sq += float(value[finite].square().sum().item())
    return {
        "global_l2": math.sqrt(total_sq),
        "gradient_tensors": gradient_tensors,
        "nonfinite_elements": nonfinite_elements,
        "allowed_missing_sparse_experts": len(allowed_missing),
        "allowed_missing_examples": allowed_missing[:8],
        "forbidden_missing": len(forbidden_missing),
        "forbidden_missing_examples": forbidden_missing[:8],
    }


def _router_metrics_from_counts(counts: list[float], assignments: float, mean_entropy: float) -> dict[str, float]:
    fractions = [value / max(assignments, 1.0) for value in counts]
    mean = sum(fractions) / len(fractions)
    variance = sum((value - mean) ** 2 for value in fractions) / len(fractions)
    result = {
        "router_assignment_count": float(assignments),
        "router_entropy": float(mean_entropy),
        "router_normalized_entropy": float(mean_entropy / math.log(len(fractions))),
        "router_min_fraction": float(min(fractions)),
        "router_max_fraction": float(max(fractions)),
        "router_load_cv": float(math.sqrt(variance) / max(mean, 1e-12)),
    }
    for index, value in enumerate(fractions):
        result[f"router_expert_fraction_{index:02d}"] = float(value)
    return result


def main() -> int:
    args = build_parser().parse_args()
    visual_frame_indices = _validate_args(args)
    rank, local_rank, world_size, device = stage_a.init_distributed()
    writer = None
    try:
        effective_batch = world_size * args.batch_size * args.gradient_accumulation_steps
        if args.expected_global_batch_size is not None and effective_batch != args.expected_global_batch_size:
            raise ValueError(f"Global effective batch={effective_batch}, expected={args.expected_global_batch_size}")

        run_dir = args.output_dir.expanduser().resolve()
        if rank == 0:
            if (run_dir / "train_log.jsonl").exists() or (run_dir / "latest.json").exists():
                raise FileExistsError("Output directory already contains a MoE TA run")
            run_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        base_root = args.base_checkpoint.expanduser().resolve()
        student_root = args.student_init.expanduser().resolve()
        folded_config = validate_folded_checkpoint(
            base_root,
            reae_filename=args.reae_filename,
            transformer_subfolder=args.transformer_subfolder,
        )
        dtype = resolve_runtime_dtype(args.dtype, folded_config, device, allow_mismatch=args.allow_dtype_mismatch)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")
        seed_everything(args.seed + rank)

        train_cache = TeacherVelocityCache(args.teacher_cache)
        if train_cache.metadata.get("kind") != TA_CACHE_KIND:
            raise ValueError(f"MoE TA training requires {TA_CACHE_KIND!r}, got {train_cache.metadata.get('kind')!r}")
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
            if val_cache.metadata.get("kind") != STAGE_A_CACHE_KIND:
                raise ValueError("MoE validation must use the original Stage-A D3072 cache")
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
        transformer = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
            str(student_root),
            subfolder=args.transformer_subfolder,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        shape = transformer_moe_shape(transformer)
        if shape != expected_moe_shape(LOCKED_SPEC):
            raise ValueError(f"student-init MoE shape mismatch: {shape} != {expected_moe_shape(LOCKED_SPEC)}")

        train_scope = configure_train_scope(reae, transformer, "transformer")
        reae.to(device=device, dtype=dtype).eval()
        transformer.to(device=device, dtype=dtype)
        closure = B2BMoEVelocityDistillationForward(
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
        initial_grad_scale = float(scaler.get_scale())
        ddp_model = DistributedDataParallel(
            closure,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
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
            "trainer": "b2b_d1024_moe_d1536_ta_distill_ddp_v1",
            "experiment": "d1024_1s12e2a_vs_d768_compute_matched_race",
            "base_checkpoint": str(base_root),
            "student_init": str(student_root),
            "student_shape": shape,
            "student_parameters": sum(parameter.numel() for parameter in transformer.parameters()),
            "teacher_cache": str(args.teacher_cache.expanduser().resolve()),
            "training_teacher": "b2a_d1536_teaching_assistant",
            "training_teacher_cache_kind": TA_CACHE_KIND,
            "val_teacher_cache": None if args.val_teacher_cache is None else str(args.val_teacher_cache.expanduser().resolve()),
            "validation_teacher": "stage_a_d3072_reference",
            "validation_teacher_cache_kind": STAGE_A_CACHE_KIND,
            "training_decoder": "none",
            "validation_decoder": "original_frozen_reae",
            "gt_role": "diagnostic_only",
            "checkpoint_selection_metric": "stage_a_velocity_relative_l2",
            "router_balance_weight": args.router_balance_weight,
            "router_balance_definition": "mean per-block Dense2MoE/Switch load-balance loss; balanced optimum is 1.0",
            "router_no_capacity_dropping": True,
            "ddp_find_unused_parameters": True,
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
            "amp_dynamic_loss_scaling": bool(scaler.is_enabled()),
            "amp_initial_scale": initial_grad_scale,
            "max_amp_overflow_retries": args.max_amp_overflow_retries,
            "visuals_enabled": visuals_enabled,
            "visualize_every": visualize_every,
        }
        if rank == 0:
            base._write_json(run_dir / "run_config.json", run_config)
        dist.barrier()

        train_log = run_dir / "train_log.jsonl"
        val_log = run_dir / "val_log.jsonl"
        overflow_log = run_dir / "amp_overflow_log.jsonl"
        best_relative_l2 = float("inf")
        best_step = None
        global_step = 0
        epoch = 0
        amp_overflow_count = 0
        last_record: dict[str, object] = {}
        started = time.perf_counter()

        def run_validation(step: int):
            nonlocal best_relative_l2, best_step
            if val_loader is None or val_cache is None:
                return None
            visual_due = visuals_enabled and (step == 0 or step % visualize_every == 0 or step == args.max_steps)
            with torch.inference_mode():
                validation, visual_samples = base.validate_rank0(
                    closure,
                    val_loader,
                    val_cache,
                    device=device,
                    dtype=dtype,
                    visual_samples=args.visual_validation_samples if visual_due else 0,
                )
            append_jsonl(val_log, {"global_step": step, **validation})
            base._write_validation_scalars(writer, step, validation)
            if visual_due:
                report = base.export_validation_visuals(
                    visual_samples,
                    output_root=run_dir,
                    step=step,
                    frame_indices=visual_frame_indices,
                    fps=args.visual_video_fps,
                    difference_scale=args.visual_difference_scale,
                    writer=writer,
                )
                if report["video_errors"]:
                    print("validation visual warnings: " + json.dumps(report["video_errors"]), flush=True)
            value = float(validation["velocity_relative_l2"])
            if value < best_relative_l2:
                best_relative_l2 = value
                best_step = step
                base._write_json(
                    run_dir / "best.json",
                    {
                        "global_step": step,
                        "velocity_relative_l2": value,
                        "velocity_cosine": float(validation["velocity_cosine"]),
                        "note": "Best selected only by Stage-A velocity rel-L2; RGB/GT are diagnostic.",
                    },
                )
            return validation

        validation_configured = bool(args.val_manifest and args.val_teacher_cache)
        if args.validate_at_start and validation_configured:
            dist.barrier()
            if rank == 0:
                baseline = run_validation(0)
                print(
                    f"MoE init Stage-A rel_l2={baseline['velocity_relative_l2']:.6f} "
                    f"cos={baseline['velocity_cosine']:.6f} "
                    f"rgb_teacher_psnr={baseline['student_teacher_psnr']:.4f}",
                    flush=True,
                )
            dist.barrier()

        autocast_enabled = dtype in (torch.float16, torch.bfloat16)
        while global_step < args.max_steps:
            loader = stage_a.make_train_loader(train_dataset, rank=rank, world_size=world_size, epoch=epoch, args=args)
            if len(loader) < args.gradient_accumulation_steps:
                raise RuntimeError("Per-rank epoch is shorter than gradient accumulation")
            iterator = iter(loader)

            while global_step < args.max_steps:
                step_started = time.perf_counter()
                next_step = global_step + 1
                current_lr = base._lr_for_step(args, next_step)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

                micro_batches_cpu: list[Mapping[str, object]] = []
                for _ in range(args.gradient_accumulation_steps):
                    try:
                        micro_batches_cpu.append(next(iterator))
                    except StopIteration:
                        break
                if len(micro_batches_cpu) != args.gradient_accumulation_steps:
                    optimizer.zero_grad(set_to_none=True)
                    break

                amp_retry = 0
                while True:
                    optimizer.zero_grad(set_to_none=True)
                    sums: dict[str, float] = {}
                    route_counts = [0.0] * LOCKED_SPEC.num_experts
                    route_assignments = 0.0
                    route_entropy_sum = 0.0
                    route_observations = 0

                    for micro_index, batch_cpu in enumerate(micro_batches_cpu):
                        teacher_velocity = train_cache.load_batch(batch_cpu, device=device, dtype=dtype)
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
                                total_loss = objective["loss"] + args.router_balance_weight * output["router_balance_loss"]
                            if not torch.isfinite(total_loss.detach()).item():
                                raise FloatingPointError("Non-finite MoE TA distillation loss")
                            scaled_loss = total_loss / args.gradient_accumulation_steps
                            if scaler.is_enabled():
                                scaler.scale(scaled_loss).backward()
                            else:
                                scaled_loss.backward()

                        summary = router_summary(transformer)
                        for index, value in enumerate(summary["expert_counts"]):
                            route_counts[index] += float(value)
                        route_assignments += float(summary["assignments"])
                        route_entropy_sum += float(summary["mean_entropy"])
                        route_observations += 1

                        sums["loss"] = sums.get("loss", 0.0) + float(total_loss.detach().float().item())
                        sums["velocity_loss"] = sums.get("velocity_loss", 0.0) + float(objective["loss"].detach().float().item())
                        sums["router_balance_loss"] = sums.get("router_balance_loss", 0.0) + float(output["router_balance_loss"].detach().float().item())
                        for key in (
                            "velocity_mse",
                            "velocity_normalized_mse",
                            "velocity_cosine",
                            "velocity_cosine_loss",
                            "teacher_velocity_power",
                        ):
                            sums[key] = sums.get(key, 0.0) + float(objective[key].detach().float().item())

                    scale_before = float(scaler.get_scale())
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    gradients = _gradient_summary_allow_sparse_experts(closure)
                    if int(gradients["gradient_tensors"]) == 0:
                        raise RuntimeError("Backward produced no trainable gradients")
                    if int(gradients["forbidden_missing"]) != 0:
                        raise RuntimeError(f"Missing non-expert gradients: {gradients['forbidden_missing_examples']}")

                    local_nonfinite = int(gradients["nonfinite_elements"])
                    nonfinite_flag = torch.tensor([1 if local_nonfinite else 0], device=device, dtype=torch.int32)
                    dist.all_reduce(nonfinite_flag, op=dist.ReduceOp.MAX)
                    global_nonfinite = int(nonfinite_flag.item()) != 0
                    if global_nonfinite and not scaler.is_enabled():
                        raise FloatingPointError("Non-finite gradients without FP16 GradScaler")

                    grad_norm = float("nan")
                    if not global_nonfinite:
                        grad_norm = float(gradients["global_l2"])
                        if args.max_grad_norm > 0:
                            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                                [parameter for _, parameter in trainable_named_parameters(closure)],
                                max_norm=args.max_grad_norm,
                                error_if_nonfinite=True,
                            ).float().item())

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scale_after = float(scaler.get_scale())

                    if global_nonfinite:
                        if not scale_after < scale_before:
                            raise FloatingPointError("FP16 overflow occurred but GradScaler did not back off")
                        amp_overflow_count += 1
                        optimizer.zero_grad(set_to_none=True)
                        if rank == 0:
                            append_jsonl(overflow_log, {
                                "attempted_global_step": next_step,
                                "retry_index": amp_retry + 1,
                                "scale_before": scale_before,
                                "scale_after": scale_after,
                            })
                        if amp_retry >= args.max_amp_overflow_retries:
                            raise FloatingPointError("Exceeded AMP overflow retry limit")
                        amp_retry += 1
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    break

                global_step += 1
                scalar_keys = tuple(sums)
                packed = torch.tensor(
                    [sums[key] for key in scalar_keys] + [grad_norm, time.perf_counter() - step_started],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                denominator = args.gradient_accumulation_steps * world_size
                averages = {key: float(packed[index].item()) / denominator for index, key in enumerate(scalar_keys)}
                grad_norm_global = float(packed[-2].item()) / world_size
                step_seconds = float(packed[-1].item()) / world_size

                route_pack = torch.tensor(
                    route_counts + [route_assignments, route_entropy_sum, float(route_observations)],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(route_pack, op=dist.ReduceOp.SUM)
                global_counts = [float(value) for value in route_pack[: LOCKED_SPEC.num_experts].tolist()]
                global_assignments = float(route_pack[-3].item())
                entropy_observations = max(float(route_pack[-1].item()), 1.0)
                mean_entropy = float(route_pack[-2].item()) / entropy_observations
                route_metrics = _router_metrics_from_counts(global_counts, global_assignments, mean_entropy)

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
                    **route_metrics,
                    "allowed_missing_sparse_expert_gradients": int(gradients["allowed_missing_sparse_experts"]),
                    "gradient_norm": grad_norm_global,
                    "learning_rate": current_lr,
                    "grad_scale": float(scaler.get_scale()),
                    "amp_retries_this_step": amp_retry,
                    "amp_overflow_count": amp_overflow_count,
                    "step_seconds": step_seconds,
                    "peak_allocated_gb_per_rank": torch.cuda.max_memory_allocated(device) / 1024**3,
                }
                if rank == 0:
                    append_jsonl(train_log, last_record)
                    if global_step % args.log_every == 0:
                        print(
                            f"step={global_step} loss={averages['loss']:.7f} "
                            f"vel={averages['velocity_loss']:.7f} rel_l2={relative_l2:.6f} "
                            f"cos={averages['velocity_cosine']:.6f} bal={averages['router_balance_loss']:.4f} "
                            f"H={route_metrics['router_normalized_entropy']:.3f} "
                            f"load=[{route_metrics['router_min_fraction']:.3f},{route_metrics['router_max_fraction']:.3f}] "
                            f"cv={route_metrics['router_load_cv']:.3f} lr={current_lr:.3e} "
                            f"time={step_seconds:.3f}s peak={last_record['peak_allocated_gb_per_rank']:.2f}GB",
                            flush=True,
                        )
                    if writer is not None:
                        for key, value in last_record.items():
                            if isinstance(value, (int, float)) and key != "global_step":
                                writer.add_scalar("train/" + key, float(value), global_step)
                        writer.flush()

                validation_due = validation_configured and args.validate_every > 0 and (
                    global_step % args.validate_every == 0 or global_step == args.max_steps
                )
                if validation_due:
                    dist.barrier()
                    if rank == 0:
                        validation = run_validation(global_step)
                        print(
                            f"validation step={global_step} rel_l2={validation['velocity_relative_l2']:.6f} "
                            f"cos={validation['velocity_cosine']:.6f} "
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
                        base._save_snapshot(
                            transformer,
                            checkpoint,
                            runtime_dtype=dtype,
                            transformer_subfolder=args.transformer_subfolder,
                            metadata={
                                "trainer": "b2b_d1024_moe_d1536_ta_distill_ddp_v1",
                                "global_step": global_step,
                                "source_student_init": str(student_root),
                                "student_shape": shape,
                                "router_balance_weight": args.router_balance_weight,
                                "velocity_relative_l2_train_to_ta": relative_l2,
                                "best_stage_a_validation_velocity_relative_l2": None if best_relative_l2 == float("inf") else best_relative_l2,
                                "best_stage_a_validation_step": best_step,
                                "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
                                "amp_overflow_count": amp_overflow_count,
                                "optimizer_state_saved": False,
                            },
                        )
                        write_latest_checkpoint(run_dir, checkpoint)
                        if best_step == global_step and (run_dir / "best.json").is_file():
                            best_info = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
                            best_info["checkpoint"] = str(checkpoint.relative_to(run_dir))
                            base._write_json(run_dir / "best.json", best_info)
                        print(f"saved MoE snapshot: {checkpoint}", flush=True)
                    dist.barrier()
            epoch += 1

        if rank == 0:
            summary = {
                "status": "PASS",
                "global_step": global_step,
                "max_steps": args.max_steps,
                "elapsed_seconds": time.perf_counter() - started,
                "best_stage_a_velocity_relative_l2": None if best_relative_l2 == float("inf") else best_relative_l2,
                "best_step": best_step,
                "last_record": last_record,
                "run_dir": str(run_dir),
                "student_shape": shape,
                "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
                "world_size": world_size,
                "amp_initial_scale": initial_grad_scale,
                "amp_final_scale": float(scaler.get_scale()),
                "amp_overflow_count": amp_overflow_count,
            }
            base._write_json(run_dir / "summary.json", summary)
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
