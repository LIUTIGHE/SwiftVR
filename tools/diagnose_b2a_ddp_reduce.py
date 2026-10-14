#!/usr/bin/env python3
"""Diagnose B2-A non-finite gradients inside a real DDP backward.

This is a read-only one-step probe.  It reproduces the trainer's epoch-0
DistributedSampler, compact model, autocast, optional GradScaler, DDP wrapper,
and gradient-as-bucket-view setting.  A custom synchronous communication hook
records each gradient bucket immediately before and after NCCL all-reduce so we
can distinguish local backward failure from reduction/bucket failure.

No optimizer step is executed and no model checkpoint is written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_teacher_distillation_ddp as stage_a
from diagnose_b2a_backward import gradient_report, tensor_stats
from smoke_training_forward import (
    _CANONICAL_DTYPE_NAME,
    configure_train_scope,
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training import (
    TeacherVelocityCache,
    build_fp32_adamw,
    build_grad_scaler,
    cast_trainable_parameters,
    seed_everything,
    velocity_distillation_objective,
)
from swiftvr.training.b2a_width import (
    B2ACompactVelocityDistillationForward,
    B2AWidthSpec,
    transformer_width_shape,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--student-init", type=Path, required=True)
    p.add_argument("--teacher-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--split", default="train")
    p.add_argument("--clip-length", type=int, default=13)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--views-per-record", type=int, default=8)
    p.add_argument("--view-seed", type=int, default=20260805)
    p.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    p.add_argument("--vertical-flip-probability", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--allow-dtype-mismatch", action="store_true")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--no-gradient-as-bucket-view", action="store_true")
    p.add_argument(
        "--disable-loss-scaling",
        action="store_true",
        help="For diagnosis only: bypass GradScaler even when runtime dtype is FP16.",
    )
    p.add_argument("--loss-epsilon", type=float, default=1e-8)
    p.add_argument("--max-gradient-examples", type=int, default=24)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--optimizer-eps", type=float, default=1e-8)
    p.add_argument("--output-dir", type=Path, required=True)
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


def _bucket_stats(value: torch.Tensor) -> dict[str, object]:
    data = value.detach().float()
    finite = torch.isfinite(data)
    finite_values = data[finite]
    result: dict[str, object] = {
        "elements": int(data.numel()),
        "dtype": str(value.dtype).removeprefix("torch."),
        "nonfinite": int((~finite).sum().item()),
        "nan": int(torch.isnan(data).sum().item()),
        "posinf": int(torch.isposinf(data).sum().item()),
        "neginf": int(torch.isneginf(data).sum().item()),
    }
    if finite_values.numel():
        squared = finite_values.double().square().sum()
        result.update(
            {
                "finite_l2": math.sqrt(float(squared.item())),
                "finite_max_abs": float(finite_values.abs().max().item()),
            }
        )
    return result


def _batch_indices(batch: Mapping[str, object]) -> list[int]:
    value = batch.get("distillation_index")
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise TypeError("Expected collated distillation_index tensor [B]")
    return [int(item) for item in value.tolist()]


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.max_gradient_examples <= 0:
        raise ValueError("batch-size and max-gradient-examples must be positive")

    rank, local_rank, world_size, device = stage_a.init_distributed()
    output_root = args.output_dir.expanduser().resolve()
    try:
        base_root = args.base_checkpoint.expanduser().resolve()
        student_root = args.student_init.expanduser().resolve()
        folded_config = validate_folded_checkpoint(
            base_root,
            reae_filename=args.reae_filename,
            transformer_subfolder=args.transformer_subfolder,
        )
        dtype = resolve_runtime_dtype(
            args.dtype,
            folded_config,
            device,
            allow_mismatch=args.allow_dtype_mismatch,
        )
        seed_everything(args.seed + rank)

        cache = TeacherVelocityCache(args.teacher_cache)
        if cache.metadata.get("kind") != "swiftvr_b2a_stage_a_teacher_velocity":
            raise ValueError(
                "Expected B2-A Stage-A teacher cache, got "
                f"{cache.metadata.get('kind')!r}"
            )
        dataset = stage_a.build_cached_dataset(
            args.manifest,
            cache,
            split=args.split,
            path_root=args.path_root,
            clip_length=args.clip_length,
            crop_size=args.crop_size,
            scale=args.scale,
            views_per_record=args.views_per_record,
            view_seed=args.view_seed,
            hflip=args.horizontal_flip_probability,
            vflip=args.vertical_flip_probability,
            verify_paths=False,
        )
        loader = stage_a.make_train_loader(
            dataset,
            rank=rank,
            world_size=world_size,
            epoch=0,
            args=args,
        )
        try:
            batch_cpu = next(iter(loader))
        except StopIteration as exc:
            raise RuntimeError("Epoch-0 DDP loader produced no batch") from exc
        indices = _batch_indices(batch_cpu)
        teacher_velocity = cache.load_batch(batch_cpu, device=device, dtype=dtype)
        batch = move_video_batch(batch_cpu, device=device, dtype=dtype)

        reae = ReAE(str(base_root / args.reae_filename))
        transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
            str(student_root),
            subfolder=args.transformer_subfolder,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        spec = _spec(args)
        expected_shape = {
            "hidden_dim": spec.hidden_dim,
            "num_heads": spec.num_heads,
            "head_dim": spec.head_dim,
            "ffn_dim": spec.ffn_dim,
            "num_layers": spec.num_layers,
            "adapter_dim": spec.adapter_dim,
        }
        shape = transformer_width_shape(transformer)
        if shape != expected_shape:
            raise ValueError(f"student shape mismatch: {shape} != {expected_shape}")

        configure_train_scope(reae, transformer, "transformer")
        reae.to(device=device, dtype=dtype).eval()
        transformer.to(device=device, dtype=dtype)
        closure = B2ACompactVelocityDistillationForward(
            reae,
            transformer,
            attention_backend=args.attention_backend,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        ).to(device)
        closure.train()
        closure.reae.eval()
        cast_summary = cast_trainable_parameters(closure, dtype=torch.float32)
        optimizer = build_fp32_adamw(
            closure,
            learning_rate=args.learning_rate,
            weight_decay=0.0,
            eps=args.optimizer_eps,
        )
        scaler = build_grad_scaler(device, dtype)
        scaler_active = bool(scaler.is_enabled() and not args.disable_loss_scaling)

        ddp = DistributedDataParallel(
            closure,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=not args.no_gradient_as_bucket_view,
        )

        parameter_name_by_id = {
            id(parameter): name for name, parameter in closure.named_parameters()
        }
        bucket_records: list[dict[str, object]] = []

        def diagnostic_allreduce(_state, bucket):
            buffer = bucket.buffer()
            record: dict[str, object] = {
                "bucket_index": int(bucket.index()),
                "is_last": bool(bucket.is_last()),
                "pre_reduce": _bucket_stats(buffer),
            }
            try:
                parameters = bucket.parameters()
            except Exception:
                parameters = []
            names = [parameter_name_by_id.get(id(parameter), "<unknown>") for parameter in parameters]
            record["parameter_count"] = len(names)
            record["first_parameter"] = names[0] if names else None
            record["last_parameter"] = names[-1] if names else None

            # Synchronous on purpose: this diagnostic prioritizes attribution over speed.
            dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            buffer.div_(world_size)
            record["post_reduce"] = _bucket_stats(buffer)
            bucket_records.append(record)
            future: torch.futures.Future[torch.Tensor] = torch.futures.Future()
            future.set_result(buffer)
            return future

        ddp.register_comm_hook(None, diagnostic_allreduce)

        autocast_enabled = dtype in (torch.float16, torch.bfloat16)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda",
            dtype=dtype,
            enabled=device.type == "cuda" and autocast_enabled,
        ):
            output = ddp(batch)
        objective = velocity_distillation_objective(
            output["velocity"],
            teacher_velocity,
            velocity_mse_weight=1.0,
            velocity_cosine_weight=1.0,
            output_l1_weight=0.0,
            output_temporal_weight=0.0,
            gt_loss_mode="none",
            gt_pixel_weight=0.0,
            gt_temporal_weight=0.0,
            epsilon=args.loss_epsilon,
        )
        if not torch.isfinite(objective["loss"].detach()).item():
            raise FloatingPointError("DDP diagnostic forward produced non-finite loss")

        scale_before = float(scaler.get_scale())
        if scaler_active:
            scaler.scale(objective["loss"]).backward()
            scaler.unscale_(optimizer)
        else:
            objective["loss"].backward()
        gradients = gradient_report(closure, args.max_gradient_examples)

        local_pre_nonfinite = sum(
            int(record["pre_reduce"]["nonfinite"]) for record in bucket_records
        )
        post_reduce_nonfinite = sum(
            int(record["post_reduce"]["nonfinite"]) for record in bucket_records
        )
        report: dict[str, object] = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "indices": indices,
            "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
            "grad_scaler_enabled_by_dtype": bool(scaler.is_enabled()),
            "loss_scaling_active": scaler_active,
            "grad_scale": scale_before,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "gradient_as_bucket_view": not args.no_gradient_as_bucket_view,
            "student_shape": shape,
            "cast_trainable_parameters": cast_summary,
            "teacher_velocity": tensor_stats(teacher_velocity),
            "student_velocity": tensor_stats(output["velocity"]),
            "loss": float(objective["loss"].detach().float().item()),
            "velocity_normalized_mse": float(
                objective["velocity_normalized_mse"].detach().float().item()
            ),
            "velocity_cosine": float(objective["velocity_cosine"].detach().float().item()),
            "bucket_count": len(bucket_records),
            "pre_reduce_nonfinite_elements": local_pre_nonfinite,
            "post_reduce_nonfinite_elements": post_reduce_nonfinite,
            "buckets": bucket_records,
            "final_unscaled_gradients": gradients,
        }
        if local_pre_nonfinite:
            status = "LOCAL_BACKWARD_NONFINITE"
        elif post_reduce_nonfinite:
            status = "REDUCE_NONFINITE"
        elif int(gradients["nonfinite_elements"]):
            status = "POST_REDUCE_OR_UNSCALE_NONFINITE"
        else:
            status = "PASS"
        report["status"] = status

        output_root.mkdir(parents=True, exist_ok=True)
        rank_path = output_root / f"rank_{rank}.json"
        _write_json(rank_path, report)
        print(
            f"rank={rank} indices={indices} dtype={report['runtime_dtype']} "
            f"scaler={scaler_active} scale={scale_before:g} buckets={len(bucket_records)} "
            f"pre_nf={local_pre_nonfinite} post_nf={post_reduce_nonfinite} "
            f"final_nf={gradients['nonfinite_elements']} status={status}",
            flush=True,
        )
        dist.barrier()

        if rank == 0:
            summaries = []
            for other_rank in range(world_size):
                payload = json.loads((output_root / f"rank_{other_rank}.json").read_text(encoding="utf-8"))
                summaries.append(
                    {
                        "rank": other_rank,
                        "indices": payload["indices"],
                        "runtime_dtype": payload["runtime_dtype"],
                        "loss_scaling_active": payload["loss_scaling_active"],
                        "grad_scale": payload["grad_scale"],
                        "pre_reduce_nonfinite_elements": payload["pre_reduce_nonfinite_elements"],
                        "post_reduce_nonfinite_elements": payload["post_reduce_nonfinite_elements"],
                        "final_nonfinite_elements": payload["final_unscaled_gradients"]["nonfinite_elements"],
                        "status": payload["status"],
                    }
                )
            _write_json(
                output_root / "summary.json",
                {
                    "world_size": world_size,
                    "gradient_checkpointing": not args.no_gradient_checkpointing,
                    "gradient_as_bucket_view": not args.no_gradient_as_bucket_view,
                    "disable_loss_scaling": bool(args.disable_loss_scaling),
                    "ranks": summaries,
                },
            )
            print(json.dumps(summaries, indent=2), flush=True)
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
