#!/usr/bin/env python3
"""Scan selected B2-A dataset indices for non-finite combined-loss gradients.

Read-only diagnostic. The compact student is loaded once. Selected deterministic
training views are evaluated in the provided order and grouped by --batch-size.
Each group runs one combined normalized-MSE + cosine backward from the same
initial student weights; no optimizer step or checkpoint write is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader, Subset

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
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training import (
    TeacherVelocityCache,
    cast_trainable_parameters,
    seed_everything,
    velocity_distillation_objective,
)
from swiftvr.training.b2a_width import (
    B2ACompactVelocityDistillationForward,
    B2AWidthSpec,
    transformer_width_shape,
)


def _csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("indices must be a non-empty comma-separated list")
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("indices must be non-negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--student-init", type=Path, required=True)
    p.add_argument("--teacher-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--indices", type=_csv_ints, required=True)
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--allow-dtype-mismatch", action="store_true")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--loss-epsilon", type=float, default=1e-8)
    p.add_argument("--max-gradient-examples", type=int, default=16)
    p.add_argument("--output-json", type=Path, default=None)
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


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _batch_indices(batch: Mapping[str, object]) -> list[int]:
    value = batch.get("distillation_index")
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected collated distillation_index tensor")
    return [int(item) for item in value.tolist()]


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.max_gradient_examples <= 0:
        raise ValueError("batch-size and max-gradient-examples must be positive")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Selected-backward scan currently requires CUDA")

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
    seed_everything(args.seed)

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
    if any(index >= len(dataset) for index in args.indices):
        raise IndexError(
            f"Selected index exceeds cached dataset length={len(dataset)}: {args.indices}"
        )
    selected = Subset(dataset, args.indices)
    loader = DataLoader(
        selected,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    reae = ReAE(str(base_root / args.reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(student_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    spec = _spec(args)
    shape = transformer_width_shape(transformer)
    expected = {
        "hidden_dim": spec.hidden_dim,
        "num_heads": spec.num_heads,
        "head_dim": spec.head_dim,
        "ffn_dim": spec.ffn_dim,
        "num_layers": spec.num_layers,
        "adapter_dim": spec.adapter_dim,
    }
    if shape != expected:
        raise ValueError(f"student shape mismatch: {shape} != {expected}")

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
    cast_trainable_parameters(closure, dtype=torch.float32)

    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    report: dict[str, object] = {
        "status": "PASS",
        "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "attention_backend": args.attention_backend,
        "requested_indices": list(args.indices),
        "batch_size": int(args.batch_size),
        "groups": [],
    }

    for group_index, batch_cpu in enumerate(loader):
        closure.zero_grad(set_to_none=True)
        teacher_velocity = cache.load_batch(batch_cpu, device=device, dtype=dtype)
        batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
        indices = _batch_indices(batch_cpu)
        torch.cuda.reset_peak_memory_stats(device)

        with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            output = closure(batch)
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
        group: dict[str, object] = {
            "group_index": group_index,
            "indices": indices,
            "teacher_velocity": tensor_stats(teacher_velocity),
            "student_velocity": tensor_stats(output["velocity"]),
            "loss": float(objective["loss"].detach().float().item()),
            "velocity_normalized_mse": float(
                objective["velocity_normalized_mse"].detach().float().item()
            ),
            "velocity_cosine": float(objective["velocity_cosine"].detach().float().item()),
        }

        if not torch.isfinite(objective["loss"].detach()).item():
            group["backward"] = "SKIPPED_NONFINITE_LOSS"
            group["gradients"] = None
            report["status"] = "FAIL"
        else:
            objective["loss"].backward()
            gradients = gradient_report(closure, args.max_gradient_examples)
            group["gradients"] = gradients
            nonfinite = int(gradients["nonfinite_elements"])
            group["backward"] = "PASS" if nonfinite == 0 else "NONFINITE"
            if nonfinite:
                report["status"] = "FAIL"

        group["peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
        report["groups"].append(group)
        gradients = group.get("gradients")
        nonfinite = 0 if gradients is None else int(gradients["nonfinite_elements"])
        finite_l2 = None if gradients is None else float(gradients["finite_global_l2"])
        print(
            f"group={group_index} indices={indices} "
            f"loss={group['loss']:.8f} "
            f"norm_mse={group['velocity_normalized_mse']:.8f} "
            f"cos={group['velocity_cosine']:.8f} "
            f"backward={group['backward']} "
            f"nonfinite_grad={nonfinite} finite_grad_l2={finite_l2}",
            flush=True,
        )
        if nonfinite:
            print(
                json.dumps(gradients["nonfinite_parameter_examples"], indent=2),
                flush=True,
            )

    if args.output_json is not None:
        _write_json(args.output_json, report)
    print(json.dumps({"status": report["status"], "groups": len(report["groups"])}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
