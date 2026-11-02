#!/usr/bin/env python3
"""Diagnose non-finite B2-A backward on one real cached-teacher batch.

Read-only diagnostic: no optimizer step and no checkpoint write.  It runs the
same compact DiT training forward as B2-A and evaluates independent backward
passes for normalized velocity MSE, cosine loss, and their sum.  The report
identifies the exact trainable parameters containing NaN/Inf gradients.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_teacher_distillation_ddp as stage_a
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
    cast_trainable_parameters,
    seed_everything,
    trainable_named_parameters,
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
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--allow-dtype-mismatch", action="store_true")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--loss-epsilon", type=float, default=1e-8)
    p.add_argument("--max-gradient-examples", type=int, default=24)
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


def tensor_stats(value: torch.Tensor) -> dict[str, object]:
    data = value.detach().float()
    finite = torch.isfinite(data)
    finite_values = data[finite]
    result: dict[str, object] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "elements": int(value.numel()),
        "nonfinite": int((~finite).sum().item()),
        "nan": int(torch.isnan(data).sum().item()),
        "posinf": int(torch.isposinf(data).sum().item()),
        "neginf": int(torch.isneginf(data).sum().item()),
    }
    if finite_values.numel():
        result.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "rms": float(torch.sqrt(finite_values.square().mean()).item()),
                "max_abs": float(finite_values.abs().max().item()),
            }
        )
    return result


def gradient_report(module: torch.nn.Module, limit: int) -> dict[str, object]:
    examples: list[dict[str, object]] = []
    missing: list[str] = []
    total_nonfinite = 0
    nan = 0
    posinf = 0
    neginf = 0
    tensors = 0
    finite_sq = 0.0
    finite_max = 0.0

    for name, parameter in trainable_named_parameters(module):
        grad = parameter.grad
        if grad is None:
            missing.append(name)
            continue
        tensors += 1
        g = grad.detach().float()
        finite = torch.isfinite(g)
        nf = int((~finite).sum().item())
        n_nan = int(torch.isnan(g).sum().item())
        n_pos = int(torch.isposinf(g).sum().item())
        n_neg = int(torch.isneginf(g).sum().item())
        total_nonfinite += nf
        nan += n_nan
        posinf += n_pos
        neginf += n_neg
        if finite.any():
            values = g[finite]
            finite_sq += float(values.square().sum().item())
            finite_max = max(finite_max, float(values.abs().max().item()))
        if nf and len(examples) < limit:
            examples.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "grad_dtype": str(grad.dtype).removeprefix("torch."),
                    "nonfinite": nf,
                    "nan": n_nan,
                    "posinf": n_pos,
                    "neginf": n_neg,
                    "finite_max_abs": (
                        float(g[finite].abs().max().item()) if finite.any() else None
                    ),
                }
            )

    return {
        "gradient_tensors": tensors,
        "missing_gradient_count": len(missing),
        "missing_gradient_examples": missing[:limit],
        "nonfinite_elements": total_nonfinite,
        "nan_elements": nan,
        "posinf_elements": posinf,
        "neginf_elements": neginf,
        "finite_global_l2": math.sqrt(finite_sq),
        "finite_max_abs": finite_max,
        "nonfinite_parameter_examples": examples,
    }


def _take_batch(loader: DataLoader, index: int):
    if index < 0:
        raise ValueError("batch-index must be non-negative")
    iterator = iter(loader)
    for _ in range(index + 1):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise IndexError(f"batch-index={index} exceeds loader length={len(loader)}") from exc
    return batch


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.max_gradient_examples <= 0:
        raise ValueError("batch-size and max-gradient-examples must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B2-A backward diagnostic currently requires CUDA")

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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    batch_cpu = _take_batch(loader, args.batch_index)
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

    modes = {
        "normalized_mse_only": (1.0, 0.0),
        "cosine_only": (0.0, 1.0),
        "combined": (1.0, 1.0),
    }
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    report: dict[str, object] = {
        "status": "PASS",
        "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "attention_backend": args.attention_backend,
        "student_shape": shape,
        "cast_trainable_parameters": cast_summary,
        "batch_size": args.batch_size,
        "batch_index": args.batch_index,
        "teacher_velocity": tensor_stats(teacher_velocity),
        "cases": {},
    }

    for case, (mse_weight, cosine_weight) in modes.items():
        closure.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(
            "cuda",
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            output = closure(batch)
        objective = velocity_distillation_objective(
            output["velocity"],
            teacher_velocity,
            velocity_mse_weight=mse_weight,
            velocity_cosine_weight=cosine_weight,
            output_l1_weight=0.0,
            output_temporal_weight=0.0,
            gt_loss_mode="none",
            gt_pixel_weight=0.0,
            gt_temporal_weight=0.0,
            epsilon=args.loss_epsilon,
        )
        case_report: dict[str, object] = {
            "student_velocity": tensor_stats(output["velocity"]),
            "z_lq": tensor_stats(output["z_lq"]),
            "loss": float(objective["loss"].detach().float().item()),
            "velocity_mse": float(objective["velocity_mse"].detach().float().item()),
            "velocity_normalized_mse": float(
                objective["velocity_normalized_mse"].detach().float().item()
            ),
            "velocity_cosine": float(objective["velocity_cosine"].detach().float().item()),
            "velocity_cosine_loss": float(
                objective["velocity_cosine_loss"].detach().float().item()
            ),
            "teacher_velocity_power": float(
                objective["teacher_velocity_power"].detach().float().item()
            ),
        }
        if not torch.isfinite(objective["loss"].detach()).item():
            case_report["backward"] = "SKIPPED_NONFINITE_LOSS"
            case_report["gradients"] = None
            report["status"] = "FAIL"
        else:
            objective["loss"].backward()
            gradients = gradient_report(closure, args.max_gradient_examples)
            case_report["backward"] = (
                "PASS" if int(gradients["nonfinite_elements"]) == 0 else "NONFINITE"
            )
            case_report["gradients"] = gradients
            if int(gradients["nonfinite_elements"]) != 0:
                report["status"] = "FAIL"
        case_report["peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
        report["cases"][case] = case_report

        grad = case_report.get("gradients")
        print(
            f"[{case}] loss={case_report['loss']:.8f} "
            f"norm_mse={case_report['velocity_normalized_mse']:.8f} "
            f"cos={case_report['velocity_cosine']:.8f} "
            f"backward={case_report['backward']} "
            + (
                ""
                if not isinstance(grad, dict)
                else f"nonfinite_grad={grad['nonfinite_elements']} "
                     f"finite_grad_l2={grad['finite_global_l2']:.6g}"
            ),
            flush=True,
        )
        if isinstance(grad, dict) and grad["nonfinite_parameter_examples"]:
            print("  non-finite gradient parameters:", flush=True)
            for item in grad["nonfinite_parameter_examples"]:
                print(
                    f"    {item['name']} nonfinite={item['nonfinite']} "
                    f"nan={item['nan']} +inf={item['posinf']} -inf={item['neginf']} "
                    f"finite_max={item['finite_max_abs']}",
                    flush=True,
                )

        del output, objective
        closure.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.output_json is not None:
        _write_json(args.output_json, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
