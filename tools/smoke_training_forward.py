#!/usr/bin/env python3
"""Smoke-test one real triplet batch through the folded SwiftVR student.

This is intentionally a one-batch diagnostic rather than a trainer. It loads a
prompt-free no-time checkpoint, builds one real HR/HQ/LR batch, runs the
training-safe ReAE -> DiT -> ReAE closure, and optionally executes backward.

The conservative default trains only ``prompt_free_adapter`` parameters on a
32x32 LR crop (96x96 HR at scale 3). After that passes, use
``--train-scope transformer`` and/or a larger crop to probe the intended setup.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn as nn

from swiftvr.data import build_triplet_dataloader
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training import SwiftVRTrainingForward


_DTYPE_BY_NAME = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}
_CANONICAL_DTYPE_NAME = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real folded-checkpoint SwiftVR forward/backward smoke test."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Folded prompt-free no-time checkpoint root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Triplet JSONL manifest; repeat to combine plain/text manifests.",
    )
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--clip-length", type=int, default=17)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=32,
        help="Crop in LR/HQ coordinates; HR crop is crop_size * scale.",
    )
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="auto uses transformer/config.json folded_runtime_dtype when present.",
    )
    parser.add_argument(
        "--allow-dtype-mismatch",
        action="store_true",
        help="Allow runtime dtype to differ from the dtype used for time folding.",
    )
    parser.add_argument(
        "--attention-backend",
        default="sdpa",
        help="Training attention backend. Start with sdpa for diagnosis.",
    )
    parser.add_argument(
        "--train-scope",
        choices=("adapter", "transformer", "all"),
        default="adapter",
        help="Train prompt-free adapters only, the full DiT, or DiT + ReAE.",
    )
    parser.add_argument("--latent-loss-weight", type=float, default=0.0)
    parser.add_argument("--forward-only", action="store_true")

    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable result summary.",
    )
    return parser


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required checkpoint file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_folded_checkpoint(
    root: Path,
    *,
    reae_filename: str = "reae.safetensors",
    transformer_subfolder: str = "transformer",
) -> dict[str, object]:
    root = root.expanduser().resolve()
    reae_path = root / reae_filename
    transformer_dir = root / transformer_subfolder
    config_path = transformer_dir / "config.json"

    if not reae_path.is_file():
        raise FileNotFoundError(f"Missing ReAE checkpoint: {reae_path}")
    if not transformer_dir.is_dir():
        raise FileNotFoundError(f"Missing transformer directory: {transformer_dir}")

    config = _read_json(config_path)
    if config.get("time_condition_folded") is not True:
        raise ValueError(
            f"{config_path} is not a folded no-time checkpoint: "
            "time_condition_folded must be true"
        )
    class_name = str(config.get("_class_name", ""))
    if class_name and class_name != "WanTransformer3DModelPromptFreeNoTime":
        raise ValueError(
            f"Unexpected transformer class {class_name!r}; expected "
            "'WanTransformer3DModelPromptFreeNoTime'"
        )

    weight_candidates = (
        transformer_dir / "diffusion_pytorch_model.safetensors",
        transformer_dir / "diffusion_pytorch_model.safetensors.index.json",
        transformer_dir / "diffusion_pytorch_model.bin",
        transformer_dir / "diffusion_pytorch_model.bin.index.json",
    )
    if not any(path.is_file() for path in weight_candidates):
        raise FileNotFoundError(
            f"No Diffusers transformer weights found under {transformer_dir}"
        )
    return config


def resolve_runtime_dtype(
    requested: str,
    config: Mapping[str, object],
    device: torch.device,
    *,
    allow_mismatch: bool = False,
) -> torch.dtype:
    folded_name = str(config.get("folded_runtime_dtype", "")).lower()
    folded_dtype = _DTYPE_BY_NAME.get(folded_name)

    if requested == "auto":
        if folded_dtype is not None:
            runtime_dtype = folded_dtype
        elif device.type == "cuda" and torch.cuda.is_bf16_supported():
            runtime_dtype = torch.bfloat16
        elif device.type == "cuda":
            runtime_dtype = torch.float16
        else:
            runtime_dtype = torch.float32
    else:
        runtime_dtype = _DTYPE_BY_NAME[requested]

    if device.type == "cpu" and runtime_dtype != torch.float32:
        raise ValueError("CPU smoke tests must use --dtype float32")
    if (
        device.type == "cuda"
        and runtime_dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Selected GPU does not support bfloat16")
    if (
        folded_dtype is not None
        and runtime_dtype != folded_dtype
        and not allow_mismatch
    ):
        raise ValueError(
            "Runtime dtype does not match the folded checkpoint: "
            f"requested={_CANONICAL_DTYPE_NAME[runtime_dtype]}, "
            f"folded={_CANONICAL_DTYPE_NAME[folded_dtype]}. "
            "Use --dtype auto or pass --allow-dtype-mismatch intentionally."
        )
    return runtime_dtype


def configure_train_scope(
    reae: nn.Module,
    transformer: nn.Module,
    scope: str,
) -> dict[str, int]:
    if scope not in {"adapter", "transformer", "all"}:
        raise ValueError(f"Unsupported train scope: {scope}")

    for parameter in reae.parameters():
        parameter.requires_grad_(scope == "all")
    for name, parameter in transformer.named_parameters():
        if scope == "adapter":
            parameter.requires_grad_("prompt_free_adapter" in name)
        else:
            parameter.requires_grad_(True)

    reae_total = sum(parameter.numel() for parameter in reae.parameters())
    dit_total = sum(parameter.numel() for parameter in transformer.parameters())
    reae_trainable = sum(
        parameter.numel() for parameter in reae.parameters() if parameter.requires_grad
    )
    dit_trainable = sum(
        parameter.numel()
        for parameter in transformer.parameters()
        if parameter.requires_grad
    )
    if reae_trainable + dit_trainable == 0:
        raise RuntimeError(f"Train scope {scope!r} selected no parameters")
    return {
        "reae_total": reae_total,
        "reae_trainable": reae_trainable,
        "transformer_total": dit_total,
        "transformer_trainable": dit_trainable,
    }


def move_video_batch(
    batch: Mapping[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    moved = dict(batch)
    for key in ("lr", "hq", "hr"):
        value = moved.get(key)
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(
                device=device,
                dtype=dtype,
                non_blocking=device.type == "cuda",
            )
    return moved


def _format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def gradient_summary(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> dict[str, object]:
    squared_norm = 0.0
    maximum = 0.0
    tensors = 0
    elements = 0
    nonfinite = 0
    missing: list[str] = []

    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        grad_float = gradient.detach().float()
        finite = torch.isfinite(grad_float)
        nonfinite += int((~finite).sum().item())
        if finite.any():
            finite_values = grad_float[finite]
            squared_norm += float(torch.sum(finite_values * finite_values).item())
            maximum = max(maximum, float(finite_values.abs().max().item()))
        tensors += 1
        elements += gradient.numel()

    return {
        "global_l2": math.sqrt(squared_norm),
        "max_abs": maximum,
        "gradient_tensors": tensors,
        "gradient_elements": elements,
        "nonfinite_elements": nonfinite,
        "missing_gradient_count": len(missing),
        "missing_gradient_examples": missing[:8],
    }


def _tensor_shape(value: object) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return None


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    config = validate_folded_checkpoint(
        args.checkpoint,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
    )
    dtype = resolve_runtime_dtype(
        args.dtype,
        config,
        device,
        allow_mismatch=args.allow_dtype_mismatch,
    )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset, loader = build_triplet_dataloader(
        args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        seed=args.seed,
        split=args.split,
        training=False,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
    )
    batch_cpu = next(iter(loader))

    checkpoint_root = args.checkpoint.expanduser().resolve()
    reae = ReAE(str(checkpoint_root / args.reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(checkpoint_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    parameter_counts = configure_train_scope(reae, transformer, args.train_scope)

    reae.to(device=device, dtype=dtype)
    transformer.to(device=device, dtype=dtype)
    closure = SwiftVRTrainingForward(
        reae,
        transformer,
        latent_loss_weight=args.latent_loss_weight,
        training_safe_transformer=True,
        prepare_transformer=True,
        attention_backend=args.attention_backend,
    )
    closure.train()
    if args.train_scope != "all":
        reae.eval()

    batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
    closure.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    autocast_enabled = device.type == "cuda" and dtype in {
        torch.float16,
        torch.bfloat16,
    }
    with torch.autocast(
        device_type=device.type,
        dtype=dtype if autocast_enabled else torch.float32,
        enabled=autocast_enabled,
    ):
        output = closure(batch)
        loss = output["loss"]

    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise RuntimeError("Training closure must return a scalar tensor loss")
    if not torch.isfinite(loss.detach()).item():
        raise FloatingPointError(f"Non-finite loss: {float(loss.detach().float().item())}")

    if not args.forward_only:
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    gradients = (
        {
            "global_l2": 0.0,
            "max_abs": 0.0,
            "gradient_tensors": 0,
            "gradient_elements": 0,
            "nonfinite_elements": 0,
            "missing_gradient_count": 0,
            "missing_gradient_examples": [],
        }
        if args.forward_only
        else gradient_summary(closure.named_parameters())
    )
    if not args.forward_only:
        if int(gradients["gradient_tensors"]) == 0:
            raise RuntimeError("Backward produced no trainable gradients")
        if int(gradients["nonfinite_elements"]) != 0:
            raise FloatingPointError(
                f"Backward produced {gradients['nonfinite_elements']} non-finite gradients"
            )

    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    reserved_bytes = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    )

    result: dict[str, object] = {
        "status": "PASS",
        "checkpoint": str(checkpoint_root),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "dtype": _CANONICAL_DTYPE_NAME[dtype],
        "folded_runtime_dtype": config.get("folded_runtime_dtype"),
        "folded_timestep": config.get("folded_timestep"),
        "train_scope": args.train_scope,
        "forward_only": bool(args.forward_only),
        "dataset_length": len(dataset),
        "sample_id": list(batch_cpu.get("sample_id", [])),
        "variant": list(batch_cpu.get("variant", [])),
        "frame_indices": (
            batch_cpu["frame_indices"].tolist()
            if isinstance(batch_cpu.get("frame_indices"), torch.Tensor)
            else None
        ),
        "input_shapes": {
            key: _tensor_shape(batch.get(key)) for key in ("lr", "hq", "hr")
        },
        "output_shapes": {
            key: _tensor_shape(output.get(key))
            for key in ("lq_input", "z_lq", "velocity", "z_prediction", "prediction", "target")
        },
        "loss": float(loss.detach().float().item()),
        "pixel_l1": float(output["pixel_l1"].detach().float().item()),
        "latent_velocity_mse": float(
            output["latent_velocity_mse"].detach().float().item()
        ),
        "elapsed_seconds": elapsed,
        "peak_allocated_gb": peak_bytes / (1024**3),
        "peak_reserved_gb": reserved_bytes / (1024**3),
        "parameters": parameter_counts,
        "gradients": gradients,
    }
    return result


def print_result(result: Mapping[str, object]) -> None:
    print("\n========== SwiftVR training smoke ==========")
    print("status              :", result["status"])
    print("device              :", result["device_name"])
    print("dtype               :", result["dtype"])
    print("folded timestep     :", result["folded_timestep"])
    print("train scope         :", result["train_scope"])
    print("dataset length      :", result["dataset_length"])
    print("sample / variant    :", result["sample_id"], result["variant"])
    print("input shapes        :", result["input_shapes"])
    print("output shapes       :", result["output_shapes"])
    print("loss / pixel L1     :", result["loss"], "/", result["pixel_l1"])
    print("elapsed seconds     :", f"{float(result['elapsed_seconds']):.3f}")
    print("peak allocated GB   :", f"{float(result['peak_allocated_gb']):.3f}")
    print("peak reserved GB    :", f"{float(result['peak_reserved_gb']):.3f}")

    counts = result["parameters"]
    assert isinstance(counts, Mapping)
    print(
        "trainable params    :",
        _format_count(int(counts["reae_trainable"])),
        "ReAE +",
        _format_count(int(counts["transformer_trainable"])),
        "DiT",
    )
    gradients = result["gradients"]
    assert isinstance(gradients, Mapping)
    print("gradient tensors    :", gradients["gradient_tensors"])
    print("gradient global L2  :", f"{float(gradients['global_l2']):.6g}")
    print("gradient max abs    :", f"{float(gradients['max_abs']):.6g}")
    print("nonfinite gradients :", gradients["nonfinite_elements"])
    print("missing grad count  :", gradients["missing_gradient_count"])
    if gradients["missing_gradient_examples"]:
        print("missing examples    :", gradients["missing_gradient_examples"])
    print("============================================\n")


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(
            "CUDA OOM. Retry the conservative path first: "
            "--crop-size 32 --batch-size 1 --train-scope adapter. "
            "Do not lower crop-size below 32 for scale=3 because the HR crop must "
            "remain divisible by the ReAE+DiT spatial multiple."
        )
        return 2

    print_result(result)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote JSON summary to {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
