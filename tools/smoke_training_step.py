#!/usr/bin/env python3
"""Run one real SwiftVR optimizer step and verify delta-checkpoint resume.

Frozen ReAE/DiT weights keep the folded checkpoint runtime dtype. Only trainable
parameters are promoted to FP32 before AdamW construction, so optimizer moments
and epsilon arithmetic are numerically stable on FP16-only GPUs such as V100.
FP16 forward/backward additionally uses AMP GradScaler.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Mapping

import torch

from smoke_training_forward import (
    _CANONICAL_DTYPE_NAME,
    _format_count,
    configure_train_scope,
    gradient_summary,
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.data import build_triplet_dataloader
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training import (
    SwiftVRTrainingForward,
    capture_trainable_parameters,
    cast_trainable_parameters,
    load_delta_checkpoint,
    parameter_update_summary,
    save_delta_checkpoint,
    trainable_named_parameters,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real SwiftVR AdamW step and verify checkpoint resume."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Triplet JSONL manifest; repeat to combine manifests.",
    )
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--clip-length", type=int, default=17)
    parser.add_argument("--crop-size", type=int, default=32)
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
    )
    parser.add_argument("--allow-dtype-mismatch", action="store_true")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument(
        "--train-scope",
        choices=("adapter", "transformer", "all"),
        default="adapter",
    )
    parser.add_argument(
        "--allow-large-optimizer",
        action="store_true",
        help="Acknowledge the memory risk of AdamW for transformer/all scope.",
    )
    parser.add_argument("--latent-loss-weight", type=float, default=0.0)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--optimizer-eps",
        type=float,
        default=1e-8,
        help="AdamW epsilon. Safe because trainable parameters/states are FP32.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Set <=0 to disable gradient clipping.",
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        required=True,
        help="Directory for trainable.safetensors, optimizer.pt, and metadata.json.",
    )
    parser.add_argument(
        "--skip-resume-verify",
        action="store_true",
        help="Save the delta checkpoint without perturb-and-restore verification.",
    )
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def _clone_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _assert_nested_equal(expected, actual, path: str = "root") -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{path}: expected tensor, got {type(actual).__name__}")
        torch.testing.assert_close(
            actual.detach().to(device="cpu", dtype=expected.dtype),
            expected,
            rtol=0,
            atol=0,
            msg=lambda message: f"{path}: {message}",
        )
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AssertionError(f"{path}: dictionary keys differ")
        for key in expected:
            _assert_nested_equal(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise AssertionError(f"{path}: sequence differs")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_nested_equal(left, right, f"{path}[{index}]")
        return
    if expected != actual:
        raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def _parameter_restore_exact(
    expected: Mapping[str, torch.Tensor],
    module: torch.nn.Module,
) -> None:
    actual = capture_trainable_parameters(module)
    if tuple(expected) != tuple(actual):
        raise AssertionError("Restored trainable parameter names differ")
    for name in expected:
        torch.testing.assert_close(
            actual[name],
            expected[name],
            rtol=0,
            atol=0,
            msg=lambda message, parameter_name=name: (
                f"Restored parameter {parameter_name} differs: {message}"
            ),
        )


def _build_optimizer(
    module: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    eps: float,
) -> torch.optim.AdamW:
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if eps <= 0:
        raise ValueError(f"optimizer eps must be positive, got {eps}")
    parameters = [parameter for _, parameter in trainable_named_parameters(module)]
    non_fp32 = [str(parameter.dtype) for parameter in parameters if parameter.dtype != torch.float32]
    if non_fp32:
        raise RuntimeError(
            "AdamW trainable parameters must be FP32; call "
            f"cast_trainable_parameters first. Found: {sorted(set(non_fp32))}"
        )
    return torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        eps=float(eps),
        foreach=False,
    )


def _build_grad_scaler(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda" and dtype == torch.float16
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _optimizer_state_summary(optimizer: torch.optim.Optimizer) -> dict[str, object]:
    dtype_counts: dict[str, int] = {}
    tensor_count = 0
    nonfinite = 0
    for state in optimizer.state.values():
        for value in state.values():
            if not isinstance(value, torch.Tensor):
                continue
            tensor_count += 1
            dtype_name = str(value.dtype).removeprefix("torch.")
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
            if value.is_floating_point():
                nonfinite += int((~torch.isfinite(value.detach())).sum().item())
    return {
        "tensor_count": tensor_count,
        "dtype_counts": dtype_counts,
        "nonfinite_elements": nonfinite,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.train_scope != "adapter" and not args.allow_large_optimizer:
        raise ValueError(
            "AdamW state for transformer/all scope is intentionally blocked on a "
            "single-device smoke run. Pass --allow-large-optimizer only when the "
            "available memory and distributed strategy are understood."
        )

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

    base_checkpoint = args.checkpoint.expanduser().resolve()
    reae = ReAE(str(base_checkpoint / args.reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(base_checkpoint),
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

    optimizer_precision = cast_trainable_parameters(closure, dtype=torch.float32)
    optimizer = _build_optimizer(
        closure,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.optimizer_eps,
    )
    scaler = _build_grad_scaler(device, dtype)

    before = capture_trainable_parameters(closure)
    batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
    optimizer.zero_grad(set_to_none=True)

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
        raise RuntimeError("Training closure must return a scalar loss")
    if not torch.isfinite(loss.detach()).item():
        raise FloatingPointError(f"Non-finite loss: {float(loss.detach().float())}")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    gradients = gradient_summary(closure.named_parameters())
    if int(gradients["gradient_tensors"]) == 0:
        raise RuntimeError("Backward produced no trainable gradients")
    if int(gradients["nonfinite_elements"]) != 0:
        raise FloatingPointError(
            f"Backward produced {gradients['nonfinite_elements']} non-finite gradients"
        )
    if int(gradients["missing_gradient_count"]) != 0:
        raise RuntimeError(
            "Backward missed trainable gradients: "
            f"{gradients['missing_gradient_examples']}"
        )

    clipped_norm = None
    if args.max_grad_norm > 0:
        clipped = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable_named_parameters(closure)],
            max_norm=float(args.max_grad_norm),
            error_if_nonfinite=True,
        )
        clipped_norm = float(clipped.detach().float().item())

    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    optimizer.zero_grad(set_to_none=True)

    updates = parameter_update_summary(before, closure)
    if int(updates["changed_tensors"]) == 0:
        raise RuntimeError(
            "Optimizer step completed but no trainable parameter changed. "
            "Inspect GradScaler overflow status or increase --learning-rate for this diagnostic."
        )
    if int(updates["nonfinite_elements"]) != 0:
        raise FloatingPointError("Optimizer step produced non-finite parameter deltas")

    optimizer_state = _optimizer_state_summary(optimizer)
    if int(optimizer_state["nonfinite_elements"]) != 0:
        raise FloatingPointError("AdamW state contains non-finite values")
    state_dtypes = set(optimizer_state["dtype_counts"])
    if state_dtypes and state_dtypes != {"float32"}:
        raise RuntimeError(
            "AdamW state was expected to be FP32, got "
            f"{optimizer_state['dtype_counts']}"
        )

    checkpoint_dir = args.checkpoint_output.expanduser().resolve()
    checkpoint_metadata = save_delta_checkpoint(
        checkpoint_dir,
        closure,
        optimizer,
        step=1,
        grad_scaler=scaler,
        metadata={
            "base_checkpoint": str(base_checkpoint),
            "train_scope": args.train_scope,
            "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
            "optimizer_parameter_dtype": "float32",
            "folded_timestep": config.get("folded_timestep"),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "optimizer_eps": float(args.optimizer_eps),
            "max_grad_norm": float(args.max_grad_norm),
            "amp_grad_scaler_enabled": bool(scaler.is_enabled()),
            "sample_id": list(batch_cpu.get("sample_id", [])),
            "variant": list(batch_cpu.get("variant", [])),
        },
    )

    resume_verified = False
    if not args.skip_resume_verify:
        expected_parameters = capture_trainable_parameters(closure)
        expected_optimizer = _clone_to_cpu(optimizer.state_dict())
        expected_scaler = _clone_to_cpu(scaler.state_dict())

        with torch.no_grad():
            for _, parameter in trainable_named_parameters(closure):
                parameter.add_(torch.ones_like(parameter))

        resumed_optimizer = _build_optimizer(
            closure,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.optimizer_eps,
        )
        resumed_scaler = _build_grad_scaler(device, dtype)
        loaded_metadata = load_delta_checkpoint(
            checkpoint_dir,
            closure,
            resumed_optimizer,
            strict=True,
            map_location="cpu",
            grad_scaler=resumed_scaler,
        )
        if int(loaded_metadata["step"]) != 1:
            raise AssertionError("Restored global step does not equal 1")
        _parameter_restore_exact(expected_parameters, closure)
        _assert_nested_equal(
            expected_optimizer,
            _clone_to_cpu(resumed_optimizer.state_dict()),
            path="optimizer",
        )
        _assert_nested_equal(
            expected_scaler,
            _clone_to_cpu(resumed_scaler.state_dict()),
            path="grad_scaler",
        )
        resume_verified = True

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    reserved_bytes = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    )

    return {
        "status": "PASS",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "dtype": _CANONICAL_DTYPE_NAME[dtype],
        "folded_timestep": config.get("folded_timestep"),
        "train_scope": args.train_scope,
        "dataset_length": len(dataset),
        "sample_id": list(batch_cpu.get("sample_id", [])),
        "variant": list(batch_cpu.get("variant", [])),
        "loss": float(loss.detach().float().item()),
        "pixel_l1": float(output["pixel_l1"].detach().float().item()),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "optimizer_eps": float(args.optimizer_eps),
        "max_grad_norm": float(args.max_grad_norm),
        "pre_clip_gradient_norm": clipped_norm,
        "grad_scaler_enabled": bool(scaler.is_enabled()),
        "grad_scale_before": scale_before,
        "grad_scale_after": scale_after,
        "optimizer_precision": optimizer_precision,
        "optimizer_state": optimizer_state,
        "elapsed_seconds": elapsed,
        "peak_allocated_gb": peak_bytes / (1024**3),
        "peak_reserved_gb": reserved_bytes / (1024**3),
        "parameters": parameter_counts,
        "gradients": gradients,
        "updates": updates,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint": checkpoint_metadata,
        "resume_verified": resume_verified,
    }


def print_result(result: Mapping[str, object]) -> None:
    print("\n========== SwiftVR optimizer smoke ==========")
    print("status              :", result["status"])
    print("device              :", result["device_name"])
    print("runtime dtype       :", result["dtype"])
    print("optimizer dtype     :", result["optimizer_precision"]["target_dtype"])
    print("folded timestep     :", result["folded_timestep"])
    print("train scope         :", result["train_scope"])
    print("dataset length      :", result["dataset_length"])
    print("sample / variant    :", result["sample_id"], result["variant"])
    print("loss / pixel L1     :", result["loss"], "/", result["pixel_l1"])
    print("learning rate       :", result["learning_rate"])
    print("optimizer eps       :", result["optimizer_eps"])
    print("pre-clip grad norm  :", result["pre_clip_gradient_norm"])
    print("GradScaler enabled  :", result["grad_scaler_enabled"])
    print(
        "GradScaler scale    :",
        result["grad_scale_before"],
        "->",
        result["grad_scale_after"],
    )
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
    updates = result["updates"]
    optimizer_state = result["optimizer_state"]
    assert isinstance(gradients, Mapping)
    assert isinstance(updates, Mapping)
    assert isinstance(optimizer_state, Mapping)
    print("gradient tensors    :", gradients["gradient_tensors"])
    print("gradient global L2  :", f"{float(gradients['global_l2']):.6g}")
    print("changed tensors     :", updates["changed_tensors"])
    print("changed elements    :", updates["changed_elements"])
    print("update global L2    :", f"{float(updates['global_l2']):.6g}")
    print("update max abs      :", f"{float(updates['max_abs']):.6g}")
    print("optimizer state     :", optimizer_state["dtype_counts"])
    print("checkpoint          :", result["checkpoint_dir"])
    print("resume verified     :", result["resume_verified"])
    print("=============================================\n")


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(
            "CUDA OOM during optimizer allocation/step. Use --train-scope adapter "
            "on a single 32GB V100. Full-transformer AdamW requires optimizer "
            "sharding/offload or multiple GPUs."
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
