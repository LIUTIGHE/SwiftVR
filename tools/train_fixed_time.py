#!/usr/bin/env python3
"""Minimal resumable trainer for SwiftVR's fixed-time prompt-free student.

This first training stage is intentionally single-process and adapter-focused. It
uses real triplet clips, FP16 autocast on V100, FP32 trainable adapter weights and
AdamW state, periodic delta checkpoints, and exact mid-epoch resume for
``num_workers=0``.
"""

from __future__ import annotations

import argparse
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
    TrainingCursor,
    append_jsonl,
    build_fp32_adamw,
    build_grad_scaler,
    capture_rng_state,
    cast_trainable_parameters,
    load_delta_checkpoint,
    load_trainer_state,
    optimizer_state_summary,
    resolve_resume_checkpoint,
    restore_rng_state,
    save_delta_checkpoint,
    save_trainer_state,
    seed_everything,
    skip_batches,
    trainable_named_parameters,
    write_latest_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SwiftVR prompt-free adapters with exact single-worker resume."
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
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.0)

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
        help="Acknowledge memory risk for transformer/all AdamW state.",
    )
    parser.add_argument("--latent-loss-weight", type=float, default=0.0)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help="Target global optimizer step; on resume this is the final total step.",
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint directory or 'latest' under --output-dir.",
    )
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    return parser


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_epoch_loader(args: argparse.Namespace, epoch: int):
    return build_triplet_dataloader(
        args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=args.pin_memory,
        persistent_workers=False,
        seed=args.seed + int(epoch),
        split=args.split,
        training=True,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
        horizontal_flip_probability=args.horizontal_flip_probability,
        vertical_flip_probability=args.vertical_flip_probability,
    )


def _run_fingerprint(
    args: argparse.Namespace,
    *,
    base_checkpoint: Path,
    dtype: torch.dtype,
    dataset_length: int,
    batches_per_epoch: int,
) -> dict[str, object]:
    return {
        "base_checkpoint": str(base_checkpoint),
        "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
        "path_root": str(args.path_root.expanduser().resolve()),
        "split": args.split,
        "clip_length": int(args.clip_length),
        "crop_size": int(args.crop_size),
        "scale": int(args.scale),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "horizontal_flip_probability": float(args.horizontal_flip_probability),
        "vertical_flip_probability": float(args.vertical_flip_probability),
        "dataset_length": int(dataset_length),
        "batches_per_epoch": int(batches_per_epoch),
        "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
        "attention_backend": args.attention_backend,
        "train_scope": args.train_scope,
        "latent_loss_weight": float(args.latent_loss_weight),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "optimizer_eps": float(args.optimizer_eps),
        "max_grad_norm": float(args.max_grad_norm),
    }


def _validate_resume_config(
    saved: Mapping[str, object], current: Mapping[str, object]
) -> None:
    if dict(saved) == dict(current):
        return
    keys = sorted(set(saved) | set(current))
    differences = [
        f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
        for key in keys
        if saved.get(key) != current.get(key)
    ]
    raise ValueError(
        "Resume configuration differs from the checkpoint:\n  "
        + "\n  ".join(differences[:20])
    )


def _checkpoint(
    *,
    run_dir: Path,
    closure: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    cursor: TrainingCursor,
    run_config: Mapping[str, object],
    last_record: Mapping[str, object],
) -> Path:
    checkpoint_dir = run_dir / "checkpoints" / f"step_{cursor.global_step:08d}"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_dir}")
    save_delta_checkpoint(
        checkpoint_dir,
        closure,
        optimizer,
        step=cursor.global_step,
        grad_scaler=scaler,
        metadata={
            "cursor": {
                "global_step": cursor.global_step,
                "epoch": cursor.epoch,
                "batch_in_epoch": cursor.batch_in_epoch,
            },
            "last_loss": last_record.get("loss"),
            "last_pixel_l1": last_record.get("pixel_l1"),
            "runtime_dtype": run_config["runtime_dtype"],
            "train_scope": run_config["train_scope"],
        },
    )
    save_trainer_state(
        checkpoint_dir,
        cursor=cursor,
        config=run_config,
        rng_state=capture_rng_state(),
    )
    write_latest_checkpoint(run_dir, checkpoint_dir)
    return checkpoint_dir


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.num_workers != 0:
        raise ValueError(
            "Exact mid-epoch resume currently requires --num-workers 0. "
            "Multi-worker worker/prefetch state will be handled with the distributed trainer."
        )
    if args.max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {args.max_steps}")
    if args.save_every <= 0 or args.log_every <= 0:
        raise ValueError("save_every and log_every must be positive")
    if args.train_scope != "adapter" and not args.allow_large_optimizer:
        raise ValueError(
            "Single-device transformer/all AdamW is blocked by default. "
            "Pass --allow-large-optimizer only with a suitable sharding/offload plan."
        )

    run_dir = args.output_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    config_path = run_dir / "run_config.json"

    resume_checkpoint = None
    pending_rng_state = None
    cursor = TrainingCursor()
    saved_run_config = None
    if args.resume is not None:
        resume_checkpoint = resolve_resume_checkpoint(args.resume, run_dir=run_dir)
        trainer_state = load_trainer_state(resume_checkpoint)
        loaded_cursor = trainer_state["cursor"]
        if not isinstance(loaded_cursor, TrainingCursor):
            raise TypeError("Loaded trainer cursor has an unexpected type")
        cursor = loaded_cursor
        saved_run_config = trainer_state["config"]
        pending_rng_state = trainer_state["rng_state"]
    elif log_path.exists() or (run_dir / "latest.json").exists():
        raise FileExistsError(
            f"Output directory already contains a run: {run_dir}. "
            "Use --resume latest or choose a new directory."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    base_checkpoint = args.checkpoint.expanduser().resolve()
    folded_config = validate_folded_checkpoint(
        base_checkpoint,
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
    dataset, first_loader = _build_epoch_loader(args, cursor.epoch)
    if len(first_loader) <= 0:
        raise RuntimeError("Training DataLoader contains no batches")
    run_config = _run_fingerprint(
        args,
        base_checkpoint=base_checkpoint,
        dtype=dtype,
        dataset_length=len(dataset),
        batches_per_epoch=len(first_loader),
    )
    if saved_run_config is not None:
        if not isinstance(saved_run_config, Mapping):
            raise TypeError("Saved run configuration must be a mapping")
        _validate_resume_config(saved_run_config, run_config)
    elif config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(existing, Mapping):
            raise ValueError(f"Invalid run config: {config_path}")
        _validate_resume_config(existing, run_config)
    else:
        _write_json(config_path, run_config)

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
    optimizer = build_fp32_adamw(
        closure,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.optimizer_eps,
    )
    scaler = build_grad_scaler(device, dtype)

    if resume_checkpoint is not None:
        metadata = load_delta_checkpoint(
            resume_checkpoint,
            closure,
            optimizer,
            strict=True,
            map_location="cpu",
            grad_scaler=scaler,
        )
        if int(metadata["step"]) != cursor.global_step:
            raise ValueError(
                "Delta checkpoint step and trainer cursor disagree: "
                f"delta={metadata['step']}, cursor={cursor.global_step}"
            )

    if cursor.global_step >= args.max_steps:
        return {
            "status": "ALREADY_COMPLETE",
            "global_step": cursor.global_step,
            "max_steps": args.max_steps,
            "run_dir": str(run_dir),
        }

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    last_record: dict[str, object] = {}
    last_checkpoint = resume_checkpoint
    restored_rng = pending_rng_state is None

    while cursor.global_step < args.max_steps:
        _, loader = _build_epoch_loader(args, cursor.epoch)
        if len(loader) != int(run_config["batches_per_epoch"]):
            raise RuntimeError("DataLoader length changed during the run")
        iterator = iter(loader)
        skip_batches(iterator, cursor.batch_in_epoch)
        if not restored_rng:
            if not isinstance(pending_rng_state, Mapping):
                raise TypeError("Saved RNG state must be a mapping")
            restore_rng_state(pending_rng_state)
            restored_rng = True
            pending_rng_state = None

        while cursor.global_step < args.max_steps:
            try:
                batch_cpu = next(iterator)
            except StopIteration:
                if cursor.batch_in_epoch != 0:
                    raise RuntimeError(
                        "DataLoader ended before the cursor reached the next epoch"
                    )
                break

            batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_started = time.perf_counter()

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
                raise FloatingPointError(
                    f"Non-finite loss at step {cursor.global_step + 1}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = gradient_summary(closure.named_parameters())
            if int(gradients["gradient_tensors"]) == 0:
                raise RuntimeError("Backward produced no trainable gradients")
            if int(gradients["nonfinite_elements"]) != 0:
                raise FloatingPointError(
                    f"Backward produced non-finite gradients at step {cursor.global_step + 1}"
                )
            if int(gradients["missing_gradient_count"]) != 0:
                raise RuntimeError(
                    "Backward missed trainable gradients: "
                    f"{gradients['missing_gradient_examples']}"
                )

            grad_norm = float(gradients["global_l2"])
            if args.max_grad_norm > 0:
                clipped = torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in trainable_named_parameters(closure)],
                    max_norm=float(args.max_grad_norm),
                    error_if_nonfinite=True,
                )
                grad_norm = float(clipped.detach().float().item())

            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            if scale_after < scale_before:
                raise FloatingPointError(
                    "GradScaler skipped an optimizer step because of overflow; "
                    "the minimal trainer stops to preserve exact step semantics"
                )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cursor = cursor.advance(batches_per_epoch=len(loader))
            step_seconds = time.perf_counter() - step_started
            last_record = {
                "global_step": cursor.global_step,
                "epoch": cursor.epoch,
                "batch_in_epoch": cursor.batch_in_epoch,
                "sample_id": list(batch_cpu.get("sample_id", [])),
                "variant": list(batch_cpu.get("variant", [])),
                "loss": float(loss.detach().float().item()),
                "pixel_l1": float(output["pixel_l1"].detach().float().item()),
                "latent_velocity_mse": float(
                    output["latent_velocity_mse"].detach().float().item()
                ),
                "gradient_norm": grad_norm,
                "grad_scaler_scale": scale_after,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "step_seconds": step_seconds,
                "peak_allocated_gb": (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            append_jsonl(log_path, last_record)

            if cursor.global_step % args.log_every == 0:
                print(
                    f"step={cursor.global_step} epoch={cursor.epoch} "
                    f"batch={cursor.batch_in_epoch} "
                    f"loss={last_record['loss']:.8f} "
                    f"grad={grad_norm:.6g} scale={scale_after:.1f} "
                    f"time={step_seconds:.3f}s",
                    flush=True,
                )

            should_save = (
                cursor.global_step % args.save_every == 0
                or cursor.global_step == args.max_steps
            )
            if should_save:
                state_summary = optimizer_state_summary(optimizer)
                if int(state_summary["nonfinite_elements"]) != 0:
                    raise FloatingPointError("Optimizer state contains non-finite values")
                if set(state_summary["dtype_counts"]) != {"float32"}:
                    raise RuntimeError(
                        "Expected FP32 AdamW state, got "
                        f"{state_summary['dtype_counts']}"
                    )
                last_checkpoint = _checkpoint(
                    run_dir=run_dir,
                    closure=closure,
                    optimizer=optimizer,
                    scaler=scaler,
                    cursor=cursor,
                    run_config=run_config,
                    last_record=last_record,
                )
                print(f"saved checkpoint: {last_checkpoint}", flush=True)

            if cursor.batch_in_epoch == 0:
                break

    elapsed = time.perf_counter() - started
    result = {
        "status": "PASS",
        "global_step": cursor.global_step,
        "epoch": cursor.epoch,
        "batch_in_epoch": cursor.batch_in_epoch,
        "max_steps": args.max_steps,
        "run_dir": str(run_dir),
        "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
        "optimizer_parameter_dtype": optimizer_precision["target_dtype"],
        "train_scope": args.train_scope,
        "parameters": parameter_counts,
        "last_record": last_record,
    }
    _write_json(run_dir / "summary.json", result)
    return result


def print_result(result: Mapping[str, object]) -> None:
    print("\n========== SwiftVR minimal trainer ==========")
    print("status              :", result["status"])
    print("global step         :", result["global_step"], "/", result["max_steps"])
    if result["status"] == "PASS":
        print("cursor              :", result["epoch"], result["batch_in_epoch"])
        print("device              :", result["device_name"])
        print("runtime dtype       :", result["runtime_dtype"])
        print("optimizer dtype     :", result["optimizer_parameter_dtype"])
        print("train scope         :", result["train_scope"])
        counts = result["parameters"]
        assert isinstance(counts, Mapping)
        print(
            "trainable params    :",
            _format_count(int(counts["reae_trainable"])),
            "ReAE +",
            _format_count(int(counts["transformer_trainable"])),
            "DiT",
        )
        print("last checkpoint     :", result["last_checkpoint"])
        print("elapsed seconds     :", f"{float(result['elapsed_seconds']):.3f}")
    print("=============================================\n")


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(
            "CUDA OOM. Keep --train-scope adapter --batch-size 1 --crop-size 32 "
            "for this single-V100 validation stage."
        )
        return 2
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
