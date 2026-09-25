#!/usr/bin/env python3
"""Resumable Stage-3 reconstruction baseline for SwiftVR.

This baseline follows SwiftVR's deployment-time one-step path and optimizes
pixel L1 plus consecutive-frame-difference MSE. It adds gradient accumulation
and deterministic full-reference validation while deliberately postponing LPIPS
and adversarial training to the next stage.
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
    VideoMetricAccumulator,
    advance_cursor_batches,
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
    stage3_reconstruction_objective,
    trainable_named_parameters,
    write_latest_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the non-adversarial SwiftVR Stage-3 reconstruction baseline."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--val-manifest", type=Path, action="append", default=None)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--val-crop-size", type=int, default=None)
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
    parser.add_argument("--allow-large-optimizer", action="store_true")
    parser.add_argument("--pixel-loss-weight", type=float, default=1.0)
    parser.add_argument("--temporal-loss-weight", type=float, default=1.0)

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    return parser


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_pointer(
    run_dir: Path,
    filename: str,
    checkpoint_dir: Path,
    **metadata: object,
) -> None:
    try:
        stored = checkpoint_dir.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        stored = str(checkpoint_dir.resolve())
    _write_json(run_dir / filename, {"checkpoint": stored, **metadata})


def _build_train_loader(args: argparse.Namespace, epoch: int):
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


def _build_val_loader(args: argparse.Namespace):
    if not args.val_manifest:
        return None, None
    crop_size = args.crop_size if args.val_crop_size is None else args.val_crop_size
    return build_triplet_dataloader(
        args.val_manifest,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory,
        persistent_workers=False,
        seed=args.seed,
        split=args.val_split,
        training=False,
        clip_length=args.clip_length,
        crop_size=crop_size,
        scale=args.scale,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
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
        "val_manifests": [str(path.expanduser().resolve()) for path in (args.val_manifest or [])],
        "path_root": str(args.path_root.expanduser().resolve()),
        "split": args.split,
        "val_split": args.val_split,
        "clip_length": int(args.clip_length),
        "crop_size": int(args.crop_size),
        "val_crop_size": int(args.crop_size if args.val_crop_size is None else args.val_crop_size),
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
        "pixel_loss_weight": float(args.pixel_loss_weight),
        "temporal_loss_weight": float(args.temporal_loss_weight),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "optimizer_eps": float(args.optimizer_eps),
        "max_grad_norm": float(args.max_grad_norm),
        "validate_every": int(args.validate_every),
        "validation_batches": int(args.validation_batches),
    }


def _validate_resume_config(saved: Mapping[str, object], current: Mapping[str, object]) -> None:
    if dict(saved) == dict(current):
        return
    differences = [
        f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]
    raise ValueError("Resume configuration differs:\n  " + "\n  ".join(differences[:24]))


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
            "last_temporal_mse": last_record.get("temporal_mse"),
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


@torch.no_grad()
def _validate(
    closure: SwiftVRTrainingForward,
    loader,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int,
    pixel_weight: float,
    temporal_weight: float,
    reae_frozen: bool,
) -> dict[str, float | int]:
    if max_batches <= 0:
        raise ValueError("validation-batches must be positive")
    closure.eval()
    metrics = VideoMetricAccumulator()
    sums = {"loss": 0.0, "pixel_l1": 0.0, "temporal_mse": 0.0}
    processed = 0
    autocast_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    try:
        for batch_cpu in loader:
            batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                output = closure(batch)
                objective = stage3_reconstruction_objective(
                    output,
                    pixel_weight=pixel_weight,
                    temporal_weight=temporal_weight,
                )
            for key in sums:
                sums[key] += float(objective[key].detach().float().item())
            prediction = output.get("prediction_clamped")
            target = output.get("target")
            if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise TypeError("Validation output is missing prediction/target tensors")
            metrics.update(prediction, target, clamp=True)
            processed += 1
            if processed >= max_batches:
                break
    finally:
        closure.train()
        if reae_frozen:
            closure.reae.eval()
    if processed == 0:
        raise RuntimeError("Validation DataLoader produced no batches")
    result = metrics.compute()
    result.update({key: value / processed for key, value in sums.items()})
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.num_workers != 0:
        raise ValueError("Exact resume currently requires --num-workers 0")
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient-accumulation-steps must be positive")
    if args.save_every <= 0 or args.log_every <= 0:
        raise ValueError("save-every and log-every must be positive")
    if args.validate_every > 0 and not args.val_manifest:
        raise ValueError("--validate-every > 0 requires at least one --val-manifest")
    if args.train_scope != "adapter" and not args.allow_large_optimizer:
        raise ValueError(
            "Single-device transformer/all AdamW is blocked; use a sharded optimizer plan"
        )

    run_dir = args.output_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train_log.jsonl"
    val_log = run_dir / "val_log.jsonl"
    config_path = run_dir / "run_config.json"

    cursor = TrainingCursor()
    resume_checkpoint = None
    saved_config = None
    pending_rng = None
    if args.resume is not None:
        resume_checkpoint = resolve_resume_checkpoint(args.resume, run_dir=run_dir)
        trainer_state = load_trainer_state(resume_checkpoint)
        cursor = trainer_state["cursor"]
        saved_config = trainer_state["config"]
        pending_rng = trainer_state["rng_state"]
        if not isinstance(cursor, TrainingCursor):
            raise TypeError("Invalid training cursor")
    elif train_log.exists() or (run_dir / "latest.json").exists():
        raise FileExistsError("Output directory already contains a run; use --resume latest")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
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
    dataset, first_loader = _build_train_loader(args, cursor.epoch)
    if len(first_loader) < args.gradient_accumulation_steps:
        raise RuntimeError(
            "Training epoch has fewer batches than gradient-accumulation-steps"
        )
    run_config = _run_fingerprint(
        args,
        base_checkpoint=base_checkpoint,
        dtype=dtype,
        dataset_length=len(dataset),
        batches_per_epoch=len(first_loader),
    )
    if saved_config is not None:
        if not isinstance(saved_config, Mapping):
            raise TypeError("Saved run configuration must be a mapping")
        _validate_resume_config(saved_config, run_config)
    elif config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(existing, Mapping):
            raise ValueError("Invalid run_config.json")
        _validate_resume_config(existing, run_config)
    else:
        _write_json(config_path, run_config)

    _, val_loader = _build_val_loader(args)

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
        latent_loss_weight=0.0,
        training_safe_transformer=True,
        prepare_transformer=True,
        attention_backend=args.attention_backend,
    )
    closure.train()
    reae_frozen = args.train_scope != "all"
    if reae_frozen:
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
            raise ValueError("Delta checkpoint and trainer cursor steps disagree")

    if cursor.global_step >= args.max_steps:
        return {
            "status": "ALREADY_COMPLETE",
            "global_step": cursor.global_step,
            "max_steps": args.max_steps,
            "run_dir": str(run_dir),
        }

    best_psnr = -float("inf")
    best_path = run_dir / "best.json"
    if best_path.is_file():
        best_value = json.loads(best_path.read_text(encoding="utf-8"))
        best_psnr = float(best_value.get("psnr", best_psnr))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    restored_rng = pending_rng is None
    last_record: dict[str, object] = {}
    last_validation: dict[str, object] | None = None
    last_checkpoint = resume_checkpoint

    while cursor.global_step < args.max_steps:
        _, loader = _build_train_loader(args, cursor.epoch)
        batches_per_epoch = len(loader)
        if batches_per_epoch != int(run_config["batches_per_epoch"]):
            raise RuntimeError("DataLoader length changed during the run")
        remaining = batches_per_epoch - cursor.batch_in_epoch
        if remaining < args.gradient_accumulation_steps:
            cursor = advance_cursor_batches(
                cursor,
                consumed_batches=remaining,
                batches_per_epoch=batches_per_epoch,
                optimizer_steps=0,
            )
            continue

        iterator = iter(loader)
        skip_batches(iterator, cursor.batch_in_epoch)
        if not restored_rng:
            if not isinstance(pending_rng, Mapping):
                raise TypeError("Saved RNG state must be a mapping")
            restore_rng_state(pending_rng)
            restored_rng = True
            pending_rng = None

        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        sums = {"loss": 0.0, "pixel_l1": 0.0, "temporal_mse": 0.0}
        sample_ids: list[object] = []
        variants: list[object] = []
        autocast_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}

        for _ in range(args.gradient_accumulation_steps):
            batch_cpu = next(iterator)
            batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                output = closure(batch)
                objective = stage3_reconstruction_objective(
                    output,
                    pixel_weight=args.pixel_loss_weight,
                    temporal_weight=args.temporal_loss_weight,
                )
                scaled_loss = objective["loss"] / args.gradient_accumulation_steps
            if not torch.isfinite(objective["loss"].detach()).item():
                raise FloatingPointError(f"Non-finite loss before step {cursor.global_step + 1}")
            scaler.scale(scaled_loss).backward()
            for key in sums:
                sums[key] += float(objective[key].detach().float().item())
            sample_ids.extend(list(batch_cpu.get("sample_id", [])))
            variants.extend(list(batch_cpu.get("variant", [])))

        scaler.unscale_(optimizer)
        gradients = gradient_summary(closure.named_parameters())
        if int(gradients["gradient_tensors"]) == 0:
            raise RuntimeError("Backward produced no trainable gradients")
        if int(gradients["nonfinite_elements"]) != 0:
            raise FloatingPointError("Backward produced non-finite gradients")
        if int(gradients["missing_gradient_count"]) != 0:
            raise RuntimeError(
                f"Missing gradients: {gradients['missing_gradient_examples']}"
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
            raise FloatingPointError("GradScaler skipped a step because of overflow")

        cursor = advance_cursor_batches(
            cursor,
            consumed_batches=args.gradient_accumulation_steps,
            batches_per_epoch=batches_per_epoch,
            optimizer_steps=1,
        )
        averages = {
            key: value / args.gradient_accumulation_steps for key, value in sums.items()
        }
        step_seconds = time.perf_counter() - step_started
        last_record = {
            "global_step": cursor.global_step,
            "epoch": cursor.epoch,
            "batch_in_epoch": cursor.batch_in_epoch,
            "microbatches": int(args.gradient_accumulation_steps),
            "sample_id": sample_ids,
            "variant": variants,
            **averages,
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
        append_jsonl(train_log, last_record)
        if cursor.global_step % args.log_every == 0:
            print(
                f"step={cursor.global_step} loss={averages['loss']:.8f} "
                f"pixel={averages['pixel_l1']:.8f} temp={averages['temporal_mse']:.8f} "
                f"grad={grad_norm:.6g} time={step_seconds:.3f}s",
                flush=True,
            )

        validation_due = (
            val_loader is not None
            and args.validate_every > 0
            and (cursor.global_step % args.validate_every == 0 or cursor.global_step == args.max_steps)
        )
        improved = False
        if validation_due:
            training_rng = capture_rng_state()
            last_validation = _validate(
                closure,
                val_loader,
                device=device,
                dtype=dtype,
                max_batches=args.validation_batches,
                pixel_weight=args.pixel_loss_weight,
                temporal_weight=args.temporal_loss_weight,
                reae_frozen=reae_frozen,
            )
            restore_rng_state(training_rng)
            last_validation = {"global_step": cursor.global_step, **last_validation}
            append_jsonl(val_log, last_validation)
            improved = float(last_validation["psnr"]) > best_psnr
            if improved:
                best_psnr = float(last_validation["psnr"])
            print(
                f"validation step={cursor.global_step} psnr={last_validation['psnr']:.4f} "
                f"ssim={last_validation['ssim']:.6f} mae={last_validation['mae']:.6f}",
                flush=True,
            )

        should_save = (
            cursor.global_step % args.save_every == 0
            or cursor.global_step == args.max_steps
            or validation_due
        )
        if should_save:
            state_summary = optimizer_state_summary(optimizer)
            if int(state_summary["nonfinite_elements"]) != 0:
                raise FloatingPointError("Optimizer state contains non-finite values")
            if set(state_summary["dtype_counts"]) != {"float32"}:
                raise RuntimeError(f"Expected FP32 AdamW state, got {state_summary['dtype_counts']}")
            last_checkpoint = _checkpoint(
                run_dir=run_dir,
                closure=closure,
                optimizer=optimizer,
                scaler=scaler,
                cursor=cursor,
                run_config=run_config,
                last_record=last_record,
            )
            if improved and last_validation is not None:
                _write_pointer(
                    run_dir,
                    "best.json",
                    last_checkpoint,
                    global_step=cursor.global_step,
                    psnr=float(last_validation["psnr"]),
                    ssim=float(last_validation["ssim"]),
                )
            print(f"saved checkpoint: {last_checkpoint}", flush=True)

    result = {
        "status": "PASS",
        "global_step": cursor.global_step,
        "epoch": cursor.epoch,
        "batch_in_epoch": cursor.batch_in_epoch,
        "max_steps": args.max_steps,
        "run_dir": str(run_dir),
        "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "runtime_dtype": _CANONICAL_DTYPE_NAME[dtype],
        "optimizer_parameter_dtype": optimizer_precision["target_dtype"],
        "train_scope": args.train_scope,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "parameters": parameter_counts,
        "last_record": last_record,
        "last_validation": last_validation,
        "best_psnr": None if best_psnr == -float("inf") else best_psnr,
    }
    _write_json(run_dir / "summary.json", result)
    return result


def print_result(result: Mapping[str, object]) -> None:
    print("\n========== SwiftVR Stage-3 reconstruction ==========")
    print("status              :", result["status"])
    print("global step         :", result["global_step"], "/", result["max_steps"])
    if result["status"] == "PASS":
        print("cursor              :", result["epoch"], result["batch_in_epoch"])
        print("device              :", result["device_name"])
        print("runtime dtype       :", result["runtime_dtype"])
        print("optimizer dtype     :", result["optimizer_parameter_dtype"])
        print("train scope         :", result["train_scope"])
        print("gradient accumulation:", result["gradient_accumulation_steps"])
        counts = result["parameters"]
        assert isinstance(counts, Mapping)
        print(
            "trainable params    :",
            _format_count(int(counts["reae_trainable"])),
            "ReAE +",
            _format_count(int(counts["transformer_trainable"])),
            "DiT",
        )
        print("best PSNR           :", result["best_psnr"])
        print("last checkpoint     :", result["last_checkpoint"])
        print("elapsed seconds     :", f"{float(result['elapsed_seconds']):.3f}")
    print("====================================================\n")


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(
            "CUDA OOM. Reduce --crop-size/--gradient-accumulation-steps only affects "
            "effective batch, not single-microbatch activation memory. Start with "
            "--crop-size 64 to establish the Stage-3 gate on V100."
        )
        return 2
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
