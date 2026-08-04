#!/usr/bin/env python3
"""Evaluate a prompt-free Stage-3 checkpoint against GT and a cached conditional reference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smoke_training_forward import configure_train_scope, resolve_runtime_dtype, validate_folded_checkpoint
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training import (
    SwiftVRTrainingForward,
    VideoMetricAccumulator,
    cast_trainable_parameters,
    load_delta_checkpoint,
    temporal_difference_mse,
)
from swiftvr.training.reference import (
    ConditionalReferenceCache,
    VelocityMetricAccumulator,
    batch_sample_identity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-base-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, default=None)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    return parser


def _move_batch(batch: dict[str, object], device: torch.device, dtype: torch.dtype):
    result = dict(batch)
    for key in ("lr", "hq", "hr"):
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, dtype=dtype, non_blocking=True)
    return result


def _finalize_video_metrics(
    accumulator: VideoMetricAccumulator,
    pixel_sum: float,
    temporal_sum: float,
    samples: int,
) -> dict[str, float | int]:
    result = accumulator.compute()
    result["pixel_l1"] = pixel_sum / samples
    result["temporal_mse"] = temporal_sum / samples
    return result


def _write_tensorboard(log_dir: Path, step: int, result: dict[str, object]) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        raise RuntimeError("Install tensorboard before using --tensorboard-dir") from exc
    writer = SummaryWriter(log_dir=str(log_dir.expanduser().resolve()))
    try:
        for group in ("student_gt", "reference_gt", "student_reference", "gap"):
            payload = result.get(group)
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    writer.add_scalar(f"val/{group}/{key}", float(value), step)
        writer.add_scalar("val/eval_seconds", float(result["eval_seconds"]), step)
        writer.flush()
    finally:
        writer.close()


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    base = args.student_base_checkpoint.expanduser().resolve()
    folded_config = validate_folded_checkpoint(
        base,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
    )
    dtype = resolve_runtime_dtype(args.dtype, folded_config, device, allow_mismatch=False)
    cache = ConditionalReferenceCache(args.reference_cache)
    expected_samples = int(cache.metadata["sample_count"])
    expected_config = {
        "val_split": args.val_split,
        "clip_length": int(args.clip_length),
        "crop_size": int(args.crop_size),
        "scale": int(args.scale),
    }
    differences = [
        f"{key}: cache={cache.metadata.get(key)!r}, current={value!r}"
        for key, value in expected_config.items()
        if cache.metadata.get(key) != value
    ]
    current_manifests = [path.expanduser().resolve() for path in args.val_manifest]
    saved_hashes = cache.metadata.get("val_manifest_sha256")
    if not isinstance(saved_hashes, dict):
        differences.append("cache does not contain val_manifest_sha256")
    else:
        from swiftvr.training.reference import sha256_file

        for path in current_manifests:
            current_hash = sha256_file(path)
            if saved_hashes.get(str(path)) != current_hash:
                differences.append(f"manifest hash mismatch: {path}")
    if differences:
        raise ValueError("Reference cache configuration differs:\n  " + "\n  ".join(differences))

    dataset = TripletVideoDataset(
        args.val_manifest,
        split=args.val_split,
        training=False,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        drop_short_sequences=True,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    reae = ReAE(str(base / args.reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(base),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    configure_train_scope(reae, transformer, "adapter")
    reae.to(device=device, dtype=dtype).eval()
    transformer.to(device=device, dtype=dtype)
    closure = SwiftVRTrainingForward(
        reae,
        transformer,
        latent_loss_weight=0.0,
        training_safe_transformer=True,
        prepare_transformer=True,
        attention_backend=args.attention_backend,
    )
    cast_trainable_parameters(closure, dtype=torch.float32)
    if args.student_checkpoint is None:
        metadata: dict[str, object] = {"step": 0}
    else:
        metadata = load_delta_checkpoint(
            args.student_checkpoint,
            closure,
            optimizer=None,
            strict=True,
            map_location="cpu",
        )
    closure.eval()

    student_gt_acc = VideoMetricAccumulator()
    reference_gt_acc = VideoMetricAccumulator()
    student_ref_acc = VideoMetricAccumulator()
    velocity_acc = VelocityMetricAccumulator()
    sums = {
        "student_gt_pixel": 0.0,
        "student_gt_temporal": 0.0,
        "reference_gt_pixel": 0.0,
        "reference_gt_temporal": 0.0,
        "student_ref_pixel": 0.0,
        "student_ref_temporal": 0.0,
    }
    processed = 0
    started = time.perf_counter()
    autocast_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}

    with torch.no_grad():
        for batch_cpu in loader:
            if processed >= expected_samples:
                break
            batch = _move_batch(batch_cpu, device, dtype)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                output = closure(batch)
            student_prediction = output["prediction"]
            student_velocity = output["velocity"]
            target = output["target"]
            if not all(isinstance(value, torch.Tensor) for value in (student_prediction, student_velocity, target)):
                raise TypeError("Student output is missing prediction/velocity/target")

            for local_index in range(int(student_prediction.shape[0])):
                if processed >= expected_samples:
                    break
                identity = batch_sample_identity(batch_cpu, local_index)
                reference = cache.load(identity, device=device, dtype=dtype)
                student_i = student_prediction[local_index : local_index + 1]
                target_i = target[local_index : local_index + 1]
                reference_i = reference["prediction"].unsqueeze(0)
                student_v_i = student_velocity[local_index : local_index + 1]
                reference_v_i = reference["velocity"].unsqueeze(0)

                student_gt_acc.update(student_i, target_i, clamp=True)
                reference_gt_acc.update(reference_i, target_i, clamp=True)
                student_ref_acc.update(student_i, reference_i, clamp=True)
                velocity_acc.update(student_v_i, reference_v_i)

                sums["student_gt_pixel"] += float(torch.nn.functional.l1_loss(student_i.float(), target_i.float()).item())
                sums["student_gt_temporal"] += float(temporal_difference_mse(student_i, target_i).item())
                sums["reference_gt_pixel"] += float(torch.nn.functional.l1_loss(reference_i.float(), target_i.float()).item())
                sums["reference_gt_temporal"] += float(temporal_difference_mse(reference_i, target_i).item())
                sums["student_ref_pixel"] += float(torch.nn.functional.l1_loss(student_i.float(), reference_i.float()).item())
                sums["student_ref_temporal"] += float(temporal_difference_mse(student_i, reference_i).item())
                processed += 1
                print(f"evaluated {processed}/{expected_samples}: {identity['record_uid']}", flush=True)

    if processed != expected_samples:
        raise RuntimeError(
            f"Evaluated {processed} samples but cache requires {expected_samples}"
        )
    student_gt = _finalize_video_metrics(
        student_gt_acc, sums["student_gt_pixel"], sums["student_gt_temporal"], processed
    )
    reference_gt = _finalize_video_metrics(
        reference_gt_acc, sums["reference_gt_pixel"], sums["reference_gt_temporal"], processed
    )
    student_reference = _finalize_video_metrics(
        student_ref_acc, sums["student_ref_pixel"], sums["student_ref_temporal"], processed
    )
    student_reference.update(velocity_acc.compute())
    result: dict[str, object] = {
        "global_step": int(metadata["step"]),
        "student_checkpoint": (
            None
            if args.student_checkpoint is None
            else str(args.student_checkpoint.expanduser().resolve())
        ),
        "student_base_checkpoint": str(base),
        "reference_cache": str(args.reference_cache.expanduser().resolve()),
        "sample_count": processed,
        "student_gt": student_gt,
        "reference_gt": reference_gt,
        "student_reference": student_reference,
        "gap": {
            "psnr": float(student_gt["psnr"]) - float(reference_gt["psnr"]),
            "ssim": float(student_gt["ssim"]) - float(reference_gt["ssim"]),
            "mae": float(student_gt["mae"]) - float(reference_gt["mae"]),
            "rmse": float(student_gt["rmse"]) - float(reference_gt["rmse"]),
        },
        "eval_seconds": time.perf_counter() - started,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)
    if args.tensorboard_dir is not None:
        _write_tensorboard(args.tensorboard_dir, int(metadata["step"]), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
