#!/usr/bin/env python3
"""Diagnose B2B-0C train-vs-validation behavior for the extreme ReAE decoder.

This is a read-only, single-GPU diagnostic.  It evaluates the same trained
SlimReAEDecoder on two deterministic sets drawn from the Stage-A latent caches:

* train13: 13 positions spaced uniformly across the cached formal training views;
* val13:   all 13 fixed validation views.

The purpose is to distinguish a generalization gap from dataset-scale
underfitting.  A large train13->teacher PSNR advantage over val13 suggests a
true train/validation gap.  Similar low train13 and val13 scores suggest that the
1.42M-parameter decoder is underfitting the full teacher-latent mapping itself.

No weights, optimizer state, cache entries, or training files are modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_tiny_decoder_formal_ddp as formal
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.stage3 import VideoMetricAccumulator
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _uniform_positions(length: int, count: int) -> tuple[int, ...]:
    length = int(length)
    count = int(count)
    if length <= 0:
        raise ValueError("length must be positive")
    if count <= 0 or count > length:
        raise ValueError(f"count must lie in [1,{length}], got {count}")
    if count == 1:
        return (length // 2,)
    values = tuple(round(index * (length - 1) / (count - 1)) for index in range(count))
    if len(set(values)) != count:
        raise RuntimeError(f"uniform position construction produced duplicates: {values}")
    return values


def _resolve_student_root(path: Path) -> tuple[Path, dict[str, object]]:
    root = path.expanduser().resolve()
    direct_config = root / "config.json"
    direct_weights = root / "model.safetensors"
    if direct_config.is_file() and direct_weights.is_file():
        return root, {"input": str(root), "resolved_from": "decoder_directory"}

    child = root / "tiny_decoder"
    if (child / "config.json").is_file() and (child / "model.safetensors").is_file():
        return child, {"input": str(root), "resolved_from": "checkpoint_directory"}

    for pointer_name in ("best.json", "latest.json"):
        pointer = root / pointer_name
        if not pointer.is_file():
            continue
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or not isinstance(payload.get("checkpoint"), str):
            raise ValueError(f"Invalid checkpoint pointer: {pointer}")
        checkpoint = Path(str(payload["checkpoint"]))
        checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
        checkpoint = checkpoint.resolve()
        decoder = checkpoint / "tiny_decoder"
        if not (decoder / "config.json").is_file() or not (decoder / "model.safetensors").is_file():
            raise FileNotFoundError(f"Pointer {pointer} does not resolve to a decoder: {decoder}")
        return decoder, {
            "input": str(root),
            "resolved_from": pointer_name,
            "checkpoint": str(checkpoint),
            "pointer": dict(payload),
        }

    raise FileNotFoundError(
        f"Could not resolve SlimReAEDecoder from {root}; expected a decoder directory, "
        "an epoch checkpoint with tiny_decoder/, or a run directory with best.json/latest.json"
    )


def _build_views(
    manifests: Sequence[Path],
    cache: TinyDecoderLatentCache,
    *,
    split: str,
    path_root: Path,
    clip_length: int,
    crop_size: int,
    scale: int,
    views_per_record: int,
    view_seed: int,
    hflip: float,
    vflip: float,
    verify_paths: bool,
):
    base = TripletVideoDataset(
        manifests,
        split=split,
        training=True,
        clip_length=clip_length,
        crop_size=crop_size,
        scale=scale,
        load_hq=False,
        horizontal_flip_probability=hflip,
        vertical_flip_probability=vflip,
        drop_short_sequences=True,
        path_root=path_root,
        verify_paths=verify_paths,
    )
    views = DeterministicTripletViewDataset(
        base,
        views_per_record=views_per_record,
        view_seed=view_seed,
    )
    cache.validate_dataset(
        manifests=manifests,
        split=split,
        clip_length=clip_length,
        crop_size=crop_size,
        scale=scale,
        views_per_record=views_per_record,
        view_seed=view_seed,
        horizontal_flip_probability=hflip,
        vertical_flip_probability=vflip,
        dataset_length=len(views),
    )
    return views


def _evaluate_subset(
    *,
    label: str,
    views,
    cache: TinyDecoderLatentCache,
    positions: Sequence[int],
    teacher: ReAE,
    student: SlimReAEDecoder,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    cached_indices = cache.selected_indices()
    bad = [int(position) for position in positions if position < 0 or position >= len(cached_indices)]
    if bad:
        raise IndexError(f"{label} cache positions out of range: {bad}")
    dataset_indices = [cached_indices[int(position)] for position in positions]
    loader = DataLoader(
        Subset(views, dataset_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    student_teacher = VideoMetricAccumulator()
    student_gt = VideoMetricAccumulator()
    teacher_gt = VideoMetricAccumulator()
    per_sample: list[dict[str, object]] = []
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    with torch.inference_mode():
        for position, dataset_index, batch_cpu in zip(positions, dataset_indices, loader):
            moved = formal._move_pixels(batch_cpu, device, dtype)
            prepared = prepare_training_batch(moved)
            target = prepared["target"]
            if not isinstance(target, torch.Tensor):
                raise TypeError(f"{label} batch is missing target")
            z_sr = cache.load_batch(batch_cpu, device=device, dtype=dtype)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                teacher_rgb = decode_reae_clip(
                    teacher,
                    z_sr,
                    output_frames=int(target.shape[1]),
                    clamp=False,
                )
                prediction = student(
                    z_sr,
                    output_frames=int(target.shape[1]),
                    clamp=False,
                )

            student_teacher.update(prediction, teacher_rgb, clamp=True)
            student_gt.update(prediction, target, clamp=True)
            teacher_gt.update(teacher_rgb, target, clamp=True)

            pred_c = prediction.float().clamp(0, 1)
            teacher_c = teacher_rgb.float().clamp(0, 1)
            target_c = target.float().clamp(0, 1)
            teacher_mse = float(F.mse_loss(pred_c, teacher_c).item())
            gt_mse = float(F.mse_loss(pred_c, target_c).item())
            per_sample.append(
                {
                    "cache_position": int(position),
                    "dataset_index": int(dataset_index),
                    "student_teacher_mse": teacher_mse,
                    "student_teacher_psnr": float("inf") if teacher_mse <= 0 else -10.0 * math.log10(teacher_mse),
                    "student_gt_mse": gt_mse,
                    "student_gt_psnr": float("inf") if gt_mse <= 0 else -10.0 * math.log10(gt_mse),
                }
            )

    result: dict[str, object] = {
        "label": label,
        "samples": len(per_sample),
        "cache_positions": [int(value) for value in positions],
        "dataset_indices": [int(value) for value in dataset_indices],
        "per_sample": per_sample,
    }
    for prefix, accumulator in (
        ("student_teacher", student_teacher),
        ("student_gt", student_gt),
        ("reae_teacher_gt", teacher_gt),
    ):
        for key, value in accumulator.compute().items():
            result[f"{prefix}_{key}"] = value
    return result


def _interpret(train_psnr: float, val_psnr: float) -> dict[str, object]:
    gap = float(train_psnr - val_psnr)
    if train_psnr >= 30.0 and val_psnr >= 30.0:
        label = "BOTH_STRONG"
    elif gap >= 3.0 and train_psnr >= 25.0:
        label = "GENERALIZATION_GAP_DOMINANT"
    elif gap >= 3.0:
        label = "MIXED_UNDERFIT_AND_GENERALIZATION_GAP"
    elif train_psnr < 25.0 and val_psnr < 25.0:
        label = "GLOBAL_UNDERFITTING_DOMINANT"
    else:
        label = "AMBIGUOUS"
    return {
        "heuristic": label,
        "train_minus_val_student_teacher_psnr_db": gap,
        "note": (
            "This label is a diagnostic heuristic, not a training gate. Inspect the absolute "
            "train/val metrics and visuals before changing the architecture."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True, help="B2B-0C run dir, epoch checkpoint, or tiny_decoder dir")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--val-manifest", type=Path, action="append", required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--val-crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--views-per-record", type=int, default=8)
    parser.add_argument("--view-seed", type=int, default=20260805)
    parser.add_argument("--val-views-per-record", type=int, default=1)
    parser.add_argument("--val-view-seed", type=int, default=9000001)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.0)
    parser.add_argument("--val-horizontal-flip-probability", type=float, default=0.0)
    parser.add_argument("--val-vertical-flip-probability", type=float, default=0.0)
    parser.add_argument("--train-samples", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = DTYPES[args.dtype]
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    path_root = args.path_root.expanduser().resolve()
    train_cache = TinyDecoderLatentCache(args.train_cache)
    val_cache = TinyDecoderLatentCache(args.val_cache)
    train_views = _build_views(
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
    val_views = _build_views(
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

    train_positions = _uniform_positions(len(train_cache.selected_indices()), args.train_samples)
    val_positions = tuple(range(len(val_cache.selected_indices())))
    student_root, student_resolution = _resolve_student_root(args.student)

    base = args.base_checkpoint.expanduser().resolve()
    teacher = ReAE(str(base / args.reae_filename)).to(device=device, dtype=dtype).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = SlimReAEDecoder.from_pretrained(student_root, device=device, dtype=dtype).eval()

    train_result = _evaluate_subset(
        label="train13",
        views=train_views,
        cache=train_cache,
        positions=train_positions,
        teacher=teacher,
        student=student,
        device=device,
        dtype=dtype,
    )
    val_result = _evaluate_subset(
        label="val13",
        views=val_views,
        cache=val_cache,
        positions=val_positions,
        teacher=teacher,
        student=student,
        device=device,
        dtype=dtype,
    )

    train_psnr = float(train_result["student_teacher_psnr"])
    val_psnr = float(val_result["student_teacher_psnr"])
    report = {
        "status": "PASS",
        "student": str(student_root),
        "student_resolution": student_resolution,
        "student_channels": list(student.channels),
        "train": train_result,
        "val": val_result,
        "interpretation": _interpret(train_psnr, val_psnr),
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("================ B2B-0C train/val gap diagnostic ================")
    print(f"Student channels             : {list(student.channels)}")
    print(f"Train13 -> Teacher PSNR      : {train_psnr:.4f} dB")
    print(f"Train13 -> Teacher SSIM      : {float(train_result['student_teacher_ssim']):.6f}")
    print(f"Train13 -> GT PSNR           : {float(train_result['student_gt_psnr']):.4f} dB")
    print(f"Val13   -> Teacher PSNR      : {val_psnr:.4f} dB")
    print(f"Val13   -> Teacher SSIM      : {float(val_result['student_teacher_ssim']):.6f}")
    print(f"Val13   -> GT PSNR           : {float(val_result['student_gt_psnr']):.4f} dB")
    print(f"Train-Val teacher PSNR gap   : {train_psnr - val_psnr:+.4f} dB")
    print(f"Heuristic interpretation     : {report['interpretation']['heuristic']}")
    print(f"Saved                        : {output}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
