#!/usr/bin/env python3
"""Compare any number of M8/MoE checkpoints on the full deterministic val13 set.

The first --model entry is treated as the baseline.  Every model is evaluated on
exactly the same 13 deterministic legacy validation views, with the immutable
Stage-A D3072 velocity cache and the frozen original ReAE decoder.

For each validation sample this tool exports:
  * comparison.mp4: LQ | Stage-A | model... | GT
  * differences.mp4: |model-StageA| for every model plus |StageA-GT|
  * selected full-frame PNGs
  * Stage-A-selected detail crops
  * metrics.json with per-model velocity/RGB/GT metrics

At the output root it writes:
  * aggregate_metrics.json
  * per_sample.json
  * per_sample_metrics.csv
  * overview_middle_frames.png
  * overview_stage_a_detail_crops.png

This is evaluation-only and never mutates checkpoints or caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_teacher_distillation_ddp as stage_a
from tools.smoke_training_forward import move_video_batch
from swiftvr.models import ReAE, WanTransformer3DModelPromptFreeNoTimeMoE
from swiftvr.training import (
    DistillationMetricAccumulator,
    TeacherVelocityCache,
    VideoMetricAccumulator,
    decode_student_prediction,
    decode_teacher_prediction,
)
from swiftvr.training.b2b_moe import transformer_moe_shape
from swiftvr.training.b2b_moe_training import B2BMoEVelocityDistillationForward
from swiftvr.training.perceptual_review import make_comparison_frame, parse_csv_ints


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=CHECKPOINT",
        help="MoE checkpoint to compare. Repeat; first entry is the baseline.",
    )
    p.add_argument(
        "--teacher-cache",
        type=Path,
        required=True,
        help="Stage-A D3072 val13 velocity cache.",
    )
    p.add_argument("--val-manifest", type=Path, action="append", required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--val-split", default="val")
    p.add_argument("--clip-length", type=int, default=13)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--views-per-record", type=int, default=1)
    p.add_argument("--view-seed", type=int, default=9000001)
    p.add_argument("--dtype", choices=tuple(DTYPES), default="float16")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=13)
    p.add_argument("--frame-indices", default="0,3,6,9,12")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--difference-scale", type=float, default=4.0)
    p.add_argument("--detail-crop-size", type=int, default=96)
    p.add_argument("--detail-display-size", type=int, default=288)
    p.add_argument("--no-videos", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    return p


def _parse_models(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw in values:
        label, sep, path = raw.partition("=")
        label = label.strip()
        path = path.strip()
        if not sep or not label or not path:
            raise ValueError(
                f"Invalid --model {raw!r}; expected LABEL=CHECKPOINT"
            )
        if label in seen:
            raise ValueError(f"Duplicate model label {label!r}")
        seen.add(label)
        result.append((label, Path(path).expanduser().resolve()))
    if not result:
        raise ValueError("At least one --model is required")
    return result


def _safe_name(value: object, fallback: str) -> str:
    text = _SAFE.sub("_", str(value)).strip("._-")
    return text or fallback


def _sample_name(batch: Mapping[str, object], index: int) -> str:
    for key in ("sample_id", "record_uid"):
        value = batch.get(key)
        if isinstance(value, (list, tuple)) and value:
            return f"{index:02d}_{_safe_name(value[0], f'sample_{index:02d}')}"
        if value is not None and not isinstance(value, torch.Tensor):
            return f"{index:02d}_{_safe_name(value, f'sample_{index:02d}')}"
    return f"{index:02d}_sample"


def _rgb_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    acc = VideoMetricAccumulator()
    acc.update(prediction, target, clamp=True)
    return acc.compute()


def _velocity_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    acc = DistillationMetricAccumulator()
    acc.update(prediction, target)
    return acc.compute()


def _write_video(path: Path, frames: list[Image.Image], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path),
        fps=float(fps),
        codec="libx264",
        macro_block_size=1,
        quality=8,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))


def _detail_roi(teacher_frame: torch.Tensor, crop_size: int) -> tuple[int, int, int]:
    frame = teacher_frame.detach().float().cpu().clamp(0, 1)
    _, height, width = frame.shape
    crop = min(int(crop_size), height, width)
    if crop <= 0:
        raise ValueError("detail crop size must be positive")
    gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
    gx = F.pad((gray[:, 1:] - gray[:, :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((gray[1:, :] - gray[:-1, :]).abs(), (0, 0, 0, 1))
    energy = (gx + gy)[None, None]
    stride = max(1, crop // 8)
    scores = F.avg_pool2d(energy, kernel_size=crop, stride=stride)
    flat = int(scores.reshape(-1).argmax().item())
    cols = int(scores.shape[-1])
    row, col = divmod(flat, cols)
    top = min(row * stride, height - crop)
    left = min(col * stride, width - crop)
    return int(top), int(left), int(crop)


def _detail_frame(
    frames: Mapping[str, torch.Tensor],
    *,
    top: int,
    left: int,
    crop: int,
    display_size: int,
) -> Image.Image:
    panels = OrderedDict()
    for label, frame in frames.items():
        patch = frame[:, top : top + crop, left : left + crop][None]
        patch = F.interpolate(
            patch.float(),
            size=(display_size, display_size),
            mode="bicubic",
            align_corners=False,
        )[0].clamp(0, 1)
        panels[f"{label} detail"] = patch
    return make_comparison_frame(panels)


def _vertical_sheet(images: list[Image.Image], path: Path) -> None:
    if not images:
        return
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    top = 0
    for image in images:
        sheet.paste(image, (0, top))
        top += image.height
    sheet.save(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_record(
    record: Mapping[str, object],
    labels: list[str],
    baseline: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "index": record["index"],
        "sample": record["sample"],
    }
    model_metrics = record["models"]
    assert isinstance(model_metrics, Mapping)
    baseline_metrics = model_metrics[baseline]
    assert isinstance(baseline_metrics, Mapping)
    baseline_stage_a = baseline_metrics["stage_a"]
    baseline_gt = baseline_metrics["gt"]
    assert isinstance(baseline_stage_a, Mapping)
    assert isinstance(baseline_gt, Mapping)
    for label in labels:
        metrics = model_metrics[label]
        assert isinstance(metrics, Mapping)
        velocity = metrics["velocity_stage_a"]
        stage_a_metrics = metrics["stage_a"]
        gt_metrics = metrics["gt"]
        assert isinstance(velocity, Mapping)
        assert isinstance(stage_a_metrics, Mapping)
        assert isinstance(gt_metrics, Mapping)
        prefix = _safe_name(label, "model")
        row[f"{prefix}_velocity_rel_l2"] = velocity["velocity_relative_l2"]
        row[f"{prefix}_velocity_cosine"] = velocity["velocity_cosine"]
        row[f"{prefix}_stage_a_psnr"] = stage_a_metrics["psnr"]
        row[f"{prefix}_stage_a_ssim"] = stage_a_metrics["ssim"]
        row[f"{prefix}_gt_psnr"] = gt_metrics["psnr"]
        row[f"{prefix}_gt_ssim"] = gt_metrics["ssim"]
        row[f"{prefix}_delta_stage_a_psnr_vs_{_safe_name(baseline, 'baseline')}"] = (
            float(stage_a_metrics["psnr"]) - float(baseline_stage_a["psnr"])
        )
        row[f"{prefix}_delta_gt_psnr_vs_{_safe_name(baseline, 'baseline')}"] = (
            float(gt_metrics["psnr"]) - float(baseline_gt["psnr"])
        )
    return row


def main() -> int:
    args = build_parser().parse_args()
    model_specs = _parse_models(args.model)
    labels = [label for label, _ in model_specs]
    baseline_label = labels[0]
    frame_indices = parse_csv_ints(args.frame_indices)
    if args.max_samples <= 0 or args.fps <= 0 or args.difference_scale <= 0:
        raise ValueError("max-samples/fps/difference-scale must be positive")
    if args.detail_crop_size <= 0 or args.detail_display_size <= 0:
        raise ValueError("detail crop/display sizes must be positive")

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    out_root = args.output_dir.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    cache = TeacherVelocityCache(args.teacher_cache)
    if cache.metadata.get("kind") != "swiftvr_b2a_stage_a_teacher_velocity":
        raise ValueError("Expected Stage-A teacher validation cache")
    dataset = stage_a.build_cached_dataset(
        args.val_manifest,
        cache,
        split=args.val_split,
        path_root=args.path_root,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
        hflip=0.0,
        vflip=0.0,
        verify_paths=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    base_root = args.base_checkpoint.expanduser().resolve()
    reae = ReAE(str(base_root / args.reae_filename)).to(
        device=device, dtype=dtype
    ).eval()

    models: OrderedDict[str, B2BMoEVelocityDistillationForward] = OrderedDict()
    shapes: dict[str, object] = {}
    for label, checkpoint in model_specs:
        transformer = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
            str(checkpoint),
            subfolder=args.transformer_subfolder,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device=device, dtype=dtype)
        shape = transformer_moe_shape(transformer)
        if int(shape["num_layers"]) != 20:
            raise ValueError(
                f"{label}: expected M8 L20 checkpoint, got shape={shape}"
            )
        model = B2BMoEVelocityDistillationForward(
            reae,
            transformer,
            attention_backend=args.attention_backend,
            gradient_checkpointing=False,
        ).to(device=device).eval()
        models[label] = model
        shapes[label] = shape

    aggregate_velocity = {
        label: DistillationMetricAccumulator() for label in labels
    }
    aggregate_stage_a = {
        label: VideoMetricAccumulator() for label in labels
    }
    aggregate_gt = {
        label: VideoMetricAccumulator() for label in labels
    }
    aggregate_teacher_gt = VideoMetricAccumulator()

    records: list[dict[str, object]] = []
    overview_middle: list[Image.Image] = []
    overview_detail: list[Image.Image] = []
    video_errors: list[dict[str, str]] = []

    autocast_enabled = device.type == "cuda" and dtype in (
        torch.float16,
        torch.bfloat16,
    )
    with torch.inference_mode():
        for sample_index, batch_cpu in enumerate(loader):
            if sample_index >= args.max_samples:
                break
            teacher_velocity = cache.load_batch(
                batch_cpu,
                device=device,
                dtype=dtype,
            )
            batch = move_video_batch(batch_cpu, device=device, dtype=dtype)

            outputs: OrderedDict[str, Mapping[str, object]] = OrderedDict()
            predictions: OrderedDict[str, torch.Tensor] = OrderedDict()
            with torch.autocast(
                device_type=device.type,
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                for label, model in models.items():
                    output = model(batch)
                    outputs[label] = output
                first_output = outputs[baseline_label]
                z_lq = first_output["z_lq"]
                target = first_output["target"]
                if not isinstance(z_lq, torch.Tensor) or not isinstance(target, torch.Tensor):
                    raise TypeError("M8 validation requires z_lq/target tensors")
                for label, output in outputs.items():
                    candidate_z = output["z_lq"]
                    candidate_target = output["target"]
                    if not isinstance(candidate_z, torch.Tensor) or not torch.equal(
                        candidate_z, z_lq
                    ):
                        raise RuntimeError(
                            f"{label}: ReAE encoding differs from baseline"
                        )
                    if not isinstance(candidate_target, torch.Tensor) or not torch.equal(
                        candidate_target, target
                    ):
                        raise RuntimeError(f"{label}: GT target differs from baseline")
                    velocity = output["velocity"]
                    if not isinstance(velocity, torch.Tensor):
                        raise TypeError(f"{label}: velocity is not a tensor")
                    predictions[label] = decode_student_prediction(
                        reae=reae,
                        z_lq=z_lq,
                        student_velocity=velocity,
                        output_frames=int(target.shape[1]),
                    )
                teacher_prediction = decode_teacher_prediction(
                    reae=reae,
                    z_lq=z_lq,
                    teacher_velocity=teacher_velocity,
                    output_frames=int(target.shape[1]),
                )

            name = _sample_name(batch_cpu, sample_index)
            model_record: dict[str, object] = {}
            for label in labels:
                output = outputs[label]
                velocity = output["velocity"]
                assert isinstance(velocity, torch.Tensor)
                prediction = predictions[label]
                model_record[label] = {
                    "velocity_stage_a": _velocity_metrics(
                        velocity, teacher_velocity
                    ),
                    "stage_a": _rgb_metrics(
                        prediction, teacher_prediction
                    ),
                    "gt": _rgb_metrics(prediction, target),
                }
                aggregate_velocity[label].update(velocity, teacher_velocity)
                aggregate_stage_a[label].update(
                    prediction, teacher_prediction, clamp=True
                )
                aggregate_gt[label].update(prediction, target, clamp=True)
            teacher_gt = _rgb_metrics(teacher_prediction, target)
            aggregate_teacher_gt.update(teacher_prediction, target, clamp=True)

            record: dict[str, object] = {
                "index": sample_index,
                "sample": name,
                "models": model_record,
                "stage_a_gt": teacher_gt,
            }
            records.append(record)

            sample_dir = out_root / name
            sample_dir.mkdir(parents=True, exist_ok=True)
            lq = outputs[baseline_label]["lq_input"]
            if not isinstance(lq, torch.Tensor):
                raise TypeError("Baseline output lacks lq_input tensor")
            lq_cpu = lq[0].detach().float().cpu()
            gt_cpu = target[0].detach().float().cpu()
            teacher_cpu = teacher_prediction[0].detach().float().cpu()
            prediction_cpu = OrderedDict(
                (label, prediction[0].detach().float().cpu())
                for label, prediction in predictions.items()
            )
            frames = int(target.shape[1])
            valid_frames = [index for index in frame_indices if index < frames]
            if not valid_frames:
                raise ValueError(
                    f"No selected frame lies inside {frames}-frame clip"
                )

            comparison_video: list[Image.Image] = []
            difference_video: list[Image.Image] = []
            middle_index = frames // 2
            for frame_index in range(frames):
                comparison_panels = OrderedDict()
                comparison_panels["LQ bicubic"] = lq_cpu[frame_index]
                comparison_panels["Stage-A D3072"] = teacher_cpu[frame_index]
                for label in labels:
                    comparison_panels[label] = prediction_cpu[label][frame_index]
                comparison_panels["GT"] = gt_cpu[frame_index]
                comparison = make_comparison_frame(comparison_panels)

                difference_panels = OrderedDict()
                for label in labels:
                    difference_panels[
                        f"|{label}-StageA| x{args.difference_scale:g}"
                    ] = (
                        prediction_cpu[label][frame_index]
                        - teacher_cpu[frame_index]
                    ).abs().mul(args.difference_scale).clamp(0, 1)
                difference_panels[
                    f"|StageA-GT| x{args.difference_scale:g}"
                ] = (
                    teacher_cpu[frame_index] - gt_cpu[frame_index]
                ).abs().mul(args.difference_scale).clamp(0, 1)
                difference = make_comparison_frame(difference_panels)

                comparison_video.append(comparison)
                difference_video.append(difference)

                if frame_index in valid_frames:
                    comparison.save(
                        sample_dir / f"comparison_frame_{frame_index:03d}.png"
                    )
                    difference.save(
                        sample_dir / f"difference_frame_{frame_index:03d}.png"
                    )
                    top, left, crop = _detail_roi(
                        teacher_cpu[frame_index],
                        args.detail_crop_size,
                    )
                    detail_panels = OrderedDict()
                    detail_panels["Stage-A D3072"] = teacher_cpu[frame_index]
                    for label in labels:
                        detail_panels[label] = prediction_cpu[label][frame_index]
                    detail_panels["GT"] = gt_cpu[frame_index]
                    detail = _detail_frame(
                        detail_panels,
                        top=top,
                        left=left,
                        crop=crop,
                        display_size=args.detail_display_size,
                    )
                    detail.save(
                        sample_dir
                        / f"detail_frame_{frame_index:03d}_y{top}_x{left}.png"
                    )
                    if frame_index == middle_index:
                        overview_detail.append(detail)

                if frame_index == middle_index:
                    overview_middle.append(comparison)

            if not args.no_videos:
                for filename, content in (
                    ("comparison.mp4", comparison_video),
                    ("differences.mp4", difference_video),
                ):
                    try:
                        _write_video(
                            sample_dir / filename,
                            content,
                            args.fps,
                        )
                    except Exception as exc:
                        video_errors.append(
                            {
                                "sample": name,
                                "file": filename,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

            (sample_dir / "metrics.json").write_text(
                json.dumps(record, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            flat = _flatten_record(record, labels, baseline_label)
            status = " ".join(
                f"{label}:StageA={float(model_record[label]['stage_a']['psnr']):.3f}"
                for label in labels
            )
            print(
                f"[{sample_index + 1}/{min(len(dataset), args.max_samples)}] "
                f"{name} {status}",
                flush=True,
            )

    if not records:
        raise RuntimeError("No validation samples were evaluated")

    aggregate_models: dict[str, object] = {}
    for label in labels:
        aggregate_models[label] = {
            "shape": shapes[label],
            "velocity_stage_a": aggregate_velocity[label].compute(),
            "stage_a": aggregate_stage_a[label].compute(),
            "gt": aggregate_gt[label].compute(),
        }
    aggregate_teacher_gt_metrics = aggregate_teacher_gt.compute()

    flat_rows = [
        _flatten_record(record, labels, baseline_label)
        for record in records
    ]
    baseline_safe = _safe_name(baseline_label, "baseline")
    win_counts: dict[str, object] = {}
    for label in labels[1:]:
        safe = _safe_name(label, "model")
        stage_key = f"{safe}_delta_stage_a_psnr_vs_{baseline_safe}"
        gt_key = f"{safe}_delta_gt_psnr_vs_{baseline_safe}"
        win_counts[label] = {
            "stage_a_psnr_wins": sum(
                float(row[stage_key]) > 0 for row in flat_rows
            ),
            "gt_psnr_wins": sum(
                float(row[gt_key]) > 0 for row in flat_rows
            ),
            "samples": len(flat_rows),
        }

    summary = {
        "kind": "m8_full_val13_multi_checkpoint_audit_v1",
        "base_checkpoint": str(base_root),
        "teacher_cache": str(args.teacher_cache.expanduser().resolve()),
        "models": [
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "shape": shapes[label],
            }
            for label, checkpoint in model_specs
        ],
        "baseline_label": baseline_label,
        "samples": len(records),
        "aggregate_models": aggregate_models,
        "stage_a_gt": aggregate_teacher_gt_metrics,
        "win_counts_vs_baseline": win_counts,
        "video_errors": video_errors,
        "selection_note": (
            "Detail ROIs are selected only from Stage-A D3072 high-frequency energy."
        ),
        "gt_role": "diagnostic comparison reference",
    }

    (out_root / "aggregate_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_root / "per_sample.json").write_text(
        json.dumps(records, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(out_root / "per_sample_metrics.csv", flat_rows)
    _vertical_sheet(
        overview_middle,
        out_root / "overview_middle_frames.png",
    )
    _vertical_sheet(
        overview_detail,
        out_root / "overview_stage_a_detail_crops.png",
    )

    print("\n========== M8 full val13 multi-checkpoint audit ==========")
    print(f"Samples: {len(records)}")
    print(f"Baseline: {baseline_label}")
    for label in labels:
        metrics = aggregate_models[label]
        assert isinstance(metrics, Mapping)
        velocity = metrics["velocity_stage_a"]
        stage_metrics = metrics["stage_a"]
        gt_metrics = metrics["gt"]
        assert isinstance(velocity, Mapping)
        assert isinstance(stage_metrics, Mapping)
        assert isinstance(gt_metrics, Mapping)
        print(
            f"{label:24s} "
            f"relL2={float(velocity['velocity_relative_l2']):.6f} "
            f"StageA_PSNR={float(stage_metrics['psnr']):.4f} "
            f"StageA_SSIM={float(stage_metrics['ssim']):.6f} "
            f"GT_PSNR={float(gt_metrics['psnr']):.4f} "
            f"GT_SSIM={float(gt_metrics['ssim']):.6f}"
        )
    print(
        "Stage-A -> GT: "
        f"PSNR={float(aggregate_teacher_gt_metrics['psnr']):.4f} "
        f"SSIM={float(aggregate_teacher_gt_metrics['ssim']):.6f}"
    )
    print(f"Output: {out_root}")
    print("=========================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
