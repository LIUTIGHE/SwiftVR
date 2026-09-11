#!/usr/bin/env python3
"""Full deterministic val-set audit: Stage-A teacher vs dense student vs MoE student.

This evaluation-only tool replaces the older D768-vs-D1024-specific audit.  It
loads one prompt-free/no-time dense checkpoint and one sparse-MoE checkpoint,
uses the same Stage-A D3072 velocity cache and frozen ReAE decoder as formal
validation, and exports paired full-val comparisons.

For every sample:
  * comparison.mp4: LQ | Stage-A | dense | MoE | GT
  * differences.mp4: |dense-StageA| |MoE-StageA| |MoE-dense| |StageA-GT|
  * selected full-frame PNGs
  * detail crops whose ROI is selected only from Stage-A high-frequency energy
  * per-sample metrics.json

The output root contains aggregate metrics, a per-sample CSV sorted by the MoE
minus dense Stage-A PSNR delta, and overview sheets.  GT is diagnostic only.
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
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training import (
    DistillationMetricAccumulator,
    TeacherVelocityCache,
    VideoMetricAccumulator,
    decode_student_prediction,
    decode_teacher_prediction,
)
from swiftvr.training.b2a_width import B2ACompactVelocityDistillationForward, transformer_width_shape
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
    p.add_argument("--dense-checkpoint", type=Path, required=True)
    p.add_argument("--moe-checkpoint", type=Path, required=True)
    p.add_argument("--dense-label", default=None)
    p.add_argument("--moe-label", default=None)
    p.add_argument("--teacher-cache", type=Path, required=True, help="Stage-A D3072 val velocity cache")
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


def _auto_dense_label(shape: Mapping[str, object]) -> str:
    return f"D{int(shape['hidden_dim'])}-L{int(shape['num_layers'])}"


def _auto_moe_label(shape: Mapping[str, object]) -> str:
    return f"D{int(shape['hidden_dim'])}-MoE-L{int(shape['num_layers'])}"


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
        str(path), fps=float(fps), codec="libx264", macro_block_size=1, quality=8
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))


def _difference_frame(
    dense: torch.Tensor,
    moe: torch.Tensor,
    teacher: torch.Tensor,
    gt: torch.Tensor,
    *,
    dense_label: str,
    moe_label: str,
    scale: float,
) -> Image.Image:
    return make_comparison_frame(
        OrderedDict(
            (
                (f"|{dense_label}-StageA| x{scale:g}", (dense - teacher).abs().mul(scale).clamp(0, 1)),
                (f"|{moe_label}-StageA| x{scale:g}", (moe - teacher).abs().mul(scale).clamp(0, 1)),
                (f"|{moe_label}-{dense_label}| x{scale:g}", (moe - dense).abs().mul(scale).clamp(0, 1)),
                (f"|StageA-GT| x{scale:g}", (teacher - gt).abs().mul(scale).clamp(0, 1)),
            )
        )
    )


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
            patch.float(), size=(display_size, display_size), mode="bicubic", align_corners=False
        )[0].clamp(0, 1)
        panels[f"{label} detail"] = patch
    return make_comparison_frame(panels)


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


def _flatten_row(record: dict[str, object]) -> dict[str, object]:
    dense_t = record["dense_stage_a"]
    moe_t = record["moe_stage_a"]
    dense_gt = record["dense_gt"]
    moe_gt = record["moe_gt"]
    dense_v = record["dense_velocity_stage_a"]
    moe_v = record["moe_velocity_stage_a"]
    moe_dense = record["moe_dense"]
    assert isinstance(dense_t, dict) and isinstance(moe_t, dict)
    assert isinstance(dense_gt, dict) and isinstance(moe_gt, dict)
    assert isinstance(dense_v, dict) and isinstance(moe_v, dict)
    assert isinstance(moe_dense, dict)
    return {
        "index": record["index"],
        "sample": record["sample"],
        "dense_stage_a_psnr": dense_t["psnr"],
        "moe_stage_a_psnr": moe_t["psnr"],
        "delta_stage_a_psnr_moe_minus_dense": float(moe_t["psnr"]) - float(dense_t["psnr"]),
        "dense_stage_a_ssim": dense_t["ssim"],
        "moe_stage_a_ssim": moe_t["ssim"],
        "delta_stage_a_ssim_moe_minus_dense": float(moe_t["ssim"]) - float(dense_t["ssim"]),
        "dense_gt_psnr": dense_gt["psnr"],
        "moe_gt_psnr": moe_gt["psnr"],
        "moe_dense_psnr": moe_dense["psnr"],
        "dense_velocity_rel_l2": dense_v["velocity_relative_l2"],
        "moe_velocity_rel_l2": moe_v["velocity_relative_l2"],
        "dense_velocity_cosine": dense_v["velocity_cosine"],
        "moe_velocity_cosine": moe_v["velocity_cosine"],
    }


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


def main() -> int:
    args = build_parser().parse_args()
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
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    base_root = args.base_checkpoint.expanduser().resolve()
    dense_root = args.dense_checkpoint.expanduser().resolve()
    moe_root = args.moe_checkpoint.expanduser().resolve()
    reae = ReAE(str(base_root / args.reae_filename)).to(device=device, dtype=dtype).eval()

    dense_transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(dense_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype)
    dense_shape = transformer_width_shape(dense_transformer)
    dense_label = args.dense_label or _auto_dense_label(dense_shape)
    dense_model = B2ACompactVelocityDistillationForward(
        reae,
        dense_transformer,
        attention_backend=args.attention_backend,
        gradient_checkpointing=False,
    ).to(device=device).eval()

    moe_transformer = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
        str(moe_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype)
    moe_shape = transformer_moe_shape(moe_transformer)
    moe_label = args.moe_label or _auto_moe_label(moe_shape)
    moe_model = B2BMoEVelocityDistillationForward(
        reae,
        moe_transformer,
        attention_backend=args.attention_backend,
        gradient_checkpointing=False,
    ).to(device=device).eval()

    aggregate = {
        "dense_velocity_stage_a": DistillationMetricAccumulator(),
        "moe_velocity_stage_a": DistillationMetricAccumulator(),
        "dense_stage_a": VideoMetricAccumulator(),
        "moe_stage_a": VideoMetricAccumulator(),
        "dense_gt": VideoMetricAccumulator(),
        "moe_gt": VideoMetricAccumulator(),
        "stage_a_gt": VideoMetricAccumulator(),
        "moe_dense": VideoMetricAccumulator(),
    }
    records: list[dict[str, object]] = []
    overview_middle: list[Image.Image] = []
    overview_detail: list[Image.Image] = []
    video_errors: list[dict[str, str]] = []

    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode():
        for index, batch_cpu in enumerate(loader):
            if index >= args.max_samples:
                break
            teacher_velocity = cache.load_batch(batch_cpu, device=device, dtype=dtype)
            batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                dense_out = dense_model(batch)
                moe_out = moe_model(batch)
                if not torch.allclose(dense_out["z_lq"], moe_out["z_lq"], atol=0.0, rtol=0.0):
                    raise RuntimeError("Dense and MoE ReAE encodes differ; comparison is not paired")
                target = dense_out["target"]
                if not torch.equal(target, moe_out["target"]):
                    raise RuntimeError("Dense and MoE targets differ")
                output_frames = int(target.shape[1])
                teacher_prediction = decode_teacher_prediction(
                    reae=reae,
                    z_lq=dense_out["z_lq"],
                    teacher_velocity=teacher_velocity,
                    output_frames=output_frames,
                )
                dense_prediction = decode_student_prediction(
                    reae=reae,
                    z_lq=dense_out["z_lq"],
                    student_velocity=dense_out["velocity"],
                    output_frames=output_frames,
                )
                moe_prediction = decode_student_prediction(
                    reae=reae,
                    z_lq=moe_out["z_lq"],
                    student_velocity=moe_out["velocity"],
                    output_frames=output_frames,
                )

            name = _sample_name(batch_cpu, index)
            record: dict[str, object] = {
                "index": index,
                "sample": name,
                "dense_velocity_stage_a": _velocity_metrics(dense_out["velocity"], teacher_velocity),
                "moe_velocity_stage_a": _velocity_metrics(moe_out["velocity"], teacher_velocity),
                "dense_stage_a": _rgb_metrics(dense_prediction, teacher_prediction),
                "moe_stage_a": _rgb_metrics(moe_prediction, teacher_prediction),
                "dense_gt": _rgb_metrics(dense_prediction, target),
                "moe_gt": _rgb_metrics(moe_prediction, target),
                "stage_a_gt": _rgb_metrics(teacher_prediction, target),
                "moe_dense": _rgb_metrics(moe_prediction, dense_prediction),
            }
            records.append(record)

            aggregate["dense_velocity_stage_a"].update(dense_out["velocity"], teacher_velocity)
            aggregate["moe_velocity_stage_a"].update(moe_out["velocity"], teacher_velocity)
            for key, prediction, reference in (
                ("dense_stage_a", dense_prediction, teacher_prediction),
                ("moe_stage_a", moe_prediction, teacher_prediction),
                ("dense_gt", dense_prediction, target),
                ("moe_gt", moe_prediction, target),
                ("stage_a_gt", teacher_prediction, target),
                ("moe_dense", moe_prediction, dense_prediction),
            ):
                aggregate[key].update(prediction, reference, clamp=True)

            sample_dir = out_root / name
            sample_dir.mkdir(parents=True, exist_ok=True)
            lq = dense_out["lq_input"][0].detach().float().cpu()
            gt = target[0].detach().float().cpu()
            teacher = teacher_prediction[0].detach().float().cpu()
            dense = dense_prediction[0].detach().float().cpu()
            moe = moe_prediction[0].detach().float().cpu()
            frames = int(target.shape[1])
            valid_frames = [i for i in frame_indices if i < frames]
            if not valid_frames:
                raise ValueError(f"No selected frame lies inside {frames}-frame clip")

            comparison_video: list[Image.Image] = []
            difference_video: list[Image.Image] = []
            for frame_index in range(frames):
                comparison = make_comparison_frame(
                    OrderedDict(
                        (
                            ("LQ bicubic", lq[frame_index]),
                            ("M1 Stage-A D3072", teacher[frame_index]),
                            (dense_label, dense[frame_index]),
                            (moe_label, moe[frame_index]),
                            ("GT reference", gt[frame_index]),
                        )
                    )
                )
                difference = _difference_frame(
                    dense[frame_index],
                    moe[frame_index],
                    teacher[frame_index],
                    gt[frame_index],
                    dense_label=dense_label,
                    moe_label=moe_label,
                    scale=args.difference_scale,
                )
                comparison_video.append(comparison)
                difference_video.append(difference)
                if frame_index in valid_frames:
                    comparison.save(sample_dir / f"comparison_frame_{frame_index:03d}.png")
                    difference.save(sample_dir / f"difference_frame_{frame_index:03d}.png")
                    top, left, crop = _detail_roi(teacher[frame_index], args.detail_crop_size)
                    detail = _detail_frame(
                        OrderedDict(
                            (
                                ("M1 Stage-A D3072", teacher[frame_index]),
                                (dense_label, dense[frame_index]),
                                (moe_label, moe[frame_index]),
                                ("GT", gt[frame_index]),
                            )
                        ),
                        top=top,
                        left=left,
                        crop=crop,
                        display_size=args.detail_display_size,
                    )
                    detail.save(sample_dir / f"detail_frame_{frame_index:03d}_y{top}_x{left}.png")
                    if frame_index == frames // 2:
                        overview_detail.append(detail)
                if frame_index == frames // 2:
                    overview_middle.append(comparison)

            if not args.no_videos:
                for filename, content in (
                    ("comparison.mp4", comparison_video),
                    ("differences.mp4", difference_video),
                ):
                    try:
                        _write_video(sample_dir / filename, content, args.fps)
                    except Exception as exc:
                        video_errors.append(
                            {"sample": name, "file": filename, "error": f"{type(exc).__name__}: {exc}"}
                        )

            (sample_dir / "metrics.json").write_text(
                json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
            )
            flat = _flatten_row(record)
            print(
                f"[{index + 1}/{min(len(dataset), args.max_samples)}] {name} "
                f"{dense_label}->StageA={float(flat['dense_stage_a_psnr']):.3f} "
                f"{moe_label}->StageA={float(flat['moe_stage_a_psnr']):.3f} "
                f"delta={float(flat['delta_stage_a_psnr_moe_minus_dense']):+.3f} dB",
                flush=True,
            )

    if not records:
        raise RuntimeError("No validation samples were evaluated")

    aggregate_metrics: dict[str, object] = {
        "dense_velocity_stage_a": aggregate["dense_velocity_stage_a"].compute(),
        "moe_velocity_stage_a": aggregate["moe_velocity_stage_a"].compute(),
    }
    for key in ("dense_stage_a", "moe_stage_a", "dense_gt", "moe_gt", "stage_a_gt", "moe_dense"):
        aggregate_metrics[key] = aggregate[key].compute()

    rows = [_flatten_row(record) for record in records]
    rows.sort(key=lambda row: float(row["delta_stage_a_psnr_moe_minus_dense"]))
    wins = sum(float(row["delta_stage_a_psnr_moe_minus_dense"]) > 0 for row in rows)
    ties = sum(abs(float(row["delta_stage_a_psnr_moe_minus_dense"])) < 1e-6 for row in rows)
    summary = {
        "kind": "b2b_stage_a_dense_moe_full_val_audit",
        "base_checkpoint": str(base_root),
        "dense_checkpoint": str(dense_root),
        "moe_checkpoint": str(moe_root),
        "teacher_cache": str(args.teacher_cache.expanduser().resolve()),
        "dense_label": dense_label,
        "moe_label": moe_label,
        "dense_shape": dense_shape,
        "moe_shape": moe_shape,
        "samples": len(records),
        "moe_stage_a_psnr_wins_over_dense": wins,
        "ties": ties,
        "dense_stage_a_psnr_wins_over_moe": len(rows) - wins - ties,
        "aggregate_metrics": aggregate_metrics,
        "video_errors": video_errors,
        "selection_note": "Detail ROIs are chosen only from Stage-A D3072 high-frequency energy.",
        "gt_role": "diagnostic_only",
    }
    (out_root / "aggregate_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_root / "per_sample.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(out_root / "per_sample_sorted_by_delta.csv", rows)
    _vertical_sheet(overview_middle, out_root / "overview_middle_frames.png")
    _vertical_sheet(overview_detail, out_root / "overview_stage_a_detail_crops.png")

    dense_psnr = float(aggregate_metrics["dense_stage_a"]["psnr"])
    moe_psnr = float(aggregate_metrics["moe_stage_a"]["psnr"])
    dense_rel = float(aggregate_metrics["dense_velocity_stage_a"]["velocity_relative_l2"])
    moe_rel = float(aggregate_metrics["moe_velocity_stage_a"]["velocity_relative_l2"])
    moe_dense_psnr = float(aggregate_metrics["moe_dense"]["psnr"])
    print("\n========== Stage-A vs dense vs MoE full-val audit ==========")
    print(f"Samples                       : {len(records)}")
    print(f"Dense label                   : {dense_label}")
    print(f"MoE label                     : {moe_label}")
    print(f"{dense_label} -> Stage-A PSNR : {dense_psnr:.4f}")
    print(f"{moe_label} -> Stage-A PSNR   : {moe_psnr:.4f} ({moe_psnr - dense_psnr:+.4f} dB)")
    print(f"{dense_label} velocity rel-L2 : {dense_rel:.6f}")
    print(f"{moe_label} velocity rel-L2   : {moe_rel:.6f}")
    print(f"MoE -> dense PSNR             : {moe_dense_psnr:.4f}")
    print(f"MoE per-sample Stage-A wins   : {wins}/{len(rows)}")
    print(f"Output                        : {out_root}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
