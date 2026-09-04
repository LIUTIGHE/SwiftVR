#!/usr/bin/env python3
"""Visual comparison for ReAE teacher and one or more SlimReAE decoder checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
from swiftvr.training.perceptual_review import make_comparison_frame
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated integers")
    return result


def _parse_student(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--student expects LABEL=PATH")
    label, raw = value.split("=", 1)
    label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label.strip())
    if not label or not raw.strip():
        raise argparse.ArgumentTypeError("student label/path cannot be empty")
    return label, Path(raw).expanduser()


def _write_video(path: Path, frames: list[Image.Image], fps: float) -> None:
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1, quality=8) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--student", type=_parse_student, action="append", required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--views-per-record", type=int, default=1)
    parser.add_argument("--view-seed", type=int, default=9000001)
    parser.add_argument("--sample-indices", type=_csv_ints, default=(0, 6, 12))
    parser.add_argument("--frame-indices", type=_csv_ints, default=(0, 3, 6, 9, 12))
    parser.add_argument("--difference-scale", type=float, default=16.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cache = TinyDecoderLatentCache(args.val_cache)
    base_dataset = TripletVideoDataset(
        args.val_manifest,
        split=args.val_split,
        training=True,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        load_hq=False,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        drop_short_sequences=True,
        path_root=args.path_root.expanduser().resolve(),
        verify_paths=False,
    )
    views = DeterministicTripletViewDataset(
        base_dataset, views_per_record=args.views_per_record, view_seed=args.view_seed
    )
    cache.validate_dataset(
        manifests=args.val_manifest,
        split=args.val_split,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        dataset_length=len(views),
    )
    cached = cache.selected_indices()
    selected = [cached[position] for position in args.sample_indices]
    loader = DataLoader(Subset(views, selected), batch_size=1, shuffle=False, num_workers=0)

    teacher = ReAE(str(args.base_checkpoint.expanduser().resolve() / args.reae_filename)).to(
        device=device, dtype=dtype
    ).eval()
    students: OrderedDict[str, SlimReAEDecoder] = OrderedDict()
    for label, path in args.student:
        students[label] = SlimReAEDecoder.from_pretrained(path, device=device, dtype=dtype).eval()

    report = {"students": {}, "samples": [], "difference_scale": args.difference_scale}
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    metric_sums = {label: {"teacher_mse": 0.0, "gt_mse": 0.0, "samples": 0} for label in students}

    with torch.inference_mode():
        for position, batch_cpu in zip(args.sample_indices, loader):
            moved = formal._move_pixels(batch_cpu, device, dtype)
            prepared = prepare_training_batch(moved)
            target = prepared["target"]
            lq = prepared["lq_input"]
            assert isinstance(target, torch.Tensor) and isinstance(lq, torch.Tensor)
            z_sr = cache.load_batch(batch_cpu, device=device, dtype=dtype)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                teacher_rgb = decode_reae_clip(
                    teacher, z_sr, output_frames=int(target.shape[1]), clamp=True
                )
                predictions = OrderedDict(
                    (
                        label,
                        model(z_sr, output_frames=int(target.shape[1]), clamp=True),
                    )
                    for label, model in students.items()
                )

            gt = target[0].float().cpu().clamp(0, 1)
            lq_cpu = lq[0].float().cpu().clamp(0, 1)
            teacher_cpu = teacher_rgb[0].float().cpu().clamp(0, 1)
            pred_cpu = OrderedDict((label, value[0].float().cpu().clamp(0, 1)) for label, value in predictions.items())
            sample_dir = output / f"sample_{position:03d}"
            sample_dir.mkdir(parents=True)
            comparison_video: list[Image.Image] = []
            difference_video: list[Image.Image] = []

            for frame_index in range(int(gt.shape[0])):
                panels = OrderedDict(
                    (("LQ bicubic", lq_cpu[frame_index]), ("GT", gt[frame_index]), ("ReAE teacher", teacher_cpu[frame_index]))
                )
                for label, value in pred_cpu.items():
                    panels[label] = value[frame_index]
                comparison = make_comparison_frame(panels)

                diff_panels = OrderedDict()
                for label, value in pred_cpu.items():
                    diff_panels[f"|{label}-ReAE| x{args.difference_scale:g}"] = (
                        value[frame_index] - teacher_cpu[frame_index]
                    ).abs().mul(args.difference_scale).clamp(0, 1)
                difference = make_comparison_frame(diff_panels)
                comparison_video.append(comparison)
                difference_video.append(difference)
                if frame_index in args.frame_indices:
                    comparison.save(sample_dir / f"comparison_frame_{frame_index:03d}.png")
                    difference.save(sample_dir / f"difference_teacher_frame_{frame_index:03d}.png")

            _write_video(sample_dir / "comparison.mp4", comparison_video, args.fps)
            _write_video(sample_dir / "difference_teacher.mp4", difference_video, args.fps)

            sample_metrics = {}
            for label, value in predictions.items():
                teacher_mse = float(F.mse_loss(value.float().clamp(0,1), teacher_rgb.float().clamp(0,1)).item())
                gt_mse = float(F.mse_loss(value.float().clamp(0,1), target.float().clamp(0,1)).item())
                sample_metrics[label] = {
                    "teacher_mse": teacher_mse,
                    "teacher_psnr": float("inf") if teacher_mse <= 0 else -10.0 * math.log10(teacher_mse),
                    "gt_mse": gt_mse,
                    "gt_psnr": float("inf") if gt_mse <= 0 else -10.0 * math.log10(gt_mse),
                }
                metric_sums[label]["teacher_mse"] += teacher_mse
                metric_sums[label]["gt_mse"] += gt_mse
                metric_sums[label]["samples"] += 1
            report["samples"].append({"cache_position": position, "metrics": sample_metrics})

    for label, sums in metric_sums.items():
        count = max(int(sums["samples"]), 1)
        tmse = float(sums["teacher_mse"]) / count
        gmse = float(sums["gt_mse"]) / count
        report["students"][label] = {
            "teacher_mean_mse": tmse,
            "teacher_psnr_from_mean_mse": float("inf") if tmse <= 0 else -10.0 * math.log10(tmse),
            "gt_mean_mse": gmse,
            "gt_psnr_from_mean_mse": float("inf") if gmse <= 0 else -10.0 * math.log10(gmse),
        }
    (output / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["students"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
