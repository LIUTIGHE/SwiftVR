#!/usr/bin/env python3
"""Visualize canonical and resize-conv Stage-B1 Tiny Decoder variants together.

This is an isolated read-only review tool. It uses one deterministic validation
view/cache contract and decodes each cached z_SR with the same frozen ReAE teacher,
then compares any number of Tiny Decoder checkpoints side by side. Checkpoint class
is selected from config.json, so canonical PixelShuffle TinyConditionalDecoder and
ResizeConvTinyConditionalDecoder can be reviewed in one run without changing either
model implementation or the existing single-checkpoint visualizer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder
from swiftvr.models.tiny_conditional_decoder_resize_conv import (
    ResizeConvTinyConditionalDecoder,
)
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.perceptual_review import make_comparison_frame
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
SUPPORTED_DECODER_CLASSES = {
    "TinyConditionalDecoder": TinyConditionalDecoder,
    "ResizeConvTinyConditionalDecoder": ResizeConvTinyConditionalDecoder,
}


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated integers")
    return result


def _safe_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise argparse.ArgumentTypeError("decoder label cannot be empty")
    return cleaned


def _parse_decoder_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--decoder expects LABEL=PATH")
    raw_label, raw_path = value.split("=", 1)
    label = _safe_label(raw_label)
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("decoder path cannot be empty")
    return label, Path(raw_path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--decoder",
        type=_parse_decoder_spec,
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat to compare multiple canonical/resize-conv Tiny checkpoints.",
    )
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
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.0)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.0)
    parser.add_argument("--sample-indices", type=_csv_ints, default=(0, 6, 12))
    parser.add_argument("--frame-indices", type=_csv_ints, default=(0, 6, 12))
    parser.add_argument("--difference-scale", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.clip_length <= 0 or args.crop_size <= 0 or args.scale <= 0:
        raise ValueError("clip-length, crop-size and scale must be positive")
    if args.clip_length % 4 != 1:
        raise ValueError("SwiftVR visual clips must satisfy T=4k+1")
    if args.difference_scale <= 0 or args.fps <= 0:
        raise ValueError("difference-scale and fps must be positive")
    for name, value in (
        ("horizontal-flip-probability", args.horizontal_flip_probability),
        ("vertical-flip-probability", args.vertical_flip_probability),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    labels = [label for label, _ in args.decoder]
    if len(labels) != len(set(labels)):
        raise ValueError(f"decoder labels must be unique, got {labels}")


def _move_pixels(batch: Mapping[str, object], device: torch.device, dtype: torch.dtype):
    result = dict(batch)
    for key in ("lr", "hr"):
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, dtype=dtype, non_blocking=True)
    return result


def _write_video(path: Path, frames: list[Image.Image], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path), fps=float(fps), codec="libx264", macro_block_size=1, quality=8
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a_f = a.float().clamp(0.0, 1.0)
    b_f = b.float().clamp(0.0, 1.0)
    mse = float(F.mse_loss(a_f, b_f).item())
    mae = float(F.l1_loss(a_f, b_f).item())
    rmse = math.sqrt(max(mse, 0.0))
    psnr = float("inf") if mse <= 0.0 else -10.0 * math.log10(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse, "psnr": psnr}


def _sample_uid(batch: Mapping[str, object], fallback: str) -> str:
    raw = batch.get("record_uid", [fallback])
    if isinstance(raw, (list, tuple)):
        return str(raw[0])
    return str(raw)


def _checkpoint_class_name(root: Path) -> str:
    config_path = root.expanduser().resolve() / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    class_name = str(config.get("class_name", ""))
    if class_name not in SUPPORTED_DECODER_CLASSES:
        raise ValueError(
            f"Unsupported Tiny decoder class_name={class_name!r} in {config_path}; "
            f"supported={sorted(SUPPORTED_DECODER_CLASSES)}"
        )
    return class_name


def _load_decoder(root: Path, *, device: torch.device, dtype: torch.dtype):
    resolved = root.expanduser().resolve()
    class_name = _checkpoint_class_name(resolved)
    cls = SUPPORTED_DECODER_CLASSES[class_name]
    model = cls.from_pretrained(resolved, device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, class_name


def _comparison_frame(
    lq: torch.Tensor,
    gt: torch.Tensor,
    teacher: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
) -> Image.Image:
    panels: OrderedDict[str, torch.Tensor] = OrderedDict(
        (("LQ bicubic", lq), ("GT", gt), ("ReAE teacher", teacher))
    )
    for label, prediction in predictions.items():
        panels[label] = prediction
    return make_comparison_frame(panels)


def _teacher_difference_frame(
    teacher: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    *,
    scale: float,
) -> Image.Image:
    panels: OrderedDict[str, torch.Tensor] = OrderedDict()
    for label, prediction in predictions.items():
        panels[f"|{label}-ReAE| x{scale:g}"] = (
            prediction - teacher
        ).abs().mul(scale).clamp(0, 1)
    return make_comparison_frame(panels)


def _gt_difference_frame(
    gt: torch.Tensor,
    teacher: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    *,
    scale: float,
) -> Image.Image:
    panels: OrderedDict[str, torch.Tensor] = OrderedDict(
        ((f"|ReAE-GT| x{scale:g}", (teacher - gt).abs().mul(scale).clamp(0, 1)),)
    )
    for label, prediction in predictions.items():
        panels[f"|{label}-GT| x{scale:g}"] = (
            prediction - gt
        ).abs().mul(scale).clamp(0, 1)
    return make_comparison_frame(panels)


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = DTYPES[args.dtype]
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    base = args.base_checkpoint.expanduser().resolve()
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
        horizontal_flip_probability=args.horizontal_flip_probability,
        vertical_flip_probability=args.vertical_flip_probability,
        drop_short_sequences=True,
        path_root=args.path_root.expanduser().resolve(),
        verify_paths=args.verify_paths,
    )
    views = DeterministicTripletViewDataset(
        base_dataset,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
    )
    cache.validate_dataset(
        manifests=args.val_manifest,
        split=args.val_split,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
        horizontal_flip_probability=args.horizontal_flip_probability,
        vertical_flip_probability=args.vertical_flip_probability,
        dataset_length=len(views),
    )
    cached_indices = cache.selected_indices()
    for position in args.sample_indices:
        if position >= len(cached_indices):
            raise IndexError(
                f"sample index {position} exceeds cache sample count {len(cached_indices)}"
            )
    selected_dataset_indices = [cached_indices[position] for position in args.sample_indices]
    loader = DataLoader(
        Subset(views, selected_dataset_indices),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    reae = ReAE(str(base / args.reae_filename)).to(device=device, dtype=dtype).eval()
    for parameter in reae.parameters():
        parameter.requires_grad_(False)

    decoders: OrderedDict[str, torch.nn.Module] = OrderedDict()
    decoder_metadata: dict[str, object] = {}
    for label, path in args.decoder:
        model, class_name = _load_decoder(path, device=device, dtype=dtype)
        decoders[label] = model
        decoder_metadata[label] = {
            "path": str(path.expanduser().resolve()),
            "class_name": class_name,
        }

    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    report: dict[str, object] = {
        "base_checkpoint": str(base),
        "val_cache": str(cache.root),
        "decoders": decoder_metadata,
        "sample_positions": list(args.sample_indices),
        "frame_indices": list(args.frame_indices),
        "difference_scale": float(args.difference_scale),
        "samples": [],
        "aggregate_metrics": {},
        "video_errors": [],
    }
    metric_sums = {
        label: {
            "teacher_mse": 0.0,
            "teacher_mae": 0.0,
            "gt_mse": 0.0,
            "gt_mae": 0.0,
            "samples": 0,
        }
        for label in decoders
    }

    with torch.inference_mode():
        for ordinal, (cache_position, batch_cpu) in enumerate(zip(args.sample_indices, loader)):
            moved = _move_pixels(dict(batch_cpu), device, dtype)
            prepared = prepare_training_batch(moved)
            lq_input = prepared["lq_input"]
            target = prepared["target"]
            if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise TypeError("visual batch is missing lq_input/target")
            z_sr = cache.load_batch(batch_cpu, device=device, dtype=dtype)

            with torch.autocast(
                device_type=device.type,
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                teacher = decode_reae_clip(
                    reae, z_sr, output_frames=int(target.shape[1]), clamp=True
                )
                predictions_gpu: OrderedDict[str, torch.Tensor] = OrderedDict()
                for label, model in decoders.items():
                    predictions_gpu[label] = model(
                        z_sr,
                        lq_input,
                        output_frames=int(target.shape[1]),
                        clamp=True,
                    )

            lq = lq_input[0].float().cpu().clamp(0, 1)
            gt = target[0].float().cpu().clamp(0, 1)
            teacher_cpu = teacher[0].float().cpu().clamp(0, 1)
            predictions = OrderedDict(
                (label, value[0].float().cpu().clamp(0, 1))
                for label, value in predictions_gpu.items()
            )

            frames = int(gt.shape[0])
            valid_frames = tuple(index for index in args.frame_indices if index < frames)
            if not valid_frames:
                raise ValueError(f"No requested frame lies inside {frames}-frame clip")

            uid = _sample_uid(dict(batch_cpu), f"sample_{cache_position:03d}")
            safe_uid = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in uid)
            sample_dir = output / f"sample_{cache_position:03d}_{safe_uid}"
            sample_dir.mkdir(parents=True, exist_ok=False)

            comparison_video: list[Image.Image] = []
            teacher_difference_video: list[Image.Image] = []
            gt_difference_video: list[Image.Image] = []
            for frame_index in range(frames):
                frame_predictions = OrderedDict(
                    (label, value[frame_index]) for label, value in predictions.items()
                )
                comparison = _comparison_frame(
                    lq[frame_index], gt[frame_index], teacher_cpu[frame_index], frame_predictions
                )
                teacher_difference = _teacher_difference_frame(
                    teacher_cpu[frame_index],
                    frame_predictions,
                    scale=float(args.difference_scale),
                )
                gt_difference = _gt_difference_frame(
                    gt[frame_index],
                    teacher_cpu[frame_index],
                    frame_predictions,
                    scale=float(args.difference_scale),
                )
                comparison_video.append(comparison)
                teacher_difference_video.append(teacher_difference)
                gt_difference_video.append(gt_difference)
                if frame_index in valid_frames:
                    comparison.save(sample_dir / f"comparison_frame_{frame_index:03d}.png")
                    teacher_difference.save(
                        sample_dir / f"difference_reae_frame_{frame_index:03d}.png"
                    )
                    gt_difference.save(sample_dir / f"difference_gt_frame_{frame_index:03d}.png")

            if not args.no_videos:
                for filename, content in (
                    ("comparison.mp4", comparison_video),
                    ("differences_reae.mp4", teacher_difference_video),
                    ("differences_gt.mp4", gt_difference_video),
                ):
                    try:
                        _write_video(sample_dir / filename, content, args.fps)
                    except Exception as exc:
                        report["video_errors"].append(
                            {
                                "sample_position": int(cache_position),
                                "file": filename,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

            sample_metrics: dict[str, object] = {}
            for label, prediction_gpu in predictions_gpu.items():
                teacher_metrics = _tensor_metrics(prediction_gpu, teacher)
                gt_metrics = _tensor_metrics(prediction_gpu, target)
                sample_metrics[label] = {
                    "vs_teacher": teacher_metrics,
                    "vs_gt": gt_metrics,
                }
                sums = metric_sums[label]
                sums["teacher_mse"] += teacher_metrics["mse"]
                sums["teacher_mae"] += teacher_metrics["mae"]
                sums["gt_mse"] += gt_metrics["mse"]
                sums["gt_mae"] += gt_metrics["mae"]
                sums["samples"] += 1

            report["samples"].append(
                {
                    "cache_position": int(cache_position),
                    "dataset_index": int(selected_dataset_indices[ordinal]),
                    "record_uid": uid,
                    "frames": frames,
                    "selected_frames": list(valid_frames),
                    "directory": str(sample_dir.relative_to(output)),
                    "metrics": sample_metrics,
                }
            )
            print(f"visualized cache sample {cache_position}: {uid} -> {sample_dir}", flush=True)

    aggregate: dict[str, object] = {}
    for label, sums in metric_sums.items():
        count = max(int(sums["samples"]), 1)
        teacher_mse = float(sums["teacher_mse"]) / count
        gt_mse = float(sums["gt_mse"]) / count
        aggregate[label] = {
            "samples": int(sums["samples"]),
            "vs_teacher": {
                "mean_mse": teacher_mse,
                "mean_mae": float(sums["teacher_mae"]) / count,
                "psnr_from_mean_mse": (
                    float("inf") if teacher_mse <= 0 else -10.0 * math.log10(teacher_mse)
                ),
            },
            "vs_gt": {
                "mean_mse": gt_mse,
                "mean_mae": float(sums["gt_mae"]) / count,
                "psnr_from_mean_mse": (
                    float("inf") if gt_mse <= 0 else -10.0 * math.log10(gt_mse)
                ),
            },
        }
    report["aggregate_metrics"] = aggregate

    (output / "metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"aggregate_metrics": aggregate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
