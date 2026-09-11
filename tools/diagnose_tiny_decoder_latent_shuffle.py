#!/usr/bin/env python3
"""Diagnose Tiny Decoder dependence on cached z_SR by cyclic latent shuffling.

This is an isolated Stage-B1 visual diagnostic. For each selected validation
sample A, the native LQ/GT stay fixed while its cached z_SR is replaced by the
cached z_SR of another selected sample B:

    native:          Tiny(z_SR(A), LQ(A))
    shuffled latent: Tiny(z_SR(B), LQ(A))

The script also decodes ReAE(z_SR(A)) and ReAE(z_SR(B)) so visual changes can be
attributed to the latent branch. It never loads the Stage-A DiT and never writes
model weights, caches, or training state.
"""

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
from torch.utils.data import DataLoader, Subset, default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.perceptual_review import make_comparison_frame
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated integers")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--tiny-decoder", type=Path, required=True)
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
    parser.add_argument(
        "--shuffle-offset",
        type=int,
        default=1,
        help="Cyclic offset inside --sample-indices; default maps each sample to the next one.",
    )
    parser.add_argument("--difference-scale", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    return parser


def _move_pixels(batch: dict[str, object], device: torch.device, dtype: torch.dtype):
    result = dict(batch)
    for key in ("lr", "hr"):
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, dtype=dtype, non_blocking=True)
    return result


def _sample_uid(batch: dict[str, object], fallback: str) -> str:
    raw = batch.get("record_uid", [fallback])
    if isinstance(raw, (list, tuple)):
        return str(raw[0])
    return str(raw)


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a_f = a.float().clamp(0.0, 1.0)
    b_f = b.float().clamp(0.0, 1.0)
    mse = float(F.mse_loss(a_f, b_f).item())
    mae = float(F.l1_loss(a_f, b_f).item())
    rmse = math.sqrt(max(mse, 0.0))
    psnr = float("inf") if mse <= 0.0 else -10.0 * math.log10(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse, "psnr": psnr}


def _write_video(path: Path, frames: list[Image.Image], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path), fps=float(fps), codec="libx264", macro_block_size=1, quality=8
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))


def _difference_frame(
    native_tiny: torch.Tensor,
    shuffled_tiny: torch.Tensor,
    native_teacher: torch.Tensor,
    shuffled_teacher: torch.Tensor,
    *,
    scale: float,
) -> Image.Image:
    return make_comparison_frame(
        OrderedDict(
            (
                (
                    f"|Tiny native-Tiny shuffled-z| x{scale:g}",
                    (native_tiny - shuffled_tiny).abs().mul(scale).clamp(0, 1),
                ),
                (
                    f"|Tiny native-ReAE native| x{scale:g}",
                    (native_tiny - native_teacher).abs().mul(scale).clamp(0, 1),
                ),
                (
                    f"|Tiny shuffled-z-ReAE shuffled-z| x{scale:g}",
                    (shuffled_tiny - shuffled_teacher).abs().mul(scale).clamp(0, 1),
                ),
                (
                    f"|ReAE native-ReAE shuffled-z| x{scale:g}",
                    (native_teacher - shuffled_teacher).abs().mul(scale).clamp(0, 1),
                ),
            )
        )
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_length <= 0 or args.crop_size <= 0 or args.scale <= 0:
        raise ValueError("clip-length, crop-size and scale must be positive")
    if args.clip_length % 4 != 1:
        raise ValueError("SwiftVR diagnostic clips must satisfy T=4k+1")
    if args.difference_scale <= 0 or args.fps <= 0:
        raise ValueError("difference-scale and fps must be positive")
    if len(args.sample_indices) < 2:
        raise ValueError("latent shuffle requires at least two --sample-indices")
    if args.shuffle_offset % len(args.sample_indices) == 0:
        raise ValueError(
            "--shuffle-offset maps every sample to itself; choose a non-zero offset "
            "modulo the number of selected samples"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = DTYPES[args.dtype]
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    base = args.base_checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
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
    tiny = TinyConditionalDecoder.from_pretrained(
        args.tiny_decoder, device=device, dtype=dtype
    ).eval()
    for module in (reae, tiny):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    report: dict[str, object] = {
        "diagnostic": "latent_shuffle",
        "base_checkpoint": str(base),
        "tiny_decoder": str(args.tiny_decoder.expanduser().resolve()),
        "val_cache": str(cache.root),
        "sample_positions": list(args.sample_indices),
        "frame_indices": list(args.frame_indices),
        "shuffle_offset": int(args.shuffle_offset),
        "difference_scale": float(args.difference_scale),
        "samples": [],
        "video_errors": [],
    }
    rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for ordinal, (native_position, batch_cpu) in enumerate(zip(args.sample_indices, loader)):
            moved = _move_pixels(dict(batch_cpu), device, dtype)
            prepared = prepare_training_batch(moved)
            lq_input = prepared["lq_input"]
            target = prepared["target"]
            if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise TypeError("native validation sample is missing lq_input/target")
            z_native = cache.load_batch(batch_cpu, device=device, dtype=dtype)

            shuffled_ordinal = (ordinal + args.shuffle_offset) % len(selected_dataset_indices)
            shuffled_position = int(args.sample_indices[shuffled_ordinal])
            shuffled_dataset_index = selected_dataset_indices[shuffled_ordinal]
            shuffled_sample = views[shuffled_dataset_index]
            shuffled_batch_cpu = default_collate([shuffled_sample])
            shuffled_moved = _move_pixels(dict(shuffled_batch_cpu), device, dtype)
            shuffled_prepared = prepare_training_batch(shuffled_moved)
            shuffled_target = shuffled_prepared["target"]
            if not isinstance(shuffled_target, torch.Tensor):
                raise TypeError("shuffled validation sample is missing target")
            z_shuffled = cache.load_batch(shuffled_batch_cpu, device=device, dtype=dtype)

            if z_native.shape != z_shuffled.shape:
                raise ValueError(
                    "Native/shuffled latent geometry differs: "
                    f"native={tuple(z_native.shape)}, shuffled={tuple(z_shuffled.shape)}"
                )
            if target.shape != shuffled_target.shape:
                raise ValueError(
                    "Native/shuffled target geometry differs: "
                    f"native={tuple(target.shape)}, shuffled={tuple(shuffled_target.shape)}"
                )

            with torch.autocast(
                device_type=device.type,
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                teacher_native = decode_reae_clip(
                    reae, z_native, output_frames=int(target.shape[1]), clamp=True
                )
                teacher_shuffled = decode_reae_clip(
                    reae, z_shuffled, output_frames=int(target.shape[1]), clamp=True
                )
                tiny_native = tiny(
                    z_native, lq_input, output_frames=int(target.shape[1]), clamp=True
                )
                tiny_shuffled = tiny(
                    z_shuffled, lq_input, output_frames=int(target.shape[1]), clamp=True
                )

            native_uid = _sample_uid(dict(batch_cpu), f"sample_{native_position:03d}")
            shuffled_uid = _sample_uid(
                dict(shuffled_batch_cpu), f"sample_{shuffled_position:03d}"
            )

            native_shuffled = _tensor_metrics(tiny_native, tiny_shuffled)
            native_teacher = _tensor_metrics(tiny_native, teacher_native)
            shuffled_teacher = _tensor_metrics(tiny_shuffled, teacher_shuffled)
            teacher_native_shuffled = _tensor_metrics(teacher_native, teacher_shuffled)
            mse_ratio = native_shuffled["mse"] / max(native_teacher["mse"], 1e-12)
            mae_ratio = native_shuffled["mae"] / max(native_teacher["mae"], 1e-12)

            row: dict[str, object] = {
                "native_cache_position": int(native_position),
                "native_record_uid": native_uid,
                "shuffled_cache_position": shuffled_position,
                "shuffled_record_uid": shuffled_uid,
                "native_shuffled": native_shuffled,
                "native_teacher": native_teacher,
                "shuffled_teacher": shuffled_teacher,
                "teacher_native_shuffled": teacher_native_shuffled,
                "native_shuffled_over_native_teacher_mse": mse_ratio,
                "native_shuffled_over_native_teacher_mae": mae_ratio,
            }
            rows.append(row)

            lq_cpu = lq_input[0].float().cpu().clamp(0, 1)
            gt_cpu = target[0].float().cpu().clamp(0, 1)
            teacher_native_cpu = teacher_native[0].float().cpu().clamp(0, 1)
            teacher_shuffled_cpu = teacher_shuffled[0].float().cpu().clamp(0, 1)
            tiny_native_cpu = tiny_native[0].float().cpu().clamp(0, 1)
            tiny_shuffled_cpu = tiny_shuffled[0].float().cpu().clamp(0, 1)

            frames = int(gt_cpu.shape[0])
            valid_frames = tuple(index for index in args.frame_indices if index < frames)
            if not valid_frames:
                raise ValueError(f"No requested frame lies inside {frames}-frame clip")

            safe_uid = "".join(
                ch if ch.isalnum() or ch in "._-" else "_" for ch in native_uid
            )
            sample_dir = output / f"sample_{native_position:03d}_{safe_uid}"
            sample_dir.mkdir(parents=True, exist_ok=False)
            comparison_video: list[Image.Image] = []
            difference_video: list[Image.Image] = []

            for frame_index in range(frames):
                comparison = make_comparison_frame(
                    OrderedDict(
                        (
                            ("Native LQ", lq_cpu[frame_index]),
                            ("Native GT", gt_cpu[frame_index]),
                            ("ReAE native-z", teacher_native_cpu[frame_index]),
                            ("ReAE shuffled-z", teacher_shuffled_cpu[frame_index]),
                            ("Tiny native", tiny_native_cpu[frame_index]),
                            ("Tiny shuffled-z", tiny_shuffled_cpu[frame_index]),
                        )
                    )
                )
                difference = _difference_frame(
                    tiny_native_cpu[frame_index],
                    tiny_shuffled_cpu[frame_index],
                    teacher_native_cpu[frame_index],
                    teacher_shuffled_cpu[frame_index],
                    scale=float(args.difference_scale),
                )
                comparison_video.append(comparison)
                difference_video.append(difference)
                if frame_index in valid_frames:
                    comparison.save(sample_dir / f"comparison_frame_{frame_index:03d}.png")
                    difference.save(sample_dir / f"difference_frame_{frame_index:03d}.png")

            if not args.no_videos:
                for filename, content in (
                    ("comparison.mp4", comparison_video),
                    ("differences.mp4", difference_video),
                ):
                    try:
                        _write_video(sample_dir / filename, content, args.fps)
                    except Exception as exc:
                        report["video_errors"].append(
                            {
                                "native_cache_position": int(native_position),
                                "file": filename,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

            report["samples"].append(
                {
                    "cache_position": int(native_position),
                    "dataset_index": int(selected_dataset_indices[ordinal]),
                    "record_uid": native_uid,
                    "shuffled_cache_position": shuffled_position,
                    "shuffled_record_uid": shuffled_uid,
                    "frames": frames,
                    "selected_frames": list(valid_frames),
                    "directory": str(sample_dir.relative_to(output)),
                    "latent_shuffle": row,
                }
            )
            print(
                "[latent shuffle] "
                f"z_native={native_position}:{native_uid} "
                f"z_shuffled={shuffled_position}:{shuffled_uid} "
                f"native_shuffle_mse={native_shuffled['mse']:.6f} "
                f"native_teacher_mse={native_teacher['mse']:.6f} "
                f"mse_ratio={mse_ratio:.3f}",
                flush=True,
            )

    report["latent_shuffle_summary"] = {
        "pairs": len(rows),
        "mean_native_shuffled_mse": float(
            sum(float(row["native_shuffled"]["mse"]) for row in rows) / len(rows)
        ),
        "mean_native_teacher_mse": float(
            sum(float(row["native_teacher"]["mse"]) for row in rows) / len(rows)
        ),
        "mean_mse_ratio": float(
            sum(float(row["native_shuffled_over_native_teacher_mse"]) for row in rows)
            / len(rows)
        ),
        "min_mse_ratio": float(
            min(float(row["native_shuffled_over_native_teacher_mse"]) for row in rows)
        ),
        "max_mse_ratio": float(
            max(float(row["native_shuffled_over_native_teacher_mse"]) for row in rows)
        ),
        "mean_mae_ratio": float(
            sum(float(row["native_shuffled_over_native_teacher_mae"]) for row in rows)
            / len(rows)
        ),
    }

    metadata = output / "metadata.json"
    metadata.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
