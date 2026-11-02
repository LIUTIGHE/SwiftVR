#!/usr/bin/env python3
"""Export fixed Stage-B1 Tiny Decoder validation comparisons from cached z_SR.

This is a visual-review utility only. It never loads the Stage-A DiT: the same
immutable z_SR cache used by formal Tiny Decoder validation is decoded by the
frozen ReAE decoder and by a selected TinyConditionalDecoder checkpoint.

An optional ``--shuffle-condition`` diagnostic keeps each cached z_SR fixed while
replacing its aligned LQ condition with another selected validation sample. This
measures how strongly the Tiny Decoder output causally depends on LQ semantics,
without changing model weights, caches, or the formal training path.
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
    parser.add_argument("--difference-scale", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument(
        "--shuffle-condition",
        action="store_true",
        help=(
            "Diagnostic only: keep each selected sample's cached z_SR fixed but "
            "replace its aligned LQ condition with another selected sample."
        ),
    )
    parser.add_argument(
        "--shuffle-offset",
        type=int,
        default=1,
        help=(
            "Cyclic offset inside --sample-indices used by --shuffle-condition; "
            "default 1 maps each sample to the next selected sample."
        ),
    )
    return parser


def _move_pixels(batch: dict[str, object], device: torch.device, dtype: torch.dtype):
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


def _difference_frame(
    tiny: torch.Tensor,
    teacher: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: float,
) -> Image.Image:
    return make_comparison_frame(
        OrderedDict(
            (
                (f"|Tiny-ReAE| x{scale:g}", (tiny - teacher).abs().mul(scale).clamp(0, 1)),
                (f"|Tiny-GT| x{scale:g}", (tiny - target).abs().mul(scale).clamp(0, 1)),
                (f"|ReAE-GT| x{scale:g}", (teacher - target).abs().mul(scale).clamp(0, 1)),
            )
        )
    )


def _shuffle_difference_frame(
    native: torch.Tensor,
    shuffled: torch.Tensor,
    teacher: torch.Tensor,
    *,
    scale: float,
) -> Image.Image:
    return make_comparison_frame(
        OrderedDict(
            (
                (
                    f"|Native-Shuffled| x{scale:g}",
                    (native - shuffled).abs().mul(scale).clamp(0, 1),
                ),
                (
                    f"|Native-ReAE| x{scale:g}",
                    (native - teacher).abs().mul(scale).clamp(0, 1),
                ),
                (
                    f"|Shuffled-ReAE| x{scale:g}",
                    (shuffled - teacher).abs().mul(scale).clamp(0, 1),
                ),
            )
        )
    )


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a_f = a.float().clamp(0.0, 1.0)
    b_f = b.float().clamp(0.0, 1.0)
    mse = float(F.mse_loss(a_f, b_f).item())
    mae = float(F.l1_loss(a_f, b_f).item())
    rmse = math.sqrt(max(mse, 0.0))
    psnr = float("inf") if mse <= 0.0 else -10.0 * math.log10(mse)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
    }


def _sample_uid(batch: dict[str, object], fallback: str) -> str:
    raw = batch.get("record_uid", [fallback])
    if isinstance(raw, (list, tuple)):
        return str(raw[0])
    return str(raw)


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_length <= 0 or args.crop_size <= 0 or args.scale <= 0:
        raise ValueError("clip-length, crop-size and scale must be positive")
    if args.clip_length % 4 != 1:
        raise ValueError("SwiftVR visual clips must satisfy T=4k+1")
    if args.difference_scale <= 0 or args.fps <= 0:
        raise ValueError("difference-scale and fps must be positive")
    if args.shuffle_condition:
        if len(args.sample_indices) < 2:
            raise ValueError("--shuffle-condition requires at least two --sample-indices")
        if args.shuffle_offset % len(args.sample_indices) == 0:
            raise ValueError(
                "--shuffle-offset maps every sample to itself; choose a non-zero "
                "offset modulo the number of selected samples"
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
        "base_checkpoint": str(base),
        "tiny_decoder": str(args.tiny_decoder.expanduser().resolve()),
        "val_cache": str(cache.root),
        "sample_positions": list(args.sample_indices),
        "frame_indices": list(args.frame_indices),
        "difference_scale": float(args.difference_scale),
        "shuffle_condition": bool(args.shuffle_condition),
        "shuffle_offset": int(args.shuffle_offset),
        "samples": [],
        "video_errors": [],
    }
    shuffle_rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for ordinal, (cache_position, batch_cpu) in enumerate(zip(args.sample_indices, loader)):
            moved = _move_pixels(dict(batch_cpu), device, dtype)
            prepared = prepare_training_batch(moved)
            lq_input = prepared["lq_input"]
            target = prepared["target"]
            if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise TypeError("visual batch is missing lq_input/target")
            z_sr = cache.load_batch(batch_cpu, device=device, dtype=dtype)

            shuffled_lq_input = None
            shuffled_position = None
            shuffled_uid = None
            if args.shuffle_condition:
                shuffled_ordinal = (ordinal + args.shuffle_offset) % len(selected_dataset_indices)
                shuffled_position = int(args.sample_indices[shuffled_ordinal])
                shuffled_dataset_index = selected_dataset_indices[shuffled_ordinal]
                wrong_sample = views[shuffled_dataset_index]
                wrong_batch_cpu = default_collate([wrong_sample])
                wrong_moved = _move_pixels(dict(wrong_batch_cpu), device, dtype)
                wrong_prepared = prepare_training_batch(wrong_moved)
                shuffled_lq_input = wrong_prepared["lq_input"]
                if not isinstance(shuffled_lq_input, torch.Tensor):
                    raise TypeError("shuffled validation sample is missing lq_input")
                shuffled_uid = _sample_uid(
                    dict(wrong_batch_cpu), f"sample_{shuffled_position:03d}"
                )
                if shuffled_lq_input.shape != lq_input.shape:
                    raise ValueError(
                        "Native/shuffled LQ geometry differs: "
                        f"native={tuple(lq_input.shape)}, "
                        f"shuffled={tuple(shuffled_lq_input.shape)}"
                    )

            with torch.autocast(
                device_type=device.type,
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                teacher = decode_reae_clip(
                    reae, z_sr, output_frames=int(target.shape[1]), clamp=True
                )
                prediction = tiny(
                    z_sr, lq_input, output_frames=int(target.shape[1]), clamp=True
                )
                shuffled_prediction = None
                if shuffled_lq_input is not None:
                    shuffled_prediction = tiny(
                        z_sr,
                        shuffled_lq_input,
                        output_frames=int(target.shape[1]),
                        clamp=True,
                    )

            lq = lq_input[0].float().cpu().clamp(0, 1)
            gt = target[0].float().cpu().clamp(0, 1)
            teacher_cpu = teacher[0].float().cpu().clamp(0, 1)
            tiny_cpu = prediction[0].float().cpu().clamp(0, 1)
            shuffled_lq_cpu = (
                None
                if shuffled_lq_input is None
                else shuffled_lq_input[0].float().cpu().clamp(0, 1)
            )
            shuffled_tiny_cpu = (
                None
                if shuffled_prediction is None
                else shuffled_prediction[0].float().cpu().clamp(0, 1)
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
            difference_video: list[Image.Image] = []
            shuffle_comparison_video: list[Image.Image] = []
            shuffle_difference_video: list[Image.Image] = []

            for frame_index in range(frames):
                comparison = make_comparison_frame(
                    OrderedDict(
                        (
                            ("LQ bicubic", lq[frame_index]),
                            ("GT", gt[frame_index]),
                            ("ReAE teacher", teacher_cpu[frame_index]),
                            ("Tiny decoder", tiny_cpu[frame_index]),
                        )
                    )
                )
                difference = _difference_frame(
                    tiny_cpu[frame_index],
                    teacher_cpu[frame_index],
                    gt[frame_index],
                    scale=float(args.difference_scale),
                )
                comparison_video.append(comparison)
                difference_video.append(difference)
                if frame_index in valid_frames:
                    comparison.save(sample_dir / f"comparison_frame_{frame_index:03d}.png")
                    difference.save(sample_dir / f"difference_frame_{frame_index:03d}.png")

                if shuffled_tiny_cpu is not None and shuffled_lq_cpu is not None:
                    shuffle_comparison = make_comparison_frame(
                        OrderedDict(
                            (
                                ("Native LQ", lq[frame_index]),
                                ("Shuffled LQ", shuffled_lq_cpu[frame_index]),
                                ("ReAE teacher", teacher_cpu[frame_index]),
                                ("Tiny native", tiny_cpu[frame_index]),
                                ("Tiny shuffled-LQ", shuffled_tiny_cpu[frame_index]),
                            )
                        )
                    )
                    shuffle_difference = _shuffle_difference_frame(
                        tiny_cpu[frame_index],
                        shuffled_tiny_cpu[frame_index],
                        teacher_cpu[frame_index],
                        scale=float(args.difference_scale),
                    )
                    shuffle_comparison_video.append(shuffle_comparison)
                    shuffle_difference_video.append(shuffle_difference)
                    if frame_index in valid_frames:
                        shuffle_comparison.save(
                            sample_dir / f"shuffle_comparison_frame_{frame_index:03d}.png"
                        )
                        shuffle_difference.save(
                            sample_dir / f"shuffle_difference_frame_{frame_index:03d}.png"
                        )

            if not args.no_videos:
                video_sets: list[tuple[str, list[Image.Image]]] = [
                    ("comparison.mp4", comparison_video),
                    ("differences.mp4", difference_video),
                ]
                if shuffle_comparison_video:
                    video_sets.extend(
                        [
                            ("shuffle_comparison.mp4", shuffle_comparison_video),
                            ("shuffle_differences.mp4", shuffle_difference_video),
                        ]
                    )
                for filename, content in video_sets:
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

            sample_report: dict[str, object] = {
                "cache_position": int(cache_position),
                "dataset_index": int(selected_dataset_indices[ordinal]),
                "record_uid": uid,
                "frames": frames,
                "selected_frames": list(valid_frames),
                "directory": str(sample_dir.relative_to(output)),
            }

            if shuffled_prediction is not None:
                native_shuffled = _tensor_metrics(prediction, shuffled_prediction)
                native_teacher = _tensor_metrics(prediction, teacher)
                shuffled_teacher = _tensor_metrics(shuffled_prediction, teacher)
                mse_ratio = native_shuffled["mse"] / max(native_teacher["mse"], 1e-12)
                mae_ratio = native_shuffled["mae"] / max(native_teacher["mae"], 1e-12)
                shuffle_row: dict[str, object] = {
                    "native_cache_position": int(cache_position),
                    "native_record_uid": uid,
                    "shuffled_cache_position": int(shuffled_position),
                    "shuffled_record_uid": str(shuffled_uid),
                    "native_shuffled": native_shuffled,
                    "native_teacher": native_teacher,
                    "shuffled_teacher": shuffled_teacher,
                    "native_shuffled_over_native_teacher_mse": float(mse_ratio),
                    "native_shuffled_over_native_teacher_mae": float(mae_ratio),
                }
                shuffle_rows.append(shuffle_row)
                sample_report["condition_shuffle"] = shuffle_row
                print(
                    "[condition shuffle] "
                    f"native={cache_position}:{uid} "
                    f"shuffled={shuffled_position}:{shuffled_uid} "
                    f"native_shuffle_mse={native_shuffled['mse']:.8f} "
                    f"native_teacher_mse={native_teacher['mse']:.8f} "
                    f"mse_ratio={mse_ratio:.4f}",
                    flush=True,
                )

            report["samples"].append(sample_report)
            print(
                f"visualized cache sample {cache_position}: {uid} -> {sample_dir}",
                flush=True,
            )

    if shuffle_rows:
        mse_ratios = [
            float(row["native_shuffled_over_native_teacher_mse"])
            for row in shuffle_rows
        ]
        mae_ratios = [
            float(row["native_shuffled_over_native_teacher_mae"])
            for row in shuffle_rows
        ]
        native_shuffled_mse = [
            float(row["native_shuffled"]["mse"])  # type: ignore[index]
            for row in shuffle_rows
        ]
        native_teacher_mse = [
            float(row["native_teacher"]["mse"])  # type: ignore[index]
            for row in shuffle_rows
        ]
        report["condition_shuffle_summary"] = {
            "pairs": len(shuffle_rows),
            "mean_native_shuffled_mse": float(sum(native_shuffled_mse) / len(shuffle_rows)),
            "mean_native_teacher_mse": float(sum(native_teacher_mse) / len(shuffle_rows)),
            "mean_mse_ratio": float(sum(mse_ratios) / len(mse_ratios)),
            "min_mse_ratio": float(min(mse_ratios)),
            "max_mse_ratio": float(max(mse_ratios)),
            "mean_mae_ratio": float(sum(mae_ratios) / len(mae_ratios)),
        }

    (output / "metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
