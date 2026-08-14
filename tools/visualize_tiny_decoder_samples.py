#!/usr/bin/env python3
"""Export fixed Stage-B1 Tiny Decoder validation comparisons from cached z_SR.

This is a visual-review utility only. It never loads the Stage-A DiT: the same
immutable z_SR cache used by formal Tiny Decoder validation is decoded by the
frozen ReAE decoder and by a selected TinyConditionalDecoder checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

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


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_length <= 0 or args.crop_size <= 0 or args.scale <= 0:
        raise ValueError("clip-length, crop-size and scale must be positive")
    if args.clip_length % 4 != 1:
        raise ValueError("SwiftVR visual clips must satisfy T=4k+1")
    if args.difference_scale <= 0 or args.fps <= 0:
        raise ValueError("difference-scale and fps must be positive")

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
        "samples": [],
        "video_errors": [],
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
                prediction = tiny(
                    z_sr, lq_input, output_frames=int(target.shape[1]), clamp=True
                )

            lq = lq_input[0].float().cpu().clamp(0, 1)
            gt = target[0].float().cpu().clamp(0, 1)
            teacher_cpu = teacher[0].float().cpu().clamp(0, 1)
            tiny_cpu = prediction[0].float().cpu().clamp(0, 1)
            frames = int(gt.shape[0])
            valid_frames = tuple(index for index in args.frame_indices if index < frames)
            if not valid_frames:
                raise ValueError(f"No requested frame lies inside {frames}-frame clip")

            uid_raw = batch_cpu.get("record_uid", [f"sample_{cache_position:03d}"])
            uid = str(uid_raw[0] if isinstance(uid_raw, (list, tuple)) else uid_raw)
            safe_uid = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in uid)
            sample_dir = output / f"sample_{cache_position:03d}_{safe_uid}"
            sample_dir.mkdir(parents=True, exist_ok=False)
            comparison_video: list[Image.Image] = []
            difference_video: list[Image.Image] = []

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
                                "sample_position": int(cache_position),
                                "file": filename,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

            report["samples"].append(
                {
                    "cache_position": int(cache_position),
                    "dataset_index": int(selected_dataset_indices[ordinal]),
                    "record_uid": uid,
                    "frames": frames,
                    "selected_frames": list(valid_frames),
                    "directory": str(sample_dir.relative_to(output)),
                }
            )
            print(
                f"visualized cache sample {cache_position}: {uid} -> {sample_dir}",
                flush=True,
            )

    (output / "metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
