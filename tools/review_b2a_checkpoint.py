#!/usr/bin/env python3
"""Standalone visual/metric review for a saved B2-A compact checkpoint.

This tool is intentionally evaluation-only: it creates no optimizer, performs no
training step, and does not require torchrun/DDP.  It reuses the exact B2-A
validation path and exports more fixed validation examples than the trainer's
small default visual sample count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_teacher_distillation_ddp as stage_a
from smoke_training_forward import resolve_runtime_dtype, validate_folded_checkpoint
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training import TeacherVelocityCache
from swiftvr.training.b2a_width import (
    B2ACompactVelocityDistillationForward,
    transformer_width_shape,
)
from swiftvr.training.distillation_visuals import export_validation_visuals
from swiftvr.training.perceptual_review import parse_csv_ints
from train_b2a_compact_distill_ddp import validate_rank0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--student-checkpoint", type=Path, required=True)
    p.add_argument("--val-teacher-cache", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, action="append", required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--val-split", default="val")
    p.add_argument("--clip-length", type=int, default=13)
    p.add_argument("--val-crop-size", type=int, default=128)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--val-views-per-record", type=int, default=1)
    p.add_argument("--val-view-seed", type=int, default=9000001)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    p.add_argument("--allow-dtype-mismatch", action="store_true")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument(
        "--visual-validation-samples",
        type=int,
        default=13,
        help="Number of ordered fixed-validation samples to export; 13 exports all val13 samples.",
    )
    p.add_argument(
        "--visual-frame-indices",
        default="0,3,6,9,12",
        help="Comma-separated frame indices saved as PNG comparisons.",
    )
    p.add_argument("--visual-video-fps", type=float, default=8.0)
    p.add_argument("--visual-difference-scale", type=float, default=4.0)
    p.add_argument(
        "--no-videos",
        action="store_true",
        help="Write PNGs/metrics only; useful if ffmpeg/video export is not needed.",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="Visual folder step label; defaults to checkpoint metadata global_step when available.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    return p


def _infer_step(checkpoint: Path, explicit: int | None) -> int:
    if explicit is not None:
        if explicit < 0:
            raise ValueError("--step must be non-negative")
        return int(explicit)
    metadata_path = checkpoint / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata, dict):
            for key in ("global_step", "step"):
                if key in metadata:
                    value = int(metadata[key])
                    if value >= 0:
                        return value
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_length <= 0 or args.val_crop_size <= 0 or args.scale <= 0:
        raise ValueError("clip-length/val-crop-size/scale must be positive")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.val_views_per_record <= 0 or args.visual_validation_samples <= 0:
        raise ValueError("val views / visual sample count must be positive")
    if args.visual_video_fps <= 0 or args.visual_difference_scale <= 0:
        raise ValueError("visual FPS/difference scale must be positive")
    frame_indices = parse_csv_ints(args.visual_frame_indices)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    base_root = args.base_checkpoint.expanduser().resolve()
    student_root = args.student_checkpoint.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Review output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    folded_config = validate_folded_checkpoint(
        base_root,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
    )
    dtype = resolve_runtime_dtype(
        args.dtype,
        folded_config,
        device,
        allow_mismatch=args.allow_dtype_mismatch,
    )
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    cache = TeacherVelocityCache(args.val_teacher_cache)
    if cache.metadata.get("kind") != "swiftvr_b2a_stage_a_teacher_velocity":
        raise ValueError("Validation cache is not a B2-A Stage-A teacher cache")
    dataset = stage_a.build_cached_dataset(
        args.val_manifest,
        cache,
        split=args.val_split,
        path_root=args.path_root,
        clip_length=args.clip_length,
        crop_size=args.val_crop_size,
        scale=args.scale,
        views_per_record=args.val_views_per_record,
        view_seed=args.val_view_seed,
        hflip=0.0,
        vflip=0.0,
        verify_paths=args.verify_paths,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    reae = ReAE(str(base_root / args.reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(student_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    shape = transformer_width_shape(transformer)
    reae.to(device=device, dtype=dtype).eval()
    transformer.to(device=device, dtype=dtype).eval()
    closure = B2ACompactVelocityDistillationForward(
        reae,
        transformer,
        attention_backend=args.attention_backend,
        gradient_checkpointing=False,
    )
    closure.eval()
    closure.reae.eval()

    requested_visuals = min(args.visual_validation_samples, len(dataset))
    with torch.inference_mode():
        metrics, visual_samples = validate_rank0(
            closure,
            loader,
            cache,
            device=device,
            dtype=dtype,
            visual_samples=requested_visuals,
        )

    step = _infer_step(student_root, args.step)
    report = export_validation_visuals(
        visual_samples,
        output_root=output_root,
        step=step,
        frame_indices=frame_indices,
        fps=args.visual_video_fps,
        difference_scale=args.visual_difference_scale,
        writer=None,
        write_videos=not args.no_videos,
    )

    summary = {
        "kind": "b2a_checkpoint_visual_review",
        "student_checkpoint": str(student_root),
        "base_checkpoint": str(base_root),
        "val_teacher_cache": str(args.val_teacher_cache.expanduser().resolve()),
        "val_manifests": [str(path.expanduser().resolve()) for path in args.val_manifest],
        "step": step,
        "runtime_dtype": str(dtype).removeprefix("torch."),
        "student_shape": shape,
        "validation_samples": len(dataset),
        "exported_visual_samples": len(visual_samples),
        "frame_indices": list(frame_indices),
        "metrics": metrics,
        "visual_report": report,
    }
    (output_root / "review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\n========== B2-A checkpoint review ==========")
    print(f"Checkpoint             : {student_root}")
    print(f"Step label             : {step}")
    print(f"Student shape          : {shape}")
    print(f"Validation samples     : {len(dataset)}")
    print(f"Exported visuals       : {len(visual_samples)}")
    print(f"Velocity rel-L2        : {float(metrics['velocity_relative_l2']):.6f}")
    print(f"Velocity cosine        : {float(metrics['velocity_cosine']):.6f}")
    print(f"Latent rel-L2          : {float(metrics['restored_latent_relative_l2']):.6f}")
    print(f"Student/Teacher PSNR   : {float(metrics['student_teacher_psnr']):.4f}")
    print(f"Student/GT PSNR        : {float(metrics['student_gt_psnr']):.4f}")
    print(f"Teacher/GT PSNR        : {float(metrics['teacher_gt_psnr']):.4f}")
    print(f"Review output          : {output_root}")
    if report["video_errors"]:
        print("Video export warnings  : " + json.dumps(report["video_errors"]))
    print("============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
