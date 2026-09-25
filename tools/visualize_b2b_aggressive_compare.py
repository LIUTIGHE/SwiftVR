#!/usr/bin/env python3
"""Visual comparison for B2B aggressive staged vs joint checkpoints.

The deployment target is teacher-like generative restoration.  This tool therefore
puts the frozen Stage-A teacher next to the staged and joint B2B students, while GT
is shown as a reference rather than treated as the visual target.

For each deterministic val sample it writes:
  * one side-by-side MP4: Teacher | Staged | Joint | GT
  * first/middle/last comparison PNGs
  * per-sample Teacher/GT and Student/Teacher + Student/GT metrics
It also writes a middle-frame overview across all exported samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_teacher_distillation_ddp as stage_a
from tools.smoke_training_forward import move_video_batch
from swiftvr.models import ReAE
from swiftvr.models.reae_slim_decoder import AGGRESSIVE_CHANNELS, SlimReAEDecoder
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training import TeacherVelocityCache, VideoMetricAccumulator, decode_teacher_prediction
from swiftvr.training.b2a_width import transformer_width_shape
from swiftvr.training.b2b_joint import B2B_TINY_SPEC, B2BJointForward


DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--staged-checkpoint", type=Path, required=True)
    p.add_argument("--joint-checkpoint", type=Path, required=True)
    p.add_argument("--teacher-cache", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, action="append", required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--val-split", default="val")
    p.add_argument("--clip-length", type=int, default=13)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--views-per-record", type=int, default=1)
    p.add_argument("--view-seed", type=int, default=9000001)
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--max-samples", type=int, default=13)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    return p


def _expected_student_shape() -> dict[str, int]:
    return {
        "hidden_dim": B2B_TINY_SPEC.hidden_dim,
        "num_heads": B2B_TINY_SPEC.num_heads,
        "head_dim": B2B_TINY_SPEC.head_dim,
        "ffn_dim": B2B_TINY_SPEC.ffn_dim,
        "num_layers": B2B_TINY_SPEC.num_layers,
        "adapter_dim": B2B_TINY_SPEC.adapter_dim,
    }


def _load_student(
    checkpoint: Path,
    *,
    reae: ReAE,
    device: torch.device,
    dtype: torch.dtype,
    transformer_subfolder: str,
    attention_backend: str,
) -> B2BJointForward:
    root = checkpoint.expanduser().resolve()
    transformer_dir = root / transformer_subfolder
    decoder_dir = root / "tiny_decoder"
    if not transformer_dir.is_dir():
        raise FileNotFoundError(f"Missing transformer directory: {transformer_dir}")
    if not decoder_dir.is_dir():
        raise FileNotFoundError(f"Missing decoder directory: {decoder_dir}")

    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(root),
        subfolder=transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype)
    actual_shape = transformer_width_shape(transformer)
    expected_shape = _expected_student_shape()
    if actual_shape != expected_shape:
        raise ValueError(f"B2B DiT shape mismatch: {actual_shape} != {expected_shape}")

    decoder = SlimReAEDecoder.from_pretrained(decoder_dir, device=device, dtype=dtype)
    if tuple(decoder.channels) != tuple(AGGRESSIVE_CHANNELS):
        raise ValueError(
            f"Expected aggressive decoder {tuple(AGGRESSIVE_CHANNELS)}, "
            f"got {tuple(decoder.channels)} from {decoder_dir}"
        )

    model = B2BJointForward(
        reae,
        transformer,
        decoder,
        attention_backend=attention_backend,
        gradient_checkpointing=False,
    ).to(device=device)
    model.eval()
    return model


def _video_uint8(video: torch.Tensor) -> np.ndarray:
    if video.ndim != 5 or int(video.shape[0]) != 1 or int(video.shape[2]) != 3:
        raise ValueError(f"Expected [1,T,3,H,W] video, got {tuple(video.shape)}")
    value = video.detach().float().clamp(0.0, 1.0)[0]
    value = value.permute(0, 2, 3, 1).cpu().numpy()
    return np.rint(value * 255.0).astype(np.uint8)


def _labeled_panel(rgb: np.ndarray, label: str, *, top: int = 34) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    panel = cv2.copyMakeBorder(bgr, top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(
        panel,
        label,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _comparison_frame(
    teacher: np.ndarray,
    staged: np.ndarray,
    joint: np.ndarray,
    gt: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    panels = [
        _labeled_panel(teacher[frame_index], "Teacher"),
        _labeled_panel(staged[frame_index], "Staged D768"),
        _labeled_panel(joint[frame_index], "Joint D768+Dec"),
        _labeled_panel(gt[frame_index], "GT (reference)"),
    ]
    heights = {panel.shape[0] for panel in panels}
    if len(heights) != 1:
        raise ValueError(f"Comparison streams have different heights: {sorted(heights)}")
    return cv2.hconcat(panels)


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    accumulator = VideoMetricAccumulator()
    accumulator.update(prediction, target, clamp=True)
    return accumulator.compute()


def _sample_name(batch: dict[str, object], index: int) -> str:
    value = batch.get("sample_id")
    if isinstance(value, (list, tuple)) and value:
        raw = str(value[0])
    elif value is not None:
        raw = str(value)
    else:
        raw = f"sample_{index:02d}"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw)
    return f"{index:02d}_{safe}"


def main() -> int:
    args = build_parser().parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = TeacherVelocityCache(args.teacher_cache)
    if cache.metadata.get("kind") != "swiftvr_b2a_stage_a_teacher_velocity":
        raise ValueError("Expected Stage-A teacher velocity cache")
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
    reae = ReAE(str(base_root / args.reae_filename)).to(device=device, dtype=dtype).eval()
    staged = _load_student(
        args.staged_checkpoint,
        reae=reae,
        device=device,
        dtype=dtype,
        transformer_subfolder=args.transformer_subfolder,
        attention_backend=args.attention_backend,
    )
    joint = _load_student(
        args.joint_checkpoint,
        reae=reae,
        device=device,
        dtype=dtype,
        transformer_subfolder=args.transformer_subfolder,
        attention_backend=args.attention_backend,
    )

    autocast_enabled = device.type == "cuda" and dtype == torch.bfloat16
    all_metrics: list[dict[str, object]] = []
    overview_middle: list[np.ndarray] = []

    with torch.no_grad():
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
                staged_out = staged(batch)
                joint_out = joint(batch)
                teacher_prediction = decode_teacher_prediction(
                    reae=reae,
                    z_lq=staged_out["z_lq"],
                    teacher_velocity=teacher_velocity,
                    output_frames=int(staged_out["target"].shape[1]),
                )

            target = staged_out["target"]
            if not torch.equal(staged_out["target"], joint_out["target"]):
                raise RuntimeError("Staged/joint batches produced different GT tensors")

            sample_metrics = {
                "index": index,
                "sample": _sample_name(batch_cpu, index),
                "teacher_gt": _metrics(teacher_prediction, target),
                "staged_teacher": _metrics(staged_out["prediction"], teacher_prediction),
                "staged_gt": _metrics(staged_out["prediction"], target),
                "joint_teacher": _metrics(joint_out["prediction"], teacher_prediction),
                "joint_gt": _metrics(joint_out["prediction"], target),
            }
            all_metrics.append(sample_metrics)

            teacher_u8 = _video_uint8(teacher_prediction)
            staged_u8 = _video_uint8(staged_out["prediction"])
            joint_u8 = _video_uint8(joint_out["prediction"])
            gt_u8 = _video_uint8(target)
            frames = int(teacher_u8.shape[0])
            if not (staged_u8.shape[0] == joint_u8.shape[0] == gt_u8.shape[0] == frames):
                raise RuntimeError("Comparison videos have different frame counts")

            sample_dir = output_dir / str(sample_metrics["sample"])
            sample_dir.mkdir(parents=True, exist_ok=True)
            first_frame = _comparison_frame(teacher_u8, staged_u8, joint_u8, gt_u8, 0)
            height, width = first_frame.shape[:2]
            video_path = sample_dir / "comparison.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps),
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                writer = None
                print(f"WARNING: cannot open MP4 writer for {video_path}; PNGs will still be saved")

            key_indices = sorted({0, frames // 2, frames - 1})
            for frame_index in range(frames):
                comparison = (
                    first_frame
                    if frame_index == 0
                    else _comparison_frame(teacher_u8, staged_u8, joint_u8, gt_u8, frame_index)
                )
                if writer is not None:
                    writer.write(comparison)
                if frame_index in key_indices:
                    cv2.imwrite(str(sample_dir / f"frame_{frame_index:03d}.png"), comparison)
                if frame_index == frames // 2:
                    overview_middle.append(comparison)
            if writer is not None:
                writer.release()

            (sample_dir / "metrics.json").write_text(
                json.dumps(sample_metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(
                f"[{index + 1}/{min(len(dataset), args.max_samples)}] {sample_metrics['sample']} "
                f"staged->T={sample_metrics['staged_teacher']['psnr']:.3f} "
                f"joint->T={sample_metrics['joint_teacher']['psnr']:.3f} "
                f"staged->GT={sample_metrics['staged_gt']['psnr']:.3f} "
                f"joint->GT={sample_metrics['joint_gt']['psnr']:.3f}",
                flush=True,
            )

    if not all_metrics:
        raise RuntimeError("No validation samples were exported")
    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    if overview_middle:
        cv2.imwrite(str(output_dir / "overview_middle_frames.png"), cv2.vconcat(overview_middle))

    print("================ B2B visual comparison complete ================")
    print(f"Samples : {len(all_metrics)}")
    print(f"Output  : {output_dir}")
    print("Columns : Teacher | Staged D768 | Joint D768+Dec | GT reference")
    print("================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
