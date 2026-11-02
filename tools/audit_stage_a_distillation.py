#!/usr/bin/env python3
"""Audit Stage-A SwiftVR distillation checkpoints on a fixed teacher cache.

The audit deliberately reuses the deterministic validation view/cache contract used
by teacher-distillation training.  It compares any number of prompt-free/no-time
student delta checkpoints against the cached conditional teacher and GT, measures
component/end-to-end latency, optionally records operator-reported FLOPs, and writes
fixed multi-model visual comparisons plus JSON/Markdown reports.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Mapping

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

try:
    import train_teacher_distillation_ddp as gate
except ModuleNotFoundError:
    from tools import train_teacher_distillation_ddp as gate

from swiftvr.training import load_delta_checkpoint
from swiftvr.training.distillation import (
    DeterministicTripletViewDataset,
    DistillationMetricAccumulator,
    SwiftVRVelocityDistillationForward,
    TeacherVelocityCache,
    decode_student_prediction,
    decode_teacher_prediction,
    distillation_sample_identity,
    gt_reconstruction_constraint,
)
from swiftvr.training.distillation_generalization import build_cache_backed_subset
from swiftvr.training.forward import (
    encode_reae_clip,
    forward_prompt_free_no_time_training,
    prepare_training_batch,
)
from swiftvr.training.perceptual_review import make_comparison_frame
from swiftvr.training.stage3 import VideoMetricAccumulator, temporal_difference_mse


DTYPE_NAMES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=CHECKPOINT",
        help=(
            "Student to audit. CHECKPOINT is a delta-checkpoint directory or the "
            "literal 'base' to evaluate --base-checkpoint before distillation."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--views-per-record", type=int, default=1)
    parser.add_argument("--view-seed", type=int, default=9000001)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--allow-dtype-mismatch", action="store_true")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--visual-samples", type=int, default=2)
    parser.add_argument("--visual-frame-indices", default="0,6,12")
    parser.add_argument("--visual-video-fps", type=float, default=8.0)
    parser.add_argument("--difference-scale", type=float, default=4.0)
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=10)
    parser.add_argument(
        "--profile-flops",
        action="store_true",
        help="Use torch.utils.flop_counter when available; unsupported ops may be absent.",
    )
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="Also compute LPIPS if the optional 'lpips' Python package is installed.",
    )
    return parser


def parse_model_specs(values: list[str]) -> list[tuple[str, Path | None]]:
    result: list[tuple[str, Path | None]] = []
    seen: set[str] = set()
    for raw in values:
        label, separator, checkpoint = raw.partition("=")
        label = label.strip()
        checkpoint = checkpoint.strip()
        if not separator or not label or not checkpoint:
            raise ValueError(
                f"Invalid --model {raw!r}; expected LABEL=CHECKPOINT or LABEL=base"
            )
        if label in seen:
            raise ValueError(f"Duplicate model label: {label!r}")
        seen.add(label)
        path = None if checkpoint.lower() == "base" else Path(checkpoint).expanduser().resolve()
        result.append((label, path))
    return result


def parse_frame_indices(value: str) -> tuple[int, ...]:
    parsed = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not parsed or any(index < 0 for index in parsed):
        raise ValueError("--visual-frame-indices must contain non-negative integers")
    return parsed


def parameter_summary(module: torch.nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
    }


def build_validation_dataset(args: argparse.Namespace, cache: TeacherVelocityCache):
    base = gate.TripletVideoDataset(
        args.manifest,
        split=args.split,
        training=True,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        load_hq=False,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        drop_short_sequences=True,
        path_root=args.path_root.expanduser().resolve(),
        verify_paths=args.verify_paths,
    )
    views = DeterministicTripletViewDataset(
        base,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
    )
    cache.validate_dataset(
        manifests=args.manifest,
        split=args.split,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        dataset_length=len(views),
    )
    return build_cache_backed_subset(views, cache)


def build_loader(dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=bool(args.num_workers > 0),
    )


def load_student(
    args: argparse.Namespace,
    *,
    base_checkpoint: Path,
    delta_checkpoint: Path | None,
    device: torch.device,
    dtype: torch.dtype,
) -> SwiftVRVelocityDistillationForward:
    reae = gate.ReAE(str(base_checkpoint / args.reae_filename))
    transformer = gate.WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(base_checkpoint),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    gate.configure_train_scope(reae, transformer, "adapter")
    reae.to(device=device, dtype=dtype).eval()
    transformer.to(device=device, dtype=dtype)
    closure = SwiftVRVelocityDistillationForward(
        reae,
        transformer,
        attention_backend=args.attention_backend,
    )
    gate.cast_trainable_parameters(closure, dtype=torch.float32)
    if delta_checkpoint is not None:
        load_delta_checkpoint(delta_checkpoint, closure, strict=True)
    closure.eval()
    closure.reae.eval()
    return closure


def _batch_text(batch: Mapping[str, object], key: str, index: int) -> str:
    value = batch.get(key)
    if isinstance(value, (list, tuple)):
        return str(value[index])
    if isinstance(value, torch.Tensor):
        item = value[index]
        return str(item.item() if item.ndim == 0 else item.tolist())
    return str(value)


def _load_lpips(device: torch.device):
    try:
        import lpips  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "--lpips requested but the optional 'lpips' package is unavailable"
        ) from exc
    model = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _lpips_sum(model, prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    prediction = prediction.clamp(0, 1)
    target = target.clamp(0, 1)
    flat_prediction = prediction.flatten(0, 1).mul(2).sub(1)
    flat_target = target.flatten(0, 1).mul(2).sub(1)
    values = model(flat_prediction, flat_target).reshape(-1)
    return float(values.float().sum().item()), int(values.numel())


def evaluate_model(
    closure: SwiftVRVelocityDistillationForward,
    loader: DataLoader,
    cache: TeacherVelocityCache,
    *,
    device: torch.device,
    dtype: torch.dtype,
    lpips_model,
    visual_limit: int,
) -> tuple[dict[str, float | int], dict[str, dict[str, object]]]:
    velocity = DistillationMetricAccumulator()
    student_teacher = VideoMetricAccumulator()
    student_gt = VideoMetricAccumulator()
    teacher_gt = VideoMetricAccumulator()
    student_temporal_sum = 0.0
    teacher_temporal_sum = 0.0
    pixel_violations = 0.0
    temporal_violations = 0.0
    gt_batches = 0
    lpips_st_sum = lpips_sg_sum = lpips_tg_sum = 0.0
    lpips_frames = 0
    visuals: dict[str, dict[str, object]] = {}
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)

    closure.eval()
    with torch.no_grad():
        for batch_cpu in loader:
            teacher_velocity = cache.load_batch(batch_cpu, device=device, dtype=dtype)
            batch = gate.move_video_batch(batch_cpu, device=device, dtype=dtype)
            with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                output = closure(batch)
                output_frames = int(output["target"].shape[1])
                student_prediction = decode_student_prediction(
                    reae=closure.reae,
                    z_lq=output["z_lq"],
                    student_velocity=output["velocity"],
                    output_frames=output_frames,
                )
                teacher_prediction = decode_teacher_prediction(
                    reae=closure.reae,
                    z_lq=output["z_lq"],
                    teacher_velocity=teacher_velocity,
                    output_frames=output_frames,
                )

            velocity.update(output["velocity"], teacher_velocity)
            student_teacher.update(student_prediction, teacher_prediction, clamp=True)
            student_gt.update(student_prediction, output["target"], clamp=True)
            teacher_gt.update(teacher_prediction, output["target"], clamp=True)
            student_temporal_sum += float(
                temporal_difference_mse(student_prediction.float(), output["target"].float()).item()
            )
            teacher_temporal_sum += float(
                temporal_difference_mse(teacher_prediction.float(), output["target"].float()).item()
            )
            guard = gt_reconstruction_constraint(
                student_prediction,
                teacher_prediction,
                output["target"],
                mode="guard",
            )
            pixel_violations += float(guard["gt_pixel_violation_rate"].item())
            temporal_violations += float(guard["gt_temporal_violation_rate"].item())
            gt_batches += 1

            if lpips_model is not None:
                value, count = _lpips_sum(lpips_model, student_prediction, teacher_prediction)
                lpips_st_sum += value
                value, _ = _lpips_sum(lpips_model, student_prediction, output["target"])
                lpips_sg_sum += value
                value, _ = _lpips_sum(lpips_model, teacher_prediction, output["target"])
                lpips_tg_sum += value
                lpips_frames += count

            batch_size = int(student_prediction.shape[0])
            for local_index in range(batch_size):
                if len(visuals) >= visual_limit:
                    break
                identity = distillation_sample_identity(batch_cpu, local_index)
                key = str(identity["key"])
                visuals[key] = {
                    "record_uid": _batch_text(batch_cpu, "record_uid", local_index),
                    "lq_input": output["lq_input"][local_index].clamp(0, 1).float().cpu(),
                    "target": output["target"][local_index].clamp(0, 1).float().cpu(),
                    "teacher": teacher_prediction[local_index].clamp(0, 1).float().cpu(),
                    "student": student_prediction[local_index].clamp(0, 1).float().cpu(),
                }

    result: dict[str, float | int] = {**velocity.compute()}
    result.update({f"student_teacher_{key}": value for key, value in student_teacher.compute().items()})
    result.update({f"student_gt_{key}": value for key, value in student_gt.compute().items()})
    result.update({f"teacher_gt_{key}": value for key, value in teacher_gt.compute().items()})
    result["student_gt_temporal_difference_mse"] = student_temporal_sum / max(gt_batches, 1)
    result["teacher_gt_temporal_difference_mse"] = teacher_temporal_sum / max(gt_batches, 1)
    result["gt_pixel_violation_rate"] = pixel_violations / max(gt_batches, 1)
    result["gt_temporal_violation_rate"] = temporal_violations / max(gt_batches, 1)
    if lpips_model is not None:
        result["student_teacher_lpips"] = lpips_st_sum / max(lpips_frames, 1)
        result["student_gt_lpips"] = lpips_sg_sum / max(lpips_frames, 1)
        result["teacher_gt_lpips"] = lpips_tg_sum / max(lpips_frames, 1)
    return result, visuals


def _benchmark_cuda(fn: Callable[[], object], *, warmup: int, repeats: int) -> dict[str, float]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("latency warmup must be non-negative and repeats positive")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _count_flops(fn: Callable[[], object]) -> tuple[int | None, str | None]:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as exc:
        return None, f"FlopCounterMode unavailable: {type(exc).__name__}: {exc}"
    try:
        with FlopCounterMode(display=False) as counter:
            fn()
        return int(counter.get_total_flops()), None
    except Exception as exc:
        return None, f"FLOP counting failed: {type(exc).__name__}: {exc}"


def profile_model(
    closure: SwiftVRVelocityDistillationForward,
    batch_cpu: Mapping[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, object]:
    batch = gate.move_video_batch(batch_cpu, device=device, dtype=dtype)
    prepared = prepare_training_batch(batch)
    lq_input = prepared["lq_input"]
    target = prepared["target"]
    if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("Profiling batch is missing lq_input/target")
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        z_ntchw = encode_reae_clip(closure.reae, lq_input, require_4k_plus_1=True)
        z_lq = z_ntchw.permute(0, 2, 1, 3, 4).contiguous()
        velocity = forward_prompt_free_no_time_training(closure.transformer, z_lq)

    def encode_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            return encode_reae_clip(closure.reae, lq_input, require_4k_plus_1=True)

    def transformer_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            return forward_prompt_free_no_time_training(closure.transformer, z_lq)

    def decode_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            return decode_student_prediction(
                reae=closure.reae,
                z_lq=z_lq,
                student_velocity=velocity,
                output_frames=int(target.shape[1]),
            )

    def end_to_end_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            output = closure(batch)
            return decode_student_prediction(
                reae=closure.reae,
                z_lq=output["z_lq"],
                student_velocity=output["velocity"],
                output_frames=int(output["target"].shape[1]),
            )

    profile: dict[str, object] = {
        "input_shape": list(lq_input.shape),
        "latent_shape": list(z_lq.shape),
        "encoder_latency": _benchmark_cuda(
            encode_fn, warmup=args.latency_warmup, repeats=args.latency_repeats
        ),
        "transformer_latency": _benchmark_cuda(
            transformer_fn, warmup=args.latency_warmup, repeats=args.latency_repeats
        ),
        "decoder_latency": _benchmark_cuda(
            decode_fn, warmup=args.latency_warmup, repeats=args.latency_repeats
        ),
        "end_to_end_latency": _benchmark_cuda(
            end_to_end_fn, warmup=args.latency_warmup, repeats=args.latency_repeats
        ),
    }
    median_ms = float(profile["end_to_end_latency"]["median_ms"])  # type: ignore[index]
    frames = int(target.shape[1])
    profile["effective_fps"] = frames * 1000.0 / max(median_ms, 1e-9)

    torch.cuda.reset_peak_memory_stats(device)
    end_to_end_fn()
    torch.cuda.synchronize()
    profile["peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3

    if args.profile_flops:
        flops: dict[str, object] = {}
        for name, fn in (
            ("encoder", encode_fn),
            ("transformer", transformer_fn),
            ("decoder", decode_fn),
            ("end_to_end", end_to_end_fn),
        ):
            count, error = _count_flops(fn)
            flops[name] = {"reported_flops": count, "error": error}
        flops["note"] = (
            "Operator-reported FLOPs only; unsupported/custom/fused kernels can be absent. "
            "Use latency as the deployment-facing source of truth."
        )
        profile["flops"] = flops
    return profile


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _write_video(path: Path, frames: list[Image.Image], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path), fps=float(fps), codec="libx264", macro_block_size=1, quality=8
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))


def export_visuals(
    visual_bank: dict[str, dict[str, object]],
    model_order: list[str],
    *,
    output_dir: Path,
    frame_indices: tuple[int, ...],
    fps: float,
    difference_scale: float,
) -> dict[str, object]:
    root = output_dir / "visuals"
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"samples": [], "video_errors": []}
    for sample_index, (key, sample) in enumerate(visual_bank.items()):
        sample_dir = root / f"sample_{sample_index:03d}_{_safe_name(str(sample['record_uid']))}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        target = sample["target"]
        teacher = sample["teacher"]
        lq = sample["lq_input"]
        if not all(isinstance(value, torch.Tensor) for value in (target, teacher, lq)):
            raise TypeError("Visual bank tensors are invalid")
        frames = int(target.shape[0])
        valid = tuple(index for index in frame_indices if index < frames)
        comparison_video: list[Image.Image] = []
        difference_video: list[Image.Image] = []
        for frame_index in range(frames):
            panels = OrderedDict(
                (("LQ bicubic", lq[frame_index]), ("GT", target[frame_index]), ("Teacher", teacher[frame_index]))
            )
            differences = OrderedDict()
            for label in model_order:
                student = sample["students"][label]
                panels[label] = student[frame_index]
                differences[f"|{label}-Teacher| x{difference_scale:g}"] = (
                    student[frame_index] - teacher[frame_index]
                ).abs().mul(difference_scale).clamp(0, 1)
                differences[f"|{label}-GT| x{difference_scale:g}"] = (
                    student[frame_index] - target[frame_index]
                ).abs().mul(difference_scale).clamp(0, 1)
            comparison = make_comparison_frame(panels)
            difference = make_comparison_frame(differences)
            comparison_video.append(comparison)
            difference_video.append(difference)
            if frame_index in valid:
                comparison.save(sample_dir / f"comparison_frame_{frame_index:03d}.png")
                difference.save(sample_dir / f"difference_frame_{frame_index:03d}.png")
        for filename, content in (
            ("comparison.mp4", comparison_video),
            ("differences.mp4", difference_video),
        ):
            try:
                _write_video(sample_dir / filename, content, fps)
            except Exception as exc:
                report["video_errors"].append(
                    {"sample": key, "file": filename, "error": f"{type(exc).__name__}: {exc}"}
                )
        report["samples"].append(
            {
                "key": key,
                "record_uid": sample["record_uid"],
                "directory": str(sample_dir.relative_to(output_dir)),
                "frames": frames,
                "selected_frames": list(valid),
            }
        )
    return report


def _metric(metrics: Mapping[str, object], name: str) -> str:
    value = metrics.get(name)
    if value is None:
        return "—"
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)


def write_markdown(report: Mapping[str, object], path: Path) -> None:
    models = report["models"]
    assert isinstance(models, list)
    lines = [
        "# SwiftVR Stage-A Distillation Audit",
        "",
        "## Quality",
        "",
        "| Model | Vel rel-L2 ↓ | Vel cosine ↑ | Teacher PSNR ↑ | Teacher SSIM ↑ | GT PSNR ↑ | GT SSIM ↑ | GT temporal MSE ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        metrics = model["metrics"]
        lines.append(
            "| {label} | {rel} | {cos} | {rpsnr} | {rssim} | {gpsnr} | {gssim} | {temp} |".format(
                label=model["label"],
                rel=_metric(metrics, "velocity_relative_l2"),
                cos=_metric(metrics, "velocity_cosine"),
                rpsnr=_metric(metrics, "student_teacher_psnr"),
                rssim=_metric(metrics, "student_teacher_ssim"),
                gpsnr=_metric(metrics, "student_gt_psnr"),
                gssim=_metric(metrics, "student_gt_ssim"),
                temp=_metric(metrics, "student_gt_temporal_difference_mse"),
            )
        )
    teacher = report.get("teacher")
    if isinstance(teacher, Mapping):
        metrics = teacher.get("metrics", {})
        lines.append(
            "| Conditional teacher (cached) | 0 | 1 | — | — | {gpsnr} | {gssim} | {temp} |".format(
                gpsnr=_metric(metrics, "teacher_gt_psnr"),
                gssim=_metric(metrics, "teacher_gt_ssim"),
                temp=_metric(metrics, "teacher_gt_temporal_difference_mse"),
            )
        )

    lines += [
        "",
        "## Compute",
        "",
        "| Model | Params | Trainable | Encoder ms | DiT ms | Decoder ms | E2E ms | Effective FPS | Peak GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        params = model["parameters"]
        profile = model["profile"]
        lines.append(
            "| {label} | {params:,} | {trainable:,} | {enc:.3f} | {dit:.3f} | {dec:.3f} | {e2e:.3f} | {fps:.3f} | {peak:.3f} |".format(
                label=model["label"],
                params=int(params["total_parameters"]),
                trainable=int(params["trainable_parameters"]),
                enc=float(profile["encoder_latency"]["median_ms"]),
                dit=float(profile["transformer_latency"]["median_ms"]),
                dec=float(profile["decoder_latency"]["median_ms"]),
                e2e=float(profile["end_to_end_latency"]["median_ms"]),
                fps=float(profile["effective_fps"]),
                peak=float(profile["peak_allocated_gb"]),
            )
        )
    lines += [
        "",
        "All student checkpoints share the same Stage-A architecture; compute differences should therefore be measurement noise unless the checkpoint changes structure.",
        "",
        "Teacher quality is decoded from the immutable cached conditional-teacher velocity on the same deterministic validation views. Teacher runtime is intentionally not inferred from the cache.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-A audit requires CUDA")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.visual_samples < 0 or args.latency_repeats <= 0 or args.latency_warmup < 0:
        raise ValueError("visual/latency arguments are invalid")
    if args.visual_video_fps <= 0 or args.difference_scale <= 0:
        raise ValueError("visual FPS and difference scale must be positive")

    model_specs = parse_model_specs(args.model)
    visual_frame_indices = parse_frame_indices(args.visual_frame_indices)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    device = torch.device("cuda")
    folded_config = gate.validate_folded_checkpoint(
        base_checkpoint,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
    )
    dtype = gate.resolve_runtime_dtype(
        args.dtype,
        folded_config,
        device,
        allow_mismatch=args.allow_dtype_mismatch,
    )
    cache = TeacherVelocityCache(args.teacher_cache)
    dataset = build_validation_dataset(args, cache)
    first_batch = next(iter(build_loader(dataset, args)))
    lpips_model = _load_lpips(device) if args.lpips else None

    report: dict[str, object] = {
        "format_version": 1,
        "kind": "swiftvr_stage_a_distillation_audit",
        "base_checkpoint": str(base_checkpoint),
        "teacher_cache": str(cache.root),
        "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
        "runtime_dtype": str(dtype).removeprefix("torch."),
        "attention_backend": args.attention_backend,
        "validation_samples": len(dataset),
        "models": [],
    }
    visual_bank: dict[str, dict[str, object]] = {}

    for model_index, (label, delta_checkpoint) in enumerate(model_specs):
        print(f"[{model_index + 1}/{len(model_specs)}] loading {label}", flush=True)
        closure = load_student(
            args,
            base_checkpoint=base_checkpoint,
            delta_checkpoint=delta_checkpoint,
            device=device,
            dtype=dtype,
        )
        params = parameter_summary(closure)
        loader = build_loader(dataset, args)
        started = time.perf_counter()
        metrics, visuals = evaluate_model(
            closure,
            loader,
            cache,
            device=device,
            dtype=dtype,
            lpips_model=lpips_model,
            visual_limit=args.visual_samples,
        )
        evaluation_seconds = time.perf_counter() - started
        profile = profile_model(
            closure,
            first_batch,
            device=device,
            dtype=dtype,
            args=args,
        )
        checkpoint_bytes = None
        if delta_checkpoint is not None:
            checkpoint_bytes = sum(
                path.stat().st_size for path in delta_checkpoint.rglob("*") if path.is_file()
            )
        model_record = {
            "label": label,
            "delta_checkpoint": None if delta_checkpoint is None else str(delta_checkpoint),
            "delta_checkpoint_bytes": checkpoint_bytes,
            "parameters": params,
            "metrics": metrics,
            "evaluation_seconds": evaluation_seconds,
            "profile": profile,
        }
        report["models"].append(model_record)
        print(
            f"{label}: rel_l2={metrics['velocity_relative_l2']:.6f} "
            f"cos={metrics['velocity_cosine']:.6f} "
            f"ref_psnr={metrics['student_teacher_psnr']:.4f} "
            f"gt_psnr={metrics['student_gt_psnr']:.4f}",
            flush=True,
        )

        for key, sample in visuals.items():
            if key not in visual_bank:
                visual_bank[key] = {
                    "record_uid": sample["record_uid"],
                    "lq_input": sample["lq_input"],
                    "target": sample["target"],
                    "teacher": sample["teacher"],
                    "students": {},
                }
            visual_bank[key]["students"][label] = sample["student"]

        del closure
        gc.collect()
        torch.cuda.empty_cache()

    if report["models"]:
        first_metrics = report["models"][0]["metrics"]
        report["teacher"] = {
            "kind": "cached_conditional_teacher",
            "metrics": {
                key: value
                for key, value in first_metrics.items()
                if key.startswith("teacher_gt_")
            },
            "runtime_profile": None,
        }

    missing_visual_models = {
        key: [label for label, _ in model_specs if label not in sample["students"]]
        for key, sample in visual_bank.items()
    }
    missing_visual_models = {key: value for key, value in missing_visual_models.items() if value}
    if missing_visual_models:
        raise RuntimeError(f"Visual sample identities changed across models: {missing_visual_models}")

    report["visuals"] = export_visuals(
        visual_bank,
        [label for label, _ in model_specs],
        output_dir=output_dir,
        frame_indices=visual_frame_indices,
        fps=args.visual_video_fps,
        difference_scale=args.difference_scale,
    )
    json_path = output_dir / "stage_a_audit.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = output_dir / "stage_a_audit.md"
    write_markdown(report, markdown_path)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
