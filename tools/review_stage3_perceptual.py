#!/usr/bin/env python3
"""Export fixed visual comparisons and perceptual metrics for Stage-3 checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Mapping

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from smoke_training_forward import (
    configure_train_scope,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training import (
    SwiftVRTrainingForward,
    VideoMetricAccumulator,
    capture_trainable_parameters,
    cast_trainable_parameters,
    load_delta_checkpoint,
)
from swiftvr.training.perceptual_review import (
    FULL_REFERENCE_METRICS,
    NO_REFERENCE_METRICS,
    IQAMetricSuite,
    StudentCheckpointSpec,
    build_html_report,
    make_comparison_frame,
    make_difference_frame,
    parse_csv_ints,
    parse_metric_names,
    parse_student_checkpoint,
    restore_trainable_parameters,
    sanitize_label,
    write_json_atomic,
    write_jsonl_atomic,
    write_summary_csv,
    write_tensorboard_curves,
)
from swiftvr.training.reference import (
    ConditionalReferenceCache,
    batch_sample_identity,
    pairwise_video_metrics,
    sha256_file,
)


RESERVED_LABELS = {"gt", "lq_bicubic", "conditional_reference"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--student-checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeat for each delta checkpoint, e.g. step300=/path/to/checkpoint.",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Include the zero-adapter hard-removal base as step0.",
    )
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-device", default=None)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument(
        "--metrics",
        default="lpips,dists,musiq",
        help="Comma-separated subset of lpips,dists,musiq.",
    )
    parser.add_argument("--metric-batch-size", type=int, default=1)
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--visual-frame-indices",
        default="0,6,12",
        help="Comma-separated local frame indices for PNG comparison sheets.",
    )
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--difference-scale", type=float, default=4.0)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Reuse complete per-checkpoint prediction caches in an existing output directory.",
    )
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    return parser


def _resolve_students(args: argparse.Namespace) -> list[StudentCheckpointSpec]:
    students: list[StudentCheckpointSpec] = []
    if args.include_base:
        students.append(StudentCheckpointSpec("step0", None, 0))
    students.extend(parse_student_checkpoint(value) for value in args.student_checkpoint)
    if not students:
        raise ValueError("Use --include-base and/or at least one --student-checkpoint")
    labels = [spec.label for spec in students]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate student labels: {labels}")
    invalid = sorted(set(labels) & RESERVED_LABELS)
    if invalid:
        raise ValueError(f"Student labels are reserved: {invalid}")
    for spec in students:
        if spec.path is not None and not spec.path.is_dir():
            raise FileNotFoundError(spec.path)
    return students


def _validate_reference_cache(
    cache: ConditionalReferenceCache,
    args: argparse.Namespace,
) -> None:
    expected = {
        "val_split": args.val_split,
        "clip_length": int(args.clip_length),
        "crop_size": int(args.crop_size),
        "scale": int(args.scale),
    }
    differences = [
        f"{key}: cache={cache.metadata.get(key)!r}, current={value!r}"
        for key, value in expected.items()
        if cache.metadata.get(key) != value
    ]
    saved_hashes = cache.metadata.get("val_manifest_sha256")
    if not isinstance(saved_hashes, Mapping):
        differences.append("cache does not contain val_manifest_sha256")
    else:
        for path in (item.expanduser().resolve() for item in args.val_manifest):
            if saved_hashes.get(str(path)) != sha256_file(path):
                differences.append(f"manifest hash mismatch: {path}")
    if differences:
        raise ValueError(
            "Reference cache configuration differs:\n  " + "\n  ".join(differences)
        )


def _move_batch(
    batch: dict[str, object],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    result = dict(batch)
    for key in ("lr", "hq", "hr"):
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )
    return result


def _prediction_path(
    prediction_root: Path,
    label: str,
    sample_index: int,
    key: str,
) -> Path:
    return prediction_root / label / f"{sample_index:04d}_{key}.safetensors"


def _load_prediction(
    prediction_root: Path,
    label: str,
    sample_index: int,
    key: str,
) -> torch.Tensor:
    path = _prediction_path(prediction_root, label, sample_index, key)
    tensors = load_file(str(path), device="cpu")
    if "prediction" not in tensors:
        raise KeyError(f"{path} does not contain prediction")
    return tensors["prediction"].float()


def _cached_model_row(
    *,
    prediction_root: Path,
    requested: StudentCheckpointSpec,
    expected_samples: int,
    cache: ConditionalReferenceCache,
    student_base_checkpoint: Path,
) -> dict[str, object] | None:
    model_dir = prediction_root / requested.label
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        row = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(row, dict):
        return None
    expected_checkpoint = None if requested.path is None else str(requested.path)
    if row.get("label") != requested.label:
        return None
    if row.get("checkpoint") != expected_checkpoint:
        return None
    if row.get("student_base_checkpoint") != str(student_base_checkpoint):
        return None
    if int(row.get("sample_count", -1)) != expected_samples:
        return None
    samples = cache.metadata.get("samples")
    if not isinstance(samples, list) or len(samples) < expected_samples:
        return None
    for sample_index, item in enumerate(samples[:expected_samples]):
        if not isinstance(item, Mapping) or not isinstance(item.get("key"), str):
            return None
        if not _prediction_path(
            prediction_root,
            requested.label,
            sample_index,
            str(item["key"]),
        ).is_file():
            return None
    return row


def _cache_student_predictions(
    *,
    args: argparse.Namespace,
    students: list[StudentCheckpointSpec],
    loader: DataLoader,
    expected_samples: int,
    cache: ConditionalReferenceCache,
    prediction_root: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, object]]:
    model_rows: list[dict[str, object]] = []
    pending: list[StudentCheckpointSpec] = []
    for requested in students:
        cached = _cached_model_row(
            prediction_root=prediction_root,
            requested=requested,
            expected_samples=expected_samples,
            cache=cache,
            student_base_checkpoint=args.student_base_checkpoint.expanduser().resolve(),
        )
        if cached is None:
            pending.append(requested)
        else:
            print(f"[{requested.label}] reusing complete prediction cache", flush=True)
            model_rows.append(cached)
    order = {spec.label: index for index, spec in enumerate(students)}
    if not pending:
        return sorted(model_rows, key=lambda row: order[str(row["label"])])

    base = args.student_base_checkpoint.expanduser().resolve()
    reae = ReAE(str(base / args.reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(base),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    configure_train_scope(reae, transformer, "adapter")
    reae.to(device=device, dtype=dtype).eval()
    transformer.to(device=device, dtype=dtype)
    closure = SwiftVRTrainingForward(
        reae,
        transformer,
        latent_loss_weight=0.0,
        training_safe_transformer=True,
        prepare_transformer=True,
        attention_backend=args.attention_backend,
    )
    cast_trainable_parameters(closure, dtype=torch.float32)
    base_trainable = capture_trainable_parameters(closure)
    closure.eval()

    autocast_enabled = (
        device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    )
    for requested in pending:
        if requested.path is None:
            restore_trainable_parameters(closure, base_trainable)
            step = 0
            checkpoint = None
        else:
            metadata = load_delta_checkpoint(
                requested.path,
                closure,
                optimizer=None,
                strict=True,
                map_location="cpu",
            )
            step = int(metadata["step"])
            checkpoint = str(requested.path)
        model_dir = prediction_root / requested.label
        model_dir.mkdir(parents=True, exist_ok=True)
        processed = 0
        started = time.perf_counter()

        with torch.no_grad():
            for batch_cpu in loader:
                if processed >= expected_samples:
                    break
                identity = batch_sample_identity(batch_cpu, 0)
                if str(identity["key"]) not in cache.samples_by_key:
                    raise KeyError(
                        f"Reference cache has no entry for {identity['record_uid']} "
                        f"frames={identity['frame_indices']}"
                    )
                batch = _move_batch(batch_cpu, device, dtype)
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype if autocast_enabled else torch.float32,
                    enabled=autocast_enabled,
                ):
                    output = closure(batch)
                prediction = output.get("prediction_clamped")
                if not isinstance(prediction, torch.Tensor):
                    raise TypeError("Student forward did not return prediction_clamped")
                sample = prediction[0].detach().to(device="cpu", dtype=torch.float16)
                path = _prediction_path(
                    prediction_root,
                    requested.label,
                    processed,
                    str(identity["key"]),
                )
                save_file({"prediction": sample.contiguous()}, str(path))
                processed += 1
                print(
                    f"[{requested.label}] cached {processed}/{expected_samples}: "
                    f"{identity['record_uid']}",
                    flush=True,
                )

        if processed != expected_samples:
            raise RuntimeError(
                f"{requested.label}: cached {processed}, expected {expected_samples}"
            )
        row = {
            "label": requested.label,
            "step": step,
            "checkpoint": checkpoint,
            "student_base_checkpoint": str(base),
            "sample_count": processed,
            "inference_seconds": time.perf_counter() - started,
        }
        write_json_atomic(model_dir / "metadata.json", row)
        model_rows.append(row)

    del closure, transformer, reae
    from swiftvr.models.transformer import _WindowIndexCache, _WindowRuntimeMetaCache

    _WindowIndexCache.clear()
    _WindowRuntimeMetaCache.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sorted(model_rows, key=lambda row: order[str(row["label"])])


def _append_perceptual(
    *,
    suite: IQAMetricSuite,
    candidate: torch.Tensor,
    gt: torch.Tensor,
    reference: torch.Tensor,
    row: dict[str, object],
) -> None:
    for name in suite.metric_names:
        if name in FULL_REFERENCE_METRICS:
            to_gt = suite.score_metric(name, candidate, reference=gt)
            row[f"gt_{name}"] = float(to_gt["mean"])
            row[f"gt_{name}_frames"] = to_gt["frames"]
            to_reference = suite.score_metric(name, candidate, reference=reference)
            row[f"reference_{name}"] = float(to_reference["mean"])
            row[f"reference_{name}_frames"] = to_reference["frames"]
        elif name in NO_REFERENCE_METRICS:
            no_reference = suite.score_metric(name, candidate)
            row[f"nr_{name}"] = float(no_reference["mean"])
            row[f"nr_{name}_frames"] = no_reference["frames"]


def _aggregate_summary(
    sample_metric_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    model_meta = {str(row["label"]): row for row in model_rows}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in sample_metric_rows:
        grouped[str(row["label"])].append(row)

    preferred = ["gt", "lq_bicubic", "conditional_reference"] + [
        str(row["label"]) for row in model_rows
    ]
    summaries: list[dict[str, object]] = []
    for label in preferred:
        rows = grouped.get(label, [])
        if not rows:
            continue
        summary: dict[str, object] = {
            "label": label,
            "step": model_meta.get(label, {}).get("step"),
            "checkpoint": model_meta.get(label, {}).get("checkpoint"),
            "samples": len(rows),
        }
        scalar_keys = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and key not in {"sample_index", "step"}
            }
        )
        for key in scalar_keys:
            summary[key] = sum(float(row[key]) for row in rows if key in row) / sum(
                1 for row in rows if key in row
            )
        summaries.append(summary)
    return summaries


def _write_videos(
    *,
    sample_dir: Path,
    comparison_frames: list[np.ndarray],
    difference_frames: list[np.ndarray],
    fps: float,
) -> tuple[str, str]:
    comparison_path = sample_dir / "comparison.mp4"
    difference_path = sample_dir / "difference_to_gt.mp4"
    with imageio.get_writer(
        comparison_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in comparison_frames:
            writer.append_data(frame)
    with imageio.get_writer(
        difference_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in difference_frames:
            writer.append_data(frame)
    return comparison_path.name, difference_path.name


def main() -> int:
    args = build_parser().parse_args()
    students = _resolve_students(args)
    metric_names = parse_metric_names(args.metrics)
    frame_indices = parse_csv_ints(args.visual_frame_indices)
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.difference_scale <= 0:
        raise ValueError("--difference-scale must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    if args.overwrite and args.reuse_predictions:
        raise ValueError("--overwrite and --reuse-predictions are mutually exclusive")
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif not args.reuse_predictions:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite or --reuse-predictions"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_root = output_dir / "prediction_cache"
    prediction_root.mkdir(exist_ok=True)

    device = torch.device(args.device)
    metric_device = torch.device(args.metric_device or args.device)
    base = args.student_base_checkpoint.expanduser().resolve()
    folded_config = validate_folded_checkpoint(
        base,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
    )
    dtype = resolve_runtime_dtype(
        args.dtype,
        folded_config,
        device,
        allow_mismatch=False,
    )
    cache = ConditionalReferenceCache(args.reference_cache)
    _validate_reference_cache(cache, args)

    dataset = TripletVideoDataset(
        args.val_manifest,
        split=args.val_split,
        training=False,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        drop_short_sequences=True,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
    )
    cache_samples = int(cache.metadata["sample_count"])
    expected_samples = min(len(dataset), cache_samples)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        expected_samples = min(expected_samples, int(args.max_samples))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model_rows = _cache_student_predictions(
        args=args,
        students=students,
        loader=loader,
        expected_samples=expected_samples,
        cache=cache,
        prediction_root=prediction_root,
        device=device,
        dtype=dtype,
    )

    suite = IQAMetricSuite(
        metric_names,
        device=metric_device,
        batch_size=args.metric_batch_size,
        cache_dir=args.metric_cache_dir,
    )
    visuals_root = output_dir / "visuals"
    if visuals_root.exists():
        shutil.rmtree(visuals_root)
    visuals_root.mkdir()
    sample_metric_rows: list[dict[str, object]] = []
    sample_report_rows: list[dict[str, object]] = []
    global_video_metrics: dict[str, VideoMetricAccumulator] = {}
    model_labels = [str(row["label"]) for row in model_rows]

    processed = 0
    metric_started = time.perf_counter()
    for batch_cpu in loader:
        if processed >= expected_samples:
            break
        identity = batch_sample_identity(batch_cpu, 0)
        key = str(identity["key"])
        record_uid = str(identity["record_uid"])
        gt = batch_cpu["hr"][0].float()
        lr = batch_cpu["lr"]
        if not isinstance(lr, torch.Tensor):
            raise TypeError("Dataset did not return lr tensor")
        batch_size, frames, channels, _, _ = lr.shape
        target_height, target_width = gt.shape[-2:]
        lq_bicubic = F.interpolate(
            lr.reshape(batch_size * frames, channels, *lr.shape[-2:]),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, frames, channels, target_height, target_width)[0]
        reference = cache.load(
            identity,
            device="cpu",
            dtype=torch.float32,
        )["prediction"].clamp(0.0, 1.0)

        candidates: OrderedDict[str, torch.Tensor] = OrderedDict()
        candidates["gt"] = gt
        candidates["lq_bicubic"] = lq_bicubic
        candidates["conditional_reference"] = reference
        for label in model_labels:
            candidates[label] = _load_prediction(
                prediction_root,
                label,
                processed,
                key,
            ).clamp(0.0, 1.0)

        for label, candidate in candidates.items():
            row: dict[str, object] = {
                "sample_index": processed,
                "record_uid": record_uid,
                "key": key,
                "label": label,
            }
            if label != "gt":
                accumulator = global_video_metrics.setdefault(
                    label, VideoMetricAccumulator()
                )
                accumulator.update(
                    candidate.unsqueeze(0),
                    gt.unsqueeze(0),
                    clamp=True,
                )
                conventional = pairwise_video_metrics(
                    candidate.unsqueeze(0),
                    gt.unsqueeze(0),
                )
                for name in ("psnr", "ssim", "mae", "mse", "rmse", "pixel_l1", "temporal_mse"):
                    row[f"gt_{name}"] = float(conventional[name])
                _append_perceptual(
                    suite=suite,
                    candidate=candidate,
                    gt=gt,
                    reference=reference,
                    row=row,
                )
            else:
                for name in NO_REFERENCE_METRICS & set(metric_names):
                    score = suite.score_metric(name, candidate)
                    row[f"nr_{name}"] = float(score["mean"])
                    row[f"nr_{name}_frames"] = score["frames"]
            sample_metric_rows.append(row)

        sample_dir = visuals_root / f"{processed:03d}_{sanitize_label(record_uid)}"
        sample_dir.mkdir()
        selected_images: list[str] = []
        valid_frame_indices = tuple(
            index for index in frame_indices if index < int(gt.shape[0])
        )
        if not valid_frame_indices:
            raise ValueError(
                f"No visual frame indices are valid for T={gt.shape[0]}: {frame_indices}"
            )

        comparison_video_frames: list[np.ndarray] = []
        difference_video_frames: list[np.ndarray] = []
        for frame_index in range(int(gt.shape[0])):
            visible = OrderedDict(
                (label, video[frame_index])
                for label, video in candidates.items()
            )
            comparison = make_comparison_frame(visible)
            differences = OrderedDict(
                (label, video[frame_index])
                for label, video in candidates.items()
                if label != "gt"
            )
            difference = make_difference_frame(
                differences,
                gt[frame_index],
                scale=args.difference_scale,
            )
            comparison_array = np.asarray(comparison, dtype=np.uint8)
            difference_array = np.asarray(difference, dtype=np.uint8)
            if frame_index in valid_frame_indices:
                comparison_path = sample_dir / f"frame_{frame_index:03d}.png"
                difference_path = sample_dir / f"diff_{frame_index:03d}.png"
                comparison.save(comparison_path)
                difference.save(difference_path)
                selected_images.extend(
                    [
                        str(comparison_path.relative_to(output_dir)),
                        str(difference_path.relative_to(output_dir)),
                    ]
                )
            if not args.no_videos:
                comparison_video_frames.append(comparison_array)
                difference_video_frames.append(difference_array)

        comparison_video = None
        difference_video = None
        if not args.no_videos:
            comparison_name, difference_name = _write_videos(
                sample_dir=sample_dir,
                comparison_frames=comparison_video_frames,
                difference_frames=difference_video_frames,
                fps=args.video_fps,
            )
            comparison_video = str(
                (sample_dir / comparison_name).relative_to(output_dir)
            )
            difference_video = str(
                (sample_dir / difference_name).relative_to(output_dir)
            )

        sample_report_rows.append(
            {
                "sample_index": processed,
                "record_uid": record_uid,
                "key": key,
                "comparison_video": comparison_video,
                "difference_video": difference_video,
                "comparison_images": selected_images,
            }
        )
        processed += 1
        print(
            f"[metrics/visuals] {processed}/{expected_samples}: {record_uid}",
            flush=True,
        )

    if processed != expected_samples:
        raise RuntimeError(f"Processed {processed}, expected {expected_samples}")

    summary_rows = _aggregate_summary(sample_metric_rows, model_rows)
    for summary in summary_rows:
        label = str(summary["label"])
        accumulator = global_video_metrics.get(label)
        if accumulator is None:
            continue
        global_metrics = accumulator.compute()
        for name in ("psnr", "ssim", "mae", "mse", "rmse"):
            summary[f"gt_{name}"] = float(global_metrics[name])
    metric_metadata = suite.package_metadata()
    run_metadata = {
        "format_version": 1,
        "student_base_checkpoint": str(base),
        "students": model_rows,
        "reference_cache": str(args.reference_cache.expanduser().resolve()),
        "val_manifests": [
            str(path.expanduser().resolve()) for path in args.val_manifest
        ],
        "val_manifest_sha256": {
            str(path.expanduser().resolve()): sha256_file(path)
            for path in args.val_manifest
        },
        "val_split": args.val_split,
        "clip_length": args.clip_length,
        "crop_size": args.crop_size,
        "scale": args.scale,
        "sample_count": processed,
        "metrics": metric_metadata,
        "metric_seconds": time.perf_counter() - metric_started,
        "visual_frame_indices": list(frame_indices),
        "video_fps": args.video_fps,
        "difference_scale": args.difference_scale,
    }
    write_json_atomic(output_dir / "metadata.json", run_metadata)
    write_json_atomic(output_dir / "summary.json", summary_rows)
    write_jsonl_atomic(output_dir / "per_sample_metrics.jsonl", sample_metric_rows)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_json_atomic(output_dir / "samples.json", sample_report_rows)
    tensorboard_dir = (
        args.tensorboard_dir.expanduser().resolve()
        if args.tensorboard_dir is not None
        else output_dir / "tensorboard"
    )
    write_tensorboard_curves(tensorboard_dir, summary_rows)
    build_html_report(
        output_dir / "report.html",
        title="SwiftVR Stage-3 perceptual review",
        summary_rows=summary_rows,
        sample_rows=sample_report_rows,
        metric_metadata=metric_metadata,
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
