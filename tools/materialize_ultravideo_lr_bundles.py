#!/usr/bin/env python3
"""Materialize deterministic UltraVideo training LR views into compact safetensors bundles.

Input is a cleaned deterministic view plan. One output bundle is written per raw
UltraVideo clip, with one uint8 [T,C,H,W] tensor key per selected view. Clean HR
is decoded only transiently to construct clean 3x HQ crops; it is not stored.

Degradation v1 intentionally spans a broad mild-to-strong restoration domain
instead of cloning the legacy degradation histogram. Parameters are fixed within
each 13-frame view. Only the zero-mean Gaussian noise realization changes by frame.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter
from safetensors import safe_open
from safetensors.torch import save_file

DEGRADATION_VERSION = "ultravideo_degradation_v1"
BUNDLE_KIND = "ultravideo_lr_bundle_v1"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Expected JSON object in {path}")
                rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    tmp.replace(path)


def _stable_unit(seed: int, *parts: object) -> float:
    payload = ":".join([str(int(seed)), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _stable_seed(seed: int, *parts: object) -> int:
    return int(_stable_unit(seed, *parts) * (2**32 - 1))


def _degradation_parameters(seed: int, clip_id: str, view_index: int) -> dict[str, object]:
    severity = _stable_unit(seed, clip_id, int(view_index), "severity")
    return {
        "version": DEGRADATION_VERSION,
        "severity": float(severity),
        "blur_sigma": float(0.10 + 2.00 * severity),
        "resize_scale": float(1.00 - 0.50 * severity),
        "noise_std_255": float(1.50 * severity),
        "jpeg_quality": int(round(95.0 - 39.0 * severity)),
    }


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _degrade_hq(
    hq: Image.Image,
    params: Mapping[str, object],
    *,
    noise_seed: int,
) -> Image.Image:
    sigma = float(params["blur_sigma"])
    image = hq.filter(ImageFilter.GaussianBlur(radius=sigma))

    width, height = image.size
    resize_scale = float(params["resize_scale"])
    if resize_scale < 0.999:
        small = (
            max(1, round(width * resize_scale)),
            max(1, round(height * resize_scale)),
        )
        image = image.resize(small, resample=Image.Resampling.BICUBIC)
        image = image.resize((width, height), resample=Image.Resampling.BICUBIC)

    noise_std = float(params["noise_std_255"])
    if noise_std > 0:
        rng = np.random.default_rng(int(noise_seed))
        array = np.asarray(image, dtype=np.float32)
        array += rng.normal(0.0, noise_std, size=array.shape).astype(np.float32)
        image = Image.fromarray(
            np.clip(array, 0, 255).round().astype(np.uint8),
            mode="RGB",
        )
    return _jpeg_roundtrip(image, int(params["jpeg_quality"]))


def _decord_reader(path: str):
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("decord is required for UltraVideo materialization") from exc
    return decord.VideoReader(path, ctx=decord.cpu(0))


def _batch_to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return np.asarray(value)


def _clean_hq_crop(
    raw: np.ndarray,
    row: Mapping[str, object],
) -> Image.Image:
    """Extract the planned clean HQ crop without resizing a whole 4K/8K frame."""
    image = Image.fromarray(np.asarray(raw, dtype=np.uint8), mode="RGB")
    source_width, source_height = image.size
    canonical_width = int(row["canonical_hr_width"])
    canonical_height = int(row["canonical_hr_height"])
    scale = int(row["scale"])
    top, left, crop_hq_h, crop_hq_w = (int(value) for value in row["crop_box_hq"])
    crop_hr_w = crop_hq_w * scale
    crop_hr_h = crop_hq_h * scale
    canonical_left = left * scale
    canonical_top = top * scale

    if canonical_left + crop_hr_w > canonical_width or canonical_top + crop_hr_h > canonical_height:
        raise ValueError("Planned crop exceeds canonical HR geometry")

    ratio = source_width / float(canonical_width)
    rounded_ratio = round(ratio)
    same_aspect_ratio = abs(
        source_height / float(max(canonical_height, 1)) - ratio
    ) < 0.01

    if rounded_ratio in (1, 2) and abs(ratio - rounded_ratio) < 1e-6 and same_aspect_ratio:
        factor = int(rounded_ratio)
        x0 = canonical_left * factor
        y0 = canonical_top * factor
        x1 = (canonical_left + crop_hr_w) * factor
        y1 = (canonical_top + crop_hr_h) * factor
        native_crop = image.crop((x0, y0, x1, y1))
        if factor != 1:
            native_crop = native_crop.resize(
                (crop_hr_w, crop_hr_h),
                resample=Image.Resampling.LANCZOS,
            )
        hr_crop = native_crop
    else:
        # Conservative fallback for any unexpected geometry.
        canonical = image
        if image.size != (canonical_width, canonical_height):
            target_height_before_crop = max(
                canonical_height,
                round(source_height * canonical_width / source_width),
            )
            canonical = image.resize(
                (canonical_width, target_height_before_crop),
                resample=Image.Resampling.LANCZOS,
            )
            if canonical.size[1] != canonical_height:
                canonical = canonical.crop((0, 0, canonical_width, canonical_height))
        hr_crop = canonical.crop(
            (
                canonical_left,
                canonical_top,
                canonical_left + crop_hr_w,
                canonical_top + crop_hr_h,
            )
        )

    if hr_crop.size != (crop_hr_w, crop_hr_h):
        raise ValueError(f"Unexpected HR crop size {hr_crop.size}")
    return hr_crop.resize(
        (crop_hq_w, crop_hq_h),
        resample=Image.Resampling.BOX,
    )


def _validate_existing_bundle(
    path: Path,
    expected_keys: Sequence[str],
    *,
    clip_length: int,
    crop_size: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        if keys != sorted(expected_keys):
            raise ValueError(f"{path}: tensor keys differ: {keys} vs {sorted(expected_keys)}")
        for key in expected_keys:
            tensor = handle.get_tensor(key)
            if tensor.dtype != torch.uint8:
                raise ValueError(f"{path}:{key}: expected uint8, got {tensor.dtype}")
            expected_shape = (clip_length, 3, crop_size, crop_size)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{path}:{key}: expected {expected_shape}, got {tuple(tensor.shape)}"
                )


def _view_seed(seed: int, row: Mapping[str, object]) -> int:
    return _stable_seed(seed, row["clip_id"], row["view_index"], "distillation_view")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.decode_batch_size <= 0 or args.shard_count <= 0:
        raise ValueError("decode-batch-size and shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.max_clips < 0:
        raise ValueError("max-clips must be non-negative")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")

    plan_path = args.plan.expanduser().resolve()
    rows = _read_jsonl(plan_path)
    if not rows:
        raise RuntimeError("View plan is empty")

    # Preserve global plan order so materialized_index is independent of sharding.
    indexed_rows = list(enumerate(rows))
    grouped: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    clip_order: list[str] = []
    for global_index, row in indexed_rows:
        clip_id = str(row["clip_id"])
        if clip_id not in grouped:
            clip_order.append(clip_id)
        grouped[clip_id].append((global_index, row))

    selected_clips = [
        clip_id
        for index, clip_id in enumerate(clip_order)
        if index % int(args.shard_count) == int(args.shard_index)
    ]
    if args.max_clips:
        selected_clips = selected_clips[: int(args.max_clips)]

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output}")
    bundles_dir = output / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    total_bytes = 0
    started = time.perf_counter()

    for clip_number, clip_id in enumerate(selected_clips, start=1):
        values = sorted(grouped[clip_id], key=lambda item: int(item[1]["view_index"]))
        view_rows = [row for _, row in values]
        raw_paths = {str(row["raw_video"]) for row in view_rows}
        if len(raw_paths) != 1:
            raise ValueError(f"{clip_id}: multiple raw_video paths")
        raw_video = next(iter(raw_paths))
        clip_lengths = {int(row["clip_length"]) for row in view_rows}
        crop_sizes = {int(row["crop_box_hq"][2]) for row in view_rows} | {
            int(row["crop_box_hq"][3]) for row in view_rows
        }
        if len(clip_lengths) != 1 or len(crop_sizes) != 1:
            raise ValueError(f"{clip_id}: inconsistent clip/crop geometry")
        clip_length = next(iter(clip_lengths))
        crop_size = next(iter(crop_sizes))

        bundle_path = bundles_dir / f"{clip_id}.safetensors"
        expected_keys = [f"lr_{int(row['view_index']):02d}" for row in view_rows]

        degradation_by_view = {
            int(row["view_index"]): _degradation_parameters(
                int(args.seed), clip_id, int(row["view_index"])
            )
            for row in view_rows
        }

        if bundle_path.is_file() and args.resume:
            _validate_existing_bundle(
                bundle_path,
                expected_keys,
                clip_length=clip_length,
                crop_size=crop_size,
            )
        else:
            reader = _decord_reader(raw_video)
            frame_count = len(reader)
            position_refs: dict[int, list[tuple[dict[str, object], int]]] = defaultdict(list)
            buffers: dict[int, np.ndarray] = {}
            for row in view_rows:
                view_index = int(row["view_index"])
                positions = [int(value) for value in row["raw_frame_positions"]]
                if len(positions) != clip_length:
                    raise ValueError(f"{clip_id}/view{view_index}: clip length mismatch")
                if min(positions) < 0 or max(positions) >= frame_count:
                    raise ValueError(f"{clip_id}/view{view_index}: frame position out of range")
                buffers[view_index] = np.empty(
                    (clip_length, crop_size, crop_size, 3),
                    dtype=np.uint8,
                )
                for temporal_index, position in enumerate(positions):
                    position_refs[position].append((row, temporal_index))

            positions = sorted(position_refs)
            for start in range(0, len(positions), int(args.decode_batch_size)):
                batch_positions = positions[start : start + int(args.decode_batch_size)]
                decoded = _batch_to_numpy(reader.get_batch(batch_positions))
                for local_index, position in enumerate(batch_positions):
                    raw_frame = decoded[local_index]
                    for row, temporal_index in position_refs[position]:
                        view_index = int(row["view_index"])
                        hq = _clean_hq_crop(raw_frame, row)
                        params = degradation_by_view[view_index]
                        lr = _degrade_hq(
                            hq,
                            params,
                            noise_seed=_stable_seed(
                                int(args.seed),
                                clip_id,
                                view_index,
                                int(position),
                                "noise",
                            ),
                        )
                        array = np.asarray(lr, dtype=np.uint8)
                        if bool(row.get("horizontal_flip", False)):
                            array = np.ascontiguousarray(array[:, ::-1])
                        buffers[view_index][temporal_index] = array
                del decoded

            tensors = {
                f"lr_{view_index:02d}": torch.from_numpy(array)
                .permute(0, 3, 1, 2)
                .contiguous()
                for view_index, array in buffers.items()
            }
            temp_path = bundle_path.with_suffix(".safetensors.tmp")
            save_file(
                tensors,
                str(temp_path),
                metadata={
                    "kind": BUNDLE_KIND,
                    "clip_id": clip_id,
                    "degradation_version": DEGRADATION_VERSION,
                },
            )
            os.replace(temp_path, bundle_path)
            _validate_existing_bundle(
                bundle_path,
                expected_keys,
                clip_length=clip_length,
                crop_size=crop_size,
            )

        total_bytes += bundle_path.stat().st_size
        for global_index, row in values:
            view_index = int(row["view_index"])
            positions = [int(value) for value in row["raw_frame_positions"]]
            scale = int(row["scale"])
            manifest_rows.append(
                {
                    "kind": "ultravideo_materialized_lr_view_v1",
                    "materialized_index": int(global_index),
                    "bundle_path": str(bundle_path),
                    "bundle_key": f"lr_{view_index:02d}",
                    "dataset": "UltraVideo",
                    "split": row.get("split"),
                    "sample_id": clip_id,
                    "record_uid": f"ultravideo:{clip_id}",
                    "variant": DEGRADATION_VERSION,
                    "source_group_uid": row.get("source_group_uid"),
                    "source_url": row.get("source_url"),
                    "raw_video": raw_video,
                    "frame_indices": positions,
                    "crop_top": int(row["crop_box_hq"][0]),
                    "crop_left": int(row["crop_box_hq"][1]),
                    "crop_size": crop_size,
                    "crop_box_hq": [int(value) for value in row["crop_box_hq"]],
                    "canonical_hr_width": int(row["canonical_hr_width"]),
                    "canonical_hr_height": int(row["canonical_hr_height"]),
                    "scale": scale,
                    "target_height": crop_size * scale,
                    "target_width": crop_size * scale,
                    "horizontal_flip": bool(row.get("horizontal_flip", False)),
                    "vertical_flip": False,
                    "view_index": view_index,
                    "view_seed": _view_seed(int(args.seed), row),
                    "selection_category": row.get("selection_category"),
                    "degradation": degradation_by_view[view_index],
                }
            )

        if clip_number == 1 or clip_number % int(args.progress_every) == 0 or clip_number == len(selected_clips):
            elapsed = time.perf_counter() - started
            print(
                f"materialized {clip_number}/{len(selected_clips)} clips "
                f"({clip_number / max(elapsed, 1e-9):.3f} clips/s) "
                f"bytes={total_bytes}",
                flush=True,
            )

    manifest_rows.sort(key=lambda row: int(row["materialized_index"]))
    _write_jsonl(output / "materialized_views.jsonl", manifest_rows)

    severity_values = [
        float(row["degradation"]["severity"])
        for row in manifest_rows
    ]
    summary = {
        "kind": "ultravideo_lr_materialization_v1",
        "source_plan": str(plan_path),
        "degradation_version": DEGRADATION_VERSION,
        "seed": int(args.seed),
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "input_view_count": len(rows),
        "input_clip_count": len(clip_order),
        "materialized_clip_count": len(selected_clips),
        "materialized_view_count": len(manifest_rows),
        "bundle_count": len(selected_clips),
        "bundle_bytes": int(total_bytes),
        "bundle_gib": float(total_bytes / (1024**3)),
        "severity": {
            "min": float(min(severity_values)) if severity_values else None,
            "mean": float(np.mean(severity_values)) if severity_values else None,
            "median": float(np.median(severity_values)) if severity_values else None,
            "max": float(max(severity_values)) if severity_values else None,
        },
        "stored_tensors": "LR uint8 only; clean HR/HQ are transient and reproducible from raw source + plan",
        "degradation_definition": {
            "severity_distribution": "deterministic Uniform[0,1) per selected view",
            "blur_sigma": "0.10 + 2.00 * severity",
            "resize_scale": "1.00 - 0.50 * severity",
            "noise_std_255": "1.50 * severity",
            "jpeg_quality": "round(95 - 39 * severity), RGB 4:4:4",
            "temporal_rule": "parameters fixed within view; noise realization deterministic per frame",
        },
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
