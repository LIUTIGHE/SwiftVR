#!/usr/bin/env python3
"""Audit the exact deterministic training views used by a SwiftVR run.

This is a read-only diagnostic. It reconstructs the train dataset from an
existing run_config + teacher-cache metadata, checks cache/view identity, reports
source/variant redundancy and view overlap, and profiles a source-balanced subset
for HR detail, LR detail loss, and input motion. No sample is filtered or
reweighted by this tool.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from swiftvr.data import TripletVideoDataset
from swiftvr.training.distillation import (
    DeterministicTripletViewDataset,
    TeacherVelocityCache,
)
from swiftvr.training.distillation_generalization import (
    cache_selected_indices,
    record_source_uid,
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


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


def _image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.height, image.width


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return {key: None for key in ("min", "p10", "p25", "median", "p75", "p90", "max", "mean")}
    return {
        "min": float(np.min(clean)),
        "p10": float(np.percentile(clean, 10)),
        "p25": float(np.percentile(clean, 25)),
        "median": float(np.median(clean)),
        "p75": float(np.percentile(clean, 75)),
        "p90": float(np.percentile(clean, 90)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
    }


def _pairwise_mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _interval_iou(start_a: int, length_a: int, start_b: int, length_b: int) -> float:
    end_a, end_b = start_a + length_a, start_b + length_b
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return float(intersection / union) if union > 0 else 0.0


def _box_iou(a: Sequence[int], b: Sequence[int]) -> float:
    top_a, left_a, height_a, width_a = (int(v) for v in a)
    top_b, left_b, height_b, width_b = (int(v) for v in b)
    bottom_a, right_a = top_a + height_a, left_a + width_a
    bottom_b, right_b = top_b + height_b, left_b + width_b
    intersection = max(0, min(bottom_a, bottom_b) - max(top_a, top_b)) * max(
        0, min(right_a, right_b) - max(left_a, left_b)
    )
    union = height_a * width_a + height_b * width_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _manifest_fields(manifests: Sequence[Path], split: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    continue
                if payload.get("split") != split:
                    continue
                counts.update(str(key) for key in payload)
    return dict(sorted(counts.items()))


def _resolve_training_config(
    run_config: Mapping[str, object], cache: TeacherVelocityCache
) -> dict[str, object]:
    metadata = cache.metadata
    manifests = run_config.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        manifests = metadata.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("Neither run_config nor teacher cache records training manifests")

    def cached(name: str, fallback: object | None = None) -> object:
        value = metadata.get(name, fallback)
        if value is None:
            raise ValueError(f"Teacher cache metadata is missing {name!r}")
        return value

    return {
        "manifests": [str(Path(str(path)).expanduser().resolve()) for path in manifests],
        "path_root": str(Path(str(metadata.get("path_root", "."))).expanduser().resolve()),
        "split": str(cached("split", "train")),
        "clip_length": int(cached("clip_length", run_config.get("clip_length"))),
        "crop_size": int(cached("crop_size", run_config.get("crop_size"))),
        "scale": int(cached("scale", run_config.get("scale"))),
        "views_per_record": int(cached("views_per_record", run_config.get("views_per_record"))),
        "view_seed": int(cached("view_seed", run_config.get("view_seed"))),
        "horizontal_flip_probability": float(metadata.get("horizontal_flip_probability", 0.5)),
        "vertical_flip_probability": float(metadata.get("vertical_flip_probability", 0.0)),
    }


def _view_spec(
    base: TripletVideoDataset,
    full: DeterministicTripletViewDataset,
    index: int,
    lr_sizes: Sequence[tuple[int, int]],
) -> dict[str, object]:
    record_index, view_index, seed = full.decode_index(index)
    record = base.records[record_index]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        temporal_start = base._temporal_start(record.frame_count)
        top, left, crop_height, crop_width = base._spatial_crop(lr_sizes[record_index])
        horizontal_flip = (
            base.training
            and base.horizontal_flip_probability > 0
            and float(torch.rand(()).item()) < base.horizontal_flip_probability
        )
        vertical_flip = (
            base.training
            and base.vertical_flip_probability > 0
            and float(torch.rand(()).item()) < base.vertical_flip_probability
        )
    positions = range(temporal_start, temporal_start + base.clip_length)
    frame_indices = [int(record.frame_indices[position]) for position in positions]
    return {
        "distillation_index": int(index),
        "record_index": int(record_index),
        "view_index": int(view_index),
        "view_seed": int(seed),
        "record_uid": f"{record.variant}:{record.sample_id}",
        "sample_id": record.sample_id,
        "variant": record.variant,
        "frame_indices": frame_indices,
        "temporal_start": int(temporal_start),
        "crop_box_lr": [int(top), int(left), int(crop_height), int(crop_width)],
        "horizontal_flip": bool(horizontal_flip),
        "vertical_flip": bool(vertical_flip),
    }


def _source_balanced_subset(
    cached_indices: Sequence[int],
    source_uids: Sequence[str],
    views_per_record: int,
    limit: int,
    seed: int,
) -> list[int]:
    indices = [int(index) for index in cached_indices]
    if limit <= 0 or limit >= len(indices):
        return indices
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        record_index = index // int(views_per_record)
        grouped[source_uids[record_index]].append(index)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    keys = sorted(grouped)
    key_order = torch.randperm(len(keys), generator=generator).tolist()
    keys = [keys[position] for position in key_order]
    for key in keys:
        values = grouped[key]
        order = torch.randperm(len(values), generator=generator).tolist()
        grouped[key] = [values[position] for position in order]

    selected: list[int] = []
    positions = {key: 0 for key in keys}
    while len(selected) < limit:
        added = False
        for key in keys:
            position = positions[key]
            values = grouped[key]
            if position >= len(values):
                continue
            selected.append(values[position])
            positions[key] = position + 1
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return selected


def _gray(video: torch.Tensor) -> torch.Tensor:
    weights = video.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (video * weights).sum(dim=1, keepdim=True)


def _highpass_energy(frame: torch.Tensor) -> float:
    gray = _gray(frame.unsqueeze(0))
    low = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
    return float((gray - low).abs().mean().item())


def _profile_view(sample: Mapping[str, object]) -> dict[str, float]:
    lr = sample["lr"]
    hr = sample["hr"]
    if not isinstance(lr, torch.Tensor) or not isinstance(hr, torch.Tensor):
        raise TypeError("Dataset sample must contain tensor lr/hr clips")
    middle = int(lr.shape[0] // 2)
    hr_mid = hr[middle].float()
    lr_mid = lr[middle].float()
    lr_up = F.interpolate(
        lr_mid.unsqueeze(0),
        size=tuple(int(value) for value in hr_mid.shape[-2:]),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0).clamp_(0.0, 1.0)
    lr_gray = _gray(lr.float())
    if lr.shape[0] > 1:
        adjacent_raw = (lr_gray[1:] - lr_gray[:-1]).abs().mean(dim=(1, 2, 3))
        frame_means = lr_gray.mean(dim=(1, 2, 3), keepdim=True)
        centered = lr_gray - frame_means
        adjacent_structure = (centered[1:] - centered[:-1]).abs().mean(dim=(1, 2, 3))
        mean_luma_delta = (frame_means[1:] - frame_means[:-1]).abs().flatten()
        motion = float(adjacent_raw.mean().item())
        structural_motion = float(adjacent_structure.mean().item())
        luma_motion = float(mean_luma_delta.mean().item())
        median_structure = float(adjacent_structure.median().item())
        spike_ratio = float(adjacent_structure.max().item() / max(median_structure, 1e-8))
    else:
        motion = structural_motion = luma_motion = 0.0
        spike_ratio = 1.0
    hr_detail = _highpass_energy(hr_mid)
    lr_detail = _highpass_energy(lr_up)
    return {
        "hr_highpass_l1": hr_detail,
        "lr_bicubic_highpass_l1": lr_detail,
        "recoverable_highpass_gap": hr_detail - lr_detail,
        "hr_vs_bicubic_lr_mae": float((hr_mid - lr_up).abs().mean().item()),
        "lr_temporal_l1": motion,
        "lr_temporal_structure_l1": structural_motion,
        "lr_luma_temporal_l1": luma_motion,
        "lr_temporal_spike_ratio": spike_ratio,
    }


def _tensor_to_pil(frame: torch.Tensor, size: int) -> Image.Image:
    array = (
        frame.detach().float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    )
    image = Image.fromarray(array, mode="RGB")
    return image.resize((size, size), resample=Image.Resampling.LANCZOS)


def _contact_sheet(
    path: Path,
    title: str,
    indices: Sequence[int],
    full: DeterministicTripletViewDataset,
    metrics_by_index: Mapping[int, Mapping[str, object]],
    *,
    cell_size: int = 160,
) -> None:
    if not indices:
        return
    columns = 2
    label_height = 44
    cell_width = cell_size * 2
    cell_height = cell_size + label_height
    rows = math.ceil(len(indices) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title + "  (left: bicubic LR, right: HR)", fill="black")
    for slot, index in enumerate(indices):
        sample = full[int(index)]
        lr = sample["lr"]
        hr = sample["hr"]
        assert isinstance(lr, torch.Tensor) and isinstance(hr, torch.Tensor)
        middle = int(lr.shape[0] // 2)
        lr_up = F.interpolate(
            lr[middle].unsqueeze(0),
            size=tuple(int(value) for value in hr[middle].shape[-2:]),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0, 1)
        pair = Image.new("RGB", (cell_width, cell_size), "white")
        pair.paste(_tensor_to_pil(lr_up, cell_size), (0, 0))
        pair.paste(_tensor_to_pil(hr[middle], cell_size), (cell_size, 0))
        x = (slot % columns) * cell_width
        y = 24 + (slot // columns) * cell_height
        canvas.paste(pair, (x, y))
        row = metrics_by_index[int(index)]
        label = (
            f"{row['record_uid']} v{row['view_index']}\n"
            f"HRhf={row['hr_highpass_l1']:.4f} gap={row['recoverable_highpass_gap']:.4f} "
            f"motion={row['lr_temporal_l1']:.4f}"
        )
        draw.text((x + 4, y + cell_size + 3), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def _temporal_strip_sheet(
    path: Path,
    title: str,
    indices: Sequence[int],
    full: DeterministicTripletViewDataset,
    metrics_by_index: Mapping[int, Mapping[str, object]],
    *,
    frame_positions: Sequence[int] = tuple(range(13)),
    cell_size: int = 80,
) -> None:
    """Show several bicubic-LR frames per candidate to review motion semantics."""
    if not indices:
        return
    label_width = 220
    row_height = cell_size + 6
    positions = [int(value) for value in frame_positions]
    canvas = Image.new(
        "RGB",
        (label_width + cell_size * len(positions), 28 + row_height * len(indices)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title + "  (bicubic LR temporal strip)", fill="black")
    for row_index, index in enumerate(indices):
        sample = full[int(index)]
        lr = sample["lr"]
        hr = sample["hr"]
        if not isinstance(lr, torch.Tensor) or not isinstance(hr, torch.Tensor):
            raise TypeError("Dataset sample must contain tensor lr/hr clips")
        y = 28 + row_index * row_height
        metrics = metrics_by_index[int(index)]
        label = (
            f"{metrics['record_uid']} v{metrics['view_index']}\n"
            f"raw={metrics['lr_temporal_l1']:.4f} struct={metrics['lr_temporal_structure_l1']:.4f} "
            f"luma={metrics['lr_luma_temporal_l1']:.4f} spike={metrics['lr_temporal_spike_ratio']:.2f}"
        )
        draw.text((4, y + 4), label, fill="black")
        for column, position in enumerate(positions):
            if position < 0 or position >= int(lr.shape[0]):
                continue
            up = F.interpolate(
                lr[position].unsqueeze(0),
                size=tuple(int(value) for value in hr[position].shape[-2:]),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0, 1)
            canvas.paste(
                _tensor_to_pil(up, cell_size),
                (label_width + column * cell_size, y),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def _motion_context_sheet(
    path: Path,
    title: str,
    indices: Sequence[int],
    base: TripletVideoDataset,
    metrics_by_index: Mapping[int, Mapping[str, object]],
    *,
    cell_width: int = 120,
) -> None:
    """Show all 13 consecutive full LR frames with the training crop marked."""
    if not indices:
        return
    rows: list[tuple[Mapping[str, object], list[Image.Image]]] = []
    cell_height = round(cell_width * 720 / 1280)
    for index in indices:
        metrics = metrics_by_index[int(index)]
        record = base.records[int(metrics["record_index"])]
        start = int(metrics["temporal_start"])
        top, left, crop_h, crop_w = (int(value) for value in metrics["crop_box_lr"])
        frames: list[Image.Image] = []
        for position in range(start, min(start + base.clip_length, record.frame_count)):
            with Image.open(record.lr_paths[position]) as image:
                image = image.convert("RGB")
                source_w, source_h = image.size
                thumb = image.resize((cell_width, cell_height), resample=Image.Resampling.BILINEAR)
            draw = ImageDraw.Draw(thumb)
            x0 = round(left * cell_width / source_w)
            y0 = round(top * cell_height / source_h)
            x1 = round((left + crop_w) * cell_width / source_w)
            y1 = round((top + crop_h) * cell_height / source_h)
            draw.rectangle((x0, y0, x1, y1), outline="white", width=2)
            draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), outline="black", width=1)
            frames.append(thumb)
        rows.append((metrics, frames))

    label_width = 250
    row_height = cell_height + 8
    columns = max((len(frames) for _, frames in rows), default=0)
    canvas = Image.new(
        "RGB",
        (label_width + cell_width * columns, 28 + row_height * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title + "  (full LR context; box = actual 128x128 training crop)", fill="black")
    for row_index, (metrics, frames) in enumerate(rows):
        y = 28 + row_index * row_height
        draw.text(
            (4, y + 3),
            (
                f"{metrics['record_uid']} v{metrics['view_index']}\n"
                f"struct={metrics['lr_temporal_structure_l1']:.4f} "
                f"luma={metrics['lr_luma_temporal_l1']:.4f} "
                f"spike={metrics['lr_temporal_spike_ratio']:.2f}"
            ),
            fill="black",
        )
        for column, frame in enumerate(frames):
            canvas.paste(frame, (label_width + column * cell_width, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def _representatives(rows: Sequence[Mapping[str, object]], count: int) -> dict[str, list[int]]:
    if not rows or count <= 0:
        return {}
    q_detail = float(np.percentile([float(row["hr_highpass_l1"]) for row in rows], 50))
    high_detail = [row for row in rows if float(row["hr_highpass_l1"]) >= q_detail]
    low_detail = sorted(rows, key=lambda row: float(row["hr_highpass_l1"]))[:count]
    recoverable = sorted(
        high_detail,
        key=lambda row: float(row["recoverable_highpass_gap"]),
        reverse=True,
    )[:count]
    detail_motion = sorted(
        high_detail,
        key=lambda row: (
            float(row["lr_temporal_l1"]),
            float(row["hr_highpass_l1"]),
        ),
        reverse=True,
    )[:count]
    structural_pool = [
        row for row in high_detail
        if float(row["lr_temporal_spike_ratio"]) <= 4.0
    ]
    detail_structural_motion = sorted(
        structural_pool,
        key=lambda row: (
            float(row["lr_temporal_structure_l1"]),
            float(row["recoverable_highpass_gap"]),
        ),
        reverse=True,
    )[:count]
    return {
        "low_hr_detail_candidates": [int(row["distillation_index"]) for row in low_detail],
        "recoverable_detail_candidates": [int(row["distillation_index"]) for row in recoverable],
        "detail_motion_candidates": [int(row["distillation_index"]) for row in detail_motion],
        "detail_structural_motion_candidates": [
            int(row["distillation_index"]) for row in detail_structural_motion
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-profile-views", type=int, default=512)
    parser.add_argument("--profile-seed", type=int, default=20260922)
    parser.add_argument("--representatives-per-group", type=int, default=8)
    parser.add_argument("--verify-paths", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_config_path = args.run_config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Audit output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    run_config = _read_json(run_config_path)
    cache_path = run_config.get("teacher_cache")
    if not isinstance(cache_path, str) or not cache_path:
        raise ValueError("run_config is missing teacher_cache")
    cache = TeacherVelocityCache(Path(cache_path).expanduser().resolve())
    config = _resolve_training_config(run_config, cache)
    manifests = [Path(path) for path in config["manifests"]]

    base = TripletVideoDataset(
        manifests,
        split=str(config["split"]),
        training=True,
        clip_length=int(config["clip_length"]),
        crop_size=int(config["crop_size"]),
        scale=int(config["scale"]),
        load_hq=False,
        horizontal_flip_probability=float(config["horizontal_flip_probability"]),
        vertical_flip_probability=float(config["vertical_flip_probability"]),
        path_root=Path(str(config["path_root"])),
        verify_paths=args.verify_paths,
    )
    full = DeterministicTripletViewDataset(
        base,
        views_per_record=int(config["views_per_record"]),
        view_seed=int(config["view_seed"]),
    )
    cache.validate_dataset(
        manifests=manifests,
        split=str(config["split"]),
        clip_length=int(config["clip_length"]),
        crop_size=int(config["crop_size"]),
        scale=int(config["scale"]),
        views_per_record=int(config["views_per_record"]),
        view_seed=int(config["view_seed"]),
        horizontal_flip_probability=float(config["horizontal_flip_probability"]),
        vertical_flip_probability=float(config["vertical_flip_probability"]),
        dataset_length=len(full),
    )

    cached_indices = list(cache_selected_indices(cache.metadata))
    source_uids = [record_source_uid(record) for record in base.records]
    source_records: dict[str, list[int]] = defaultdict(list)
    for record_index, source_uid in enumerate(source_uids):
        source_records[source_uid].append(record_index)

    lr_sizes = [_image_size(record.lr_paths[0]) for record in base.records]
    hr_sizes = [_image_size(record.hr_paths[0]) for record in base.records]
    view_specs = [_view_spec(base, full, index, lr_sizes) for index in cached_indices]
    for spec in view_specs:
        spec["source_uid"] = source_uids[int(spec["record_index"])]

    record_view_specs: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for spec in view_specs:
        record_view_specs[int(spec["record_index"])].append(spec)
    record_overlap_rows: list[dict[str, object]] = []
    for record_index, specs in sorted(record_view_specs.items()):
        spatial: list[float] = []
        temporal: list[float] = []
        for left in range(len(specs)):
            for right in range(left + 1, len(specs)):
                spatial.append(_box_iou(specs[left]["crop_box_lr"], specs[right]["crop_box_lr"]))
                temporal.append(
                    _interval_iou(
                        int(specs[left]["temporal_start"]),
                        int(config["clip_length"]),
                        int(specs[right]["temporal_start"]),
                        int(config["clip_length"]),
                    )
                )
        record_overlap_rows.append(
            {
                "record_index": record_index,
                "record_uid": f"{base.records[record_index].variant}:{base.records[record_index].sample_id}",
                "source_uid": source_uids[record_index],
                "cached_views": len(specs),
                "mean_pairwise_spatial_iou": _pairwise_mean(spatial),
                "mean_pairwise_temporal_iou": _pairwise_mean(temporal),
            }
        )

    profile_indices = _source_balanced_subset(
        cached_indices,
        source_uids,
        int(config["views_per_record"]),
        int(args.max_profile_views),
        int(args.profile_seed),
    )
    spec_by_index = {int(row["distillation_index"]): row for row in view_specs}
    profile_rows: list[dict[str, object]] = []
    for position, index in enumerate(profile_indices, start=1):
        sample = full[index]
        metrics = _profile_view(sample)
        row = {**spec_by_index[index], **metrics}
        profile_rows.append(row)
        if position == 1 or position % 50 == 0 or position == len(profile_indices):
            print(f"profiled {position}/{len(profile_indices)} deterministic views", flush=True)

    sample_id_records: dict[str, list[int]] = defaultdict(list)
    for record_index, record in enumerate(base.records):
        sample_id_records[record.sample_id].append(record_index)
    cross_variant_sample_ids = []
    repeated_sample_ids = []
    for sample_id, record_indices in sorted(sample_id_records.items()):
        if len(record_indices) > 1:
            repeated_sample_ids.append(sample_id)
        variants = sorted({base.records[index].variant for index in record_indices})
        if len(variants) > 1:
            cross_variant_sample_ids.append(sample_id)

    source_histogram = Counter(len(indices) for indices in source_records.values())
    variant_counts = Counter(record.variant for record in base.records)
    manifest_counts = Counter(record.source_manifest for record in base.records)
    size_pairs = Counter(
        f"LR {lr[1]}x{lr[0]} -> HR {hr[1]}x{hr[0]}" for lr, hr in zip(lr_sizes, hr_sizes)
    )
    frame_counts = [record.frame_count for record in base.records]
    duplicate_examples = []
    for source_uid, record_indices in sorted(
        source_records.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        if len(record_indices) <= 1:
            continue
        duplicate_examples.append(
            {
                "source_uid": source_uid,
                "records": [
                    {
                        "record_uid": f"{base.records[index].variant}:{base.records[index].sample_id}",
                        "source_manifest": base.records[index].source_manifest,
                        "hr_first": base.records[index].hr_paths[0],
                    }
                    for index in record_indices
                ],
            }
        )
        if len(duplicate_examples) >= 20:
            break

    metric_summary = {
        key: _quantiles([float(row[key]) for row in profile_rows])
        for key in (
            "hr_highpass_l1",
            "lr_bicubic_highpass_l1",
            "recoverable_highpass_gap",
            "hr_vs_bicubic_lr_mae",
            "lr_temporal_l1",
            "lr_temporal_structure_l1",
            "lr_luma_temporal_l1",
            "lr_temporal_spike_ratio",
        )
    }
    representatives = _representatives(profile_rows, int(args.representatives_per_group))
    metrics_by_index = {int(row["distillation_index"]): row for row in profile_rows}
    for name, indices in representatives.items():
        _contact_sheet(
            output / f"{name}.jpg",
            name.replace("_", " "),
            indices,
            full,
            metrics_by_index,
        )

    motion_indices = representatives.get("detail_motion_candidates", [])
    _temporal_strip_sheet(
        output / "detail_motion_temporal_strips.jpg",
        "detail motion candidates",
        motion_indices,
        full,
        metrics_by_index,
    )
    structural_motion_indices = representatives.get("detail_structural_motion_candidates", [])
    _temporal_strip_sheet(
        output / "detail_structural_motion_temporal_strips.jpg",
        "detail structural motion candidates",
        structural_motion_indices,
        full,
        metrics_by_index,
    )
    _motion_context_sheet(
        output / "detail_structural_motion_context.jpg",
        "detail structural motion candidates",
        structural_motion_indices,
        base,
        metrics_by_index,
    )

    summary = {
        "kind": "swiftvr_training_data_view_audit_v1",
        "run_config": str(run_config_path),
        "teacher_cache": str(cache.root),
        "training_config": config,
        "cache_identity_validation": "PASS",
        "records": {
            "eligible_record_count": len(base.records),
            "cached_view_count": len(cached_indices),
            "full_deterministic_view_count": len(full),
            "unique_hr_source_count": len(source_records),
            "duplicate_hr_source_count": sum(len(indices) > 1 for indices in source_records.values()),
            "records_per_hr_source_histogram": {
                str(key): value for key, value in sorted(source_histogram.items())
            },
            "variant_counts": dict(sorted(variant_counts.items())),
            "manifest_record_counts": dict(sorted(manifest_counts.items())),
            "frame_count": _quantiles(frame_counts),
            "first_frame_size_pairs": dict(sorted(size_pairs.items())),
            "duplicate_source_examples": duplicate_examples,
        },
        "sample_id_alias_check": {
            "unique_sample_id_count": len(sample_id_records),
            "repeated_sample_id_count": len(repeated_sample_ids),
            "cross_variant_sample_id_count": len(cross_variant_sample_ids),
            "cross_variant_sample_id_examples": cross_variant_sample_ids[:40],
            "note": (
                "Matching sample_id across variants is only a naming-level alias signal. "
                "It does not prove identical source pixels, but it can reveal plain/text "
                "records that the path-based source_uid treats as distinct."
            ),
        },
        "manifest_field_counts_for_split": _manifest_fields(manifests, str(config["split"])),
        "view_redundancy": {
            "mean_pairwise_spatial_iou": _quantiles(
                [
                    float(row["mean_pairwise_spatial_iou"])
                    for row in record_overlap_rows
                    if row["mean_pairwise_spatial_iou"] is not None
                ]
            ),
            "mean_pairwise_temporal_iou": _quantiles(
                [
                    float(row["mean_pairwise_temporal_iou"])
                    for row in record_overlap_rows
                    if row["mean_pairwise_temporal_iou"] is not None
                ]
            ),
        },
        "content_profile": {
            "selection": "source-balanced over cached deterministic views",
            "profile_seed": int(args.profile_seed),
            "profiled_views": len(profile_rows),
            "metrics": metric_summary,
            "representatives": representatives,
            "interpretation_note": (
                "These are diagnostics, not automatic quality labels. Low HR high-pass energy "
                "does not prove optical defocus, and a large HR/LR detail gap can also contain "
                "noise, compression, or sharpening artifacts. Review the contact sheets before "
                "using any score for D1 sampling."
            ),
        },
        "known_lineage_gaps": [
            "frame-sequence manifests do not preserve original-video FPS/PTS or extraction cadence",
            "manifest schema does not preserve a canonical original-video/source ID beyond resolved HR frame paths",
            "manifest schema does not preserve degradation recipe/version or codec/noise/blur parameters",
            "HR dimensions alone cannot determine whether a source was natively captured at that resolution or pre-upscaled",
        ],
        "source_identity_method": "resolved_hr_frame_paths_sha256_v1",
    }

    _write_json(output / "summary.json", summary)
    _write_jsonl(output / "view_specs.jsonl", view_specs)
    _write_jsonl(output / "record_view_overlap.jsonl", record_overlap_rows)
    _write_jsonl(output / "profiled_views.jsonl", profile_rows)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
