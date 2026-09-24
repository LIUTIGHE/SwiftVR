#!/usr/bin/env python3
"""Audit materialized UltraVideo LR views against reconstructed clean HQ centers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT, ROOT / "tools"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from tools.materialize_ultravideo_lr_bundles import (
    _batch_to_numpy,
    _clean_hq_crop,
    _decord_reader,
)
from tools.profile_triplet_degradation import _pair_metrics, _quantiles


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


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_lr_center(row: dict[str, object]) -> np.ndarray:
    with safe_open(str(row["bundle_path"]), framework="np", device="cpu") as handle:
        tensor = handle.get_tensor(str(row["bundle_key"]))
    if tensor.ndim != 4:
        raise ValueError("Expected [T,C,H,W] LR tensor")
    center = tensor[len(tensor) // 2]
    return np.transpose(center, (1, 2, 0)).astype(np.uint8, copy=False)


def _canonical_geometry_from_raw(raw: np.ndarray, target_width: int = 3840) -> tuple[int, int]:
    height, width = raw.shape[:2]
    if width > int(target_width):
        scale = float(target_width) / float(width)
        width = int(target_width)
        height = max(1, round(height * scale))
    width -= width % 3
    height -= height % 3
    if width <= 0 or height <= 0:
        raise ValueError("Canonical dimensions collapsed after divisibility crop")
    return int(width), int(height)


def _clean_center(row: dict[str, object]) -> np.ndarray:
    positions = [int(value) for value in row["frame_indices"]]
    position = positions[len(positions) // 2]
    reader = _decord_reader(str(row["raw_video"]))
    raw = _batch_to_numpy(reader.get_batch([position]))[0]

    adapted = dict(row)
    if "canonical_hr_width" not in adapted or "canonical_hr_height" not in adapted:
        width, height = _canonical_geometry_from_raw(raw)
        adapted["canonical_hr_width"] = width
        adapted["canonical_hr_height"] = height
    if "crop_box_hq" not in adapted:
        crop_size = int(adapted.get("crop_size", 0))
        if crop_size <= 0:
            raise ValueError("Legacy materialized row lacks positive crop_size")
        adapted["crop_box_hq"] = [
            int(adapted["crop_top"]),
            int(adapted["crop_left"]),
            crop_size,
            crop_size,
        ]
    return np.asarray(_clean_hq_crop(raw, adapted), dtype=np.uint8)


def _severity(row: dict[str, object]) -> float:
    degradation = row.get("degradation")
    if not isinstance(degradation, dict):
        raise ValueError("Materialized row lacks degradation metadata")
    return float(degradation["severity"])


def _sample_by_severity(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    ordered = sorted(rows, key=_severity)
    actual = min(int(count), len(ordered))
    if actual <= 0:
        return []
    if actual == 1:
        return [ordered[len(ordered) // 2]]
    indices = sorted(
        {
            int(round(index * (len(ordered) - 1) / (actual - 1)))
            for index in range(actual)
        }
    )
    return [ordered[index] for index in indices]


def _contact_sheet(
    samples: list[tuple[dict[str, object], np.ndarray, np.ndarray, dict[str, float]]],
    path: Path,
) -> None:
    scale = 2
    tile_w = 128 * scale * 2
    tile_h = 128 * scale + 42
    columns = 2
    rows = int(math.ceil(len(samples) / columns))
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (row, hq, lr, metrics) in enumerate(samples):
        x = (index % columns) * tile_w
        y = (index // columns) * tile_h
        hq_image = Image.fromarray(hq).resize((256, 256), Image.Resampling.NEAREST)
        lr_image = Image.fromarray(lr).resize((256, 256), Image.Resampling.NEAREST)
        sheet.paste(hq_image, (x, y))
        sheet.paste(lr_image, (x + 256, y))
        draw.text(
            (x + 4, y + 260),
            f"HQ | LR   sev={_severity(row):.3f} PSNR={metrics['hq_lr_psnr']:.2f} "
            f"HF={metrics['highpass_retention_ratio']:.3f}",
            fill="black",
        )
        draw.text(
            (x + 4, y + 278),
            f"{row['sample_id']} view={row['view_index']} {row.get('selection_category','')}",
            fill="black",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-samples", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_jsonl(args.manifest.expanduser().resolve())
    if not rows:
        raise RuntimeError("Materialized manifest is empty")

    metrics_rows: list[dict[str, float]] = []
    for row in rows:
        hq = _clean_center(row)
        lr = _load_lr_center(row)
        metrics_rows.append(_pair_metrics(hq, lr))

    names = (
        "hq_lr_psnr",
        "hq_lr_mae_01",
        "hq_lr_ssim_gray_global",
        "mean_luma_abs_shift",
        "highpass_retention_ratio",
        "highpass_delta",
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "kind": "ultravideo_lr_materialization_audit_v1",
        "manifest": str(args.manifest.expanduser().resolve()),
        "sample_count": len(rows),
        "severity": _quantiles([_severity(row) for row in rows]),
        "metrics": {
            name: _quantiles([float(item[name]) for item in metrics_rows])
            for name in names
        },
        "notes": [
            "Metrics use only the center frame of each fixed 13-frame view.",
            "This is a sanity audit of the new broad degradation support, not a legacy-histogram matching objective.",
            "Older smoke manifests without canonical geometry are reconstructed with the same 3840-wide planner rule.",
        ],
    }
    _write_json(output / "summary.json", summary)

    selected = _sample_by_severity(rows, int(args.visual_samples))
    row_to_metrics = {
        int(row["materialized_index"]): metrics
        for row, metrics in zip(rows, metrics_rows)
    }
    visual_items = []
    for row in selected:
        hq = _clean_center(row)
        lr = _load_lr_center(row)
        visual_items.append(
            (
                row,
                hq,
                lr,
                row_to_metrics[int(row["materialized_index"])],
            )
        )
    _contact_sheet(visual_items, output / "degradation_contact_sheet.jpg")
    print(json.dumps(summary, indent=2), flush=True)
    print(str(output / "degradation_contact_sheet.jpg"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
