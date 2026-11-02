#!/usr/bin/env python3
"""Profile the empirical HQ->LR degradation distribution of SwiftVR triplets.

This is a read-only calibration tool. It samples aligned frames from existing
triplet manifests and summarizes pixel error, global gray SSIM, luma shift and
simple high-frequency retention. The resulting distributions are intended to
calibrate a documented synthetic degradation pipeline; none of the metrics is an
automatic sample-quality label.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter

from swiftvr.data import read_triplet_manifests


def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _gray01(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    return (
        0.299 * value[..., 0] + 0.587 * value[..., 1] + 0.114 * value[..., 2]
    ) / 255.0


def _highpass_l1(gray01: np.ndarray) -> float:
    image = Image.fromarray(
        np.clip(np.asarray(gray01) * 255.0, 0, 255).round().astype(np.uint8),
        mode="L",
    )
    low = np.asarray(image.filter(ImageFilter.BoxBlur(radius=1)), dtype=np.float32) / 255.0
    return float(np.mean(np.abs(np.asarray(gray01, dtype=np.float32) - low)))


def _pair_metrics(hq: np.ndarray, lr: np.ndarray) -> dict[str, float]:
    if hq.shape != lr.shape:
        raise ValueError(f"HQ/LR shape mismatch: {hq.shape} vs {lr.shape}")
    hq32 = hq.astype(np.float32)
    lr32 = lr.astype(np.float32)
    diff = hq32 - lr32
    mae_255 = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse == 0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))

    hq_gray = _gray01(hq)
    lr_gray = _gray01(lr)
    mean_hq = float(np.mean(hq_gray))
    mean_lr = float(np.mean(lr_gray))
    var_hq = float(np.var(hq_gray))
    var_lr = float(np.var(lr_gray))
    covariance = float(np.mean((hq_gray - mean_hq) * (lr_gray - mean_lr)))
    c1 = 0.01**2
    c2 = 0.03**2
    denominator = (mean_hq**2 + mean_lr**2 + c1) * (var_hq + var_lr + c2)
    if denominator == 0:
        ssim = 1.0 if np.array_equal(hq, lr) else 0.0
    else:
        ssim = (
            (2.0 * mean_hq * mean_lr + c1) * (2.0 * covariance + c2)
        ) / denominator

    hq_hf = _highpass_l1(hq_gray)
    lr_hf = _highpass_l1(lr_gray)
    return {
        "hq_lr_psnr": psnr,
        "hq_lr_mae_01": mae_255 / 255.0,
        "hq_lr_ssim_gray_global": float(ssim),
        "hq_mean_luma": mean_hq,
        "lr_mean_luma": mean_lr,
        "mean_luma_abs_shift": abs(mean_hq - mean_lr),
        "hq_highpass_l1": hq_hf,
        "lr_highpass_l1": lr_hf,
        "highpass_retention_ratio": lr_hf / max(hq_hf, 1e-8),
        "highpass_delta": lr_hf - hq_hf,
    }


def _positions(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0 or count <= 0:
        return []
    actual = min(frame_count, count)
    if actual == 1:
        return [frame_count // 2]
    return sorted(
        {int(round(value)) for value in np.linspace(0, frame_count - 1, actual)}
    )


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    clean = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if clean.size == 0:
        return {
            key: None
            for key in ("min", "p10", "p25", "median", "p75", "p90", "max", "mean")
        }
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


def _summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    metric_names = (
        "hq_lr_psnr",
        "hq_lr_mae_01",
        "hq_lr_ssim_gray_global",
        "mean_luma_abs_shift",
        "hq_highpass_l1",
        "lr_highpass_l1",
        "highpass_retention_ratio",
        "highpass_delta",
    )
    return {
        "frame_pair_count": len(rows),
        "metrics": {
            name: _quantiles([float(row[name]) for row in rows])
            for name in metric_names
        },
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-frames", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-paths", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_frames <= 0 or args.max_records < 0:
        raise ValueError("sample-frames must be positive and max-records non-negative")

    records = read_triplet_manifests(
        args.manifest,
        split=args.split,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
    )
    if args.max_records:
        records = records[: int(args.max_records)]

    rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record_index, record in enumerate(records, start=1):
        for position in _positions(record.frame_count, int(args.sample_frames)):
            hq = _load_rgb(record.hq_paths[position])
            lr = _load_rgb(record.lr_paths[position])
            metrics = _pair_metrics(hq, lr)
            row = {
                "record_index": record_index - 1,
                "record_uid": f"{record.variant}:{record.sample_id}",
                "sample_id": record.sample_id,
                "variant": record.variant,
                "frame_position": int(position),
                "frame_index": int(record.frame_indices[position]),
                **metrics,
            }
            rows.append(row)
            grouped[record.variant].append(row)
        if record_index == 1 or record_index % 100 == 0 or record_index == len(records):
            print(f"profiled {record_index}/{len(records)} records", flush=True)

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "kind": "swiftvr_triplet_degradation_profile_v1",
        "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
        "path_root": str(args.path_root.expanduser().resolve()),
        "split": args.split,
        "record_count": len(records),
        "sample_frames_per_record": int(args.sample_frames),
        "overall": _summarize(rows),
        "by_variant": {
            name: _summarize(group_rows)
            for name, group_rows in sorted(grouped.items())
        },
        "interpretation": [
            "Metrics describe the realized HQ->LR distribution, not the unknown historical degradation parameters.",
            "Highpass retention may exceed 1 when degradation adds noise, ringing or sharpening; treat it as calibration, not quality.",
            "The profile is intended to calibrate a new documented UltraVideo degradation pipeline without claiming exact reproduction of legacy preprocessing.",
        ],
    }
    _write_jsonl(output / "frame_metrics.jsonl", rows)
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
