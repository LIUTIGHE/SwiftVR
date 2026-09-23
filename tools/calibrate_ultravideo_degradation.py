#!/usr/bin/env python3
"""Calibrate a deterministic synthetic UltraVideo degradation against legacy triplets.

The tool decodes a small source-balanced subset of the selected UltraVideo pilot,
constructs canonical HR and clean 3x-downsampled HQ frames in memory, applies
several clip-consistent candidate degradations, and compares their realized
HQ->LR PSNR / high-frequency-retention quantiles to an empirical legacy profile.

This is a calibration diagnostic only. It does not write training triplets and
does not claim to reproduce the unknown historical degradation parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for search_root in (ROOT, TOOLS):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from tools.profile_triplet_degradation import _pair_metrics, _quantiles


PRESETS: dict[str, dict[str, float]] = {
    "light": {
        "blur_sigma_min": 0.10,
        "blur_sigma_max": 0.90,
        "resize_min": 0.78,
        "noise_std_max_255": 1.25,
        "jpeg_quality_min": 76.0,
        "jpeg_quality_max": 96.0,
    },
    "base": {
        "blur_sigma_min": 0.15,
        "blur_sigma_max": 1.45,
        "resize_min": 0.62,
        "noise_std_max_255": 2.00,
        "jpeg_quality_min": 62.0,
        "jpeg_quality_max": 95.0,
    },
    "strong": {
        "blur_sigma_min": 0.25,
        "blur_sigma_max": 2.10,
        "resize_min": 0.48,
        "noise_std_max_255": 3.00,
        "jpeg_quality_min": 48.0,
        "jpeg_quality_max": 93.0,
    },
}

TARGET_METRICS = ("hq_lr_psnr", "highpass_retention_ratio")
TARGET_QUANTILES = ("p10", "median", "p90")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object in {path}")
            rows.append(value)
    return rows


def _stable_unit(seed: int, *parts: object) -> float:
    payload = ":".join([str(int(seed)), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _select_source_balanced(
    rows: Sequence[Mapping[str, object]],
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        uid = str(row.get("source_group_uid", ""))
        if uid:
            grouped[uid].append(dict(row))
    keys = sorted(
        grouped,
        key=lambda uid: hashlib.sha256(f"{int(seed)}:{uid}".encode("utf-8")).hexdigest(),
    )
    selected: list[dict[str, object]] = []
    for uid in keys[: min(int(count), len(keys))]:
        candidates = sorted(grouped[uid], key=lambda row: str(row.get("clip_id", "")))
        position = int(_stable_unit(seed, uid, "clip") * len(candidates))
        position = min(position, len(candidates) - 1)
        selected.append(candidates[position])
    return selected


def _decord_reader(path: str):
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("decord is required for UltraVideo calibration") from exc
    return decord.VideoReader(path, ctx=decord.cpu(0))


def _batch_to_numpy(value) -> np.ndarray:
    """Convert Decord/torch/numpy batch outputs to a CPU NumPy array."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return np.asarray(value)


def _sample_positions(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0 or count <= 0:
        return []
    actual = min(frame_count, count)
    if actual == 1:
        return [frame_count // 2]
    return sorted(
        {
            int(round((slot + 1) * (frame_count - 1) / (actual + 1)))
            for slot in range(actual)
        }
    )


def _canonical_hr(frame: np.ndarray, target_width: int) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    width, height = image.size
    if width > target_width:
        target_height = max(1, round(height * target_width / width))
        image = image.resize(
            (int(target_width), int(target_height)),
            resample=Image.Resampling.LANCZOS,
        )
    width, height = image.size
    width = width - width % 3
    height = height - height % 3
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid canonical size after divisibility crop: {image.size}")
    if image.size != (width, height):
        image = image.crop((0, 0, width, height))
    return image


def _hq_from_hr(hr: Image.Image) -> Image.Image:
    width, height = hr.size
    if width % 3 or height % 3:
        raise ValueError(f"HR size must be divisible by 3, got {hr.size}")
    return hr.resize((width // 3, height // 3), resample=Image.Resampling.BOX)


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _apply_degradation(
    hq: Image.Image,
    *,
    preset: Mapping[str, float],
    severity: float,
    noise_seed: int,
) -> Image.Image:
    value = min(max(float(severity), 0.0), 1.0)
    blur_sigma = float(preset["blur_sigma_min"]) + value * (
        float(preset["blur_sigma_max"]) - float(preset["blur_sigma_min"])
    )
    image = hq.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    width, height = image.size
    resize_min = float(preset["resize_min"])
    resize_scale = 1.0 - value * (1.0 - resize_min)
    if resize_scale < 0.999:
        small = (
            max(1, round(width * resize_scale)),
            max(1, round(height * resize_scale)),
        )
        image = image.resize(small, resample=Image.Resampling.BICUBIC)
        image = image.resize((width, height), resample=Image.Resampling.BICUBIC)

    noise_std = value * float(preset["noise_std_max_255"])
    if noise_std > 0:
        rng = np.random.default_rng(int(noise_seed))
        array = np.asarray(image, dtype=np.float32)
        array += rng.normal(0.0, noise_std, size=array.shape).astype(np.float32)
        image = Image.fromarray(np.clip(array, 0, 255).round().astype(np.uint8), mode="RGB")

    quality_max = float(preset["jpeg_quality_max"])
    quality_min = float(preset["jpeg_quality_min"])
    quality = round(quality_max - value * (quality_max - quality_min))
    return _jpeg_roundtrip(image, int(quality))


def _metric_summary(rows: Sequence[Mapping[str, float]]) -> dict[str, object]:
    names = (
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
            for name in names
        },
    }


def _target_summary(legacy: Mapping[str, object]) -> Mapping[str, object]:
    overall = legacy.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError("Legacy profile lacks overall")
    metrics = overall.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Legacy profile lacks overall.metrics")
    return metrics


def _distance(
    generated: Mapping[str, object],
    target: Mapping[str, object],
) -> dict[str, object]:
    generated_metrics = generated["metrics"]
    if not isinstance(generated_metrics, Mapping):
        raise TypeError("Generated metric summary is malformed")
    components: list[dict[str, object]] = []
    squared = 0.0
    for metric in TARGET_METRICS:
        generated_metric = generated_metrics.get(metric)
        target_metric = target.get(metric)
        if not isinstance(generated_metric, Mapping) or not isinstance(target_metric, Mapping):
            raise ValueError(f"Missing calibration metric {metric}")
        target_span = max(
            abs(float(target_metric["p90"]) - float(target_metric["p10"])),
            1e-6,
        )
        for quantile in TARGET_QUANTILES:
            actual = float(generated_metric[quantile])
            desired = float(target_metric[quantile])
            normalized = (actual - desired) / target_span
            squared += normalized * normalized
            components.append(
                {
                    "metric": metric,
                    "quantile": quantile,
                    "generated": actual,
                    "target": desired,
                    "normalized_error": normalized,
                }
            )
    return {
        "score": float(math.sqrt(squared / max(len(components), 1))),
        "components": components,
    }


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-train", type=Path, required=True)
    parser.add_argument("--legacy-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=32)
    parser.add_argument("--frames-per-source", type=int, default=3)
    parser.add_argument("--target-hr-width", type=int, default=3840)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--preset",
        action="append",
        choices=tuple(sorted(PRESETS)),
        default=None,
        help="Candidate preset; repeat to select several. Defaults to light/base/strong.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source_count <= 0 or args.frames_per_source <= 0 or args.target_hr_width <= 0:
        raise ValueError("source-count, frames-per-source and target-hr-width must be positive")
    pilot = _read_jsonl(args.pilot_train.expanduser().resolve())
    legacy = _read_json(args.legacy_profile.expanduser().resolve())
    target = _target_summary(legacy)
    selected = _select_source_balanced(pilot, int(args.source_count), int(args.seed))
    presets = args.preset or ["light", "base", "strong"]

    candidate_rows: dict[str, list[dict[str, float]]] = {name: [] for name in presets}
    selected_sources: list[dict[str, object]] = []
    for source_index, row in enumerate(selected, start=1):
        path = str(row.get("raw_video", ""))
        if not path:
            raise ValueError("Pilot row is missing raw_video")
        reader = _decord_reader(path)
        positions = _sample_positions(len(reader), int(args.frames_per_source))
        if not positions:
            raise ValueError(f"No decodable frames: {path}")
        frames = _batch_to_numpy(reader.get_batch(positions))
        source_record = {
            "clip_id": row.get("clip_id"),
            "source_group_uid": row.get("source_group_uid"),
            "raw_video": path,
            "frame_positions": positions,
        }
        selected_sources.append(source_record)

        severity = _stable_unit(args.seed, row.get("source_group_uid"), row.get("clip_id"), "severity")
        for local_index, frame in enumerate(frames):
            hr = _canonical_hr(frame, int(args.target_hr_width))
            hq = _hq_from_hr(hr)
            hq_array = np.asarray(hq, dtype=np.uint8)
            for preset_name in presets:
                lr = _apply_degradation(
                    hq,
                    preset=PRESETS[preset_name],
                    severity=severity,
                    noise_seed=int(
                        _stable_unit(
                            args.seed,
                            row.get("clip_id"),
                            local_index,
                            preset_name,
                            "noise",
                        )
                        * (2**32 - 1)
                    ),
                )
                candidate_rows[preset_name].append(
                    _pair_metrics(hq_array, np.asarray(lr, dtype=np.uint8))
                )
        print(f"calibrated {source_index}/{len(selected)} sources", flush=True)

    candidates: dict[str, object] = {}
    ranking: list[tuple[float, str]] = []
    for name in presets:
        summary = _metric_summary(candidate_rows[name])
        distance = _distance(summary, target)
        candidates[name] = {
            "parameters": PRESETS[name],
            "summary": summary,
            "distance": distance,
        }
        ranking.append((float(distance["score"]), name))
    ranking.sort()

    report = {
        "kind": "ultravideo_degradation_calibration_v1",
        "pilot_train": str(args.pilot_train.expanduser().resolve()),
        "legacy_profile": str(args.legacy_profile.expanduser().resolve()),
        "source_count": len(selected),
        "frames_per_source": int(args.frames_per_source),
        "target_hr_width": int(args.target_hr_width),
        "seed": int(args.seed),
        "target_metrics": {
            metric: {quantile: target[metric][quantile] for quantile in TARGET_QUANTILES}
            for metric in TARGET_METRICS
        },
        "candidates": candidates,
        "ranking": [
            {"preset": name, "score": score}
            for score, name in ranking
        ],
        "selected_sources": selected_sources,
        "notes": [
            "All degradation parameters are shared at clip level through one deterministic severity value.",
            "This first calibration intentionally excludes video codec compression so spatial/frequency degradation can be isolated.",
            "The candidate ranking uses only PSNR and high-frequency-retention p10/median/p90; luma/SSIM remain diagnostics.",
            "No training data or teacher cache is written by this calibration.",
        ],
    }
    _write_json(args.output.expanduser().resolve(), report)
    print(
        json.dumps(
            {
                "ranking": report["ranking"],
                "target_metrics": report["target_metrics"],
                "candidate_summaries": {
                    name: candidates[name]["summary"] for name in presets
                },
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
