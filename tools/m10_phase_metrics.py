"""Small CPU-only metrics shared by the val13 and offset-start diagnostics.

Phase is relative to a reset decode origin, never absolute source_id modulo 4.
HF uses the same sigma1/5x5 spatial Gaussian as M10, on RGB in [0,1].
No flow, model, network access, or automatic sharpness/winner selection.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


def decoder_phase(local_frame: int, trim: int = 3) -> int:
    if local_frame < 0 or trim < 0:
        raise ValueError("Frame and trim must be non-negative")
    return (int(local_frame) + int(trim)) % 4


def high_pass(image: np.ndarray) -> np.ndarray:
    """[H,W,3] float RGB -> spatial Gaussian residual; no temporal filtering."""
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 3 or min(x.shape[:2]) < 3:
        raise ValueError("Expected [H,W,3], H,W >= 3")
    k = np.exp(-0.5 * np.arange(-2, 3, dtype=np.float64) ** 2)
    k = (k / k.sum()).astype(np.float32)
    h, w = x.shape[:2]
    padded = np.pad(x, ((0, 0), (2, 2), (0, 0)), mode="reflect")
    low = sum(k[i] * padded[:, i:i + w] for i in range(5))
    padded = np.pad(low, ((2, 2), (0, 0), (0, 0)), mode="reflect")
    low = sum(k[i] * padded[i:i + h] for i in range(5))
    return x - low


def mean_abs(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x), dtype=np.float64))


def mse(x: np.ndarray) -> float:
    return float(np.mean(np.square(x), dtype=np.float64))


def compare_frames(a: np.ndarray, b: np.ndarray, ha=None, hb=None) -> dict:
    if a.shape != b.shape:
        raise ValueError(f"Comparison shape mismatch: {a.shape} vs {b.shape}")
    error = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    ha = high_pass(a) if ha is None else ha
    hb = high_pass(b) if hb is None else hb
    return {"rgb_mae": mean_abs(error), "rgb_mse": mse(error),
            "hf_mae": mean_abs(ha - hb)}


def aggregate(rows: list[dict], phase_key: str = "phase") -> dict:
    """Frame-weighted means, not mean dB. Temporal values exclude missing pairs."""
    def group(selected):
        result = {"frames": len(selected)}
        for key in ("rgb_mae", "rgb_mse", "hf_mae", "hf_temporal_mae"):
            values = [float(r[key]) for r in selected if r.get(key) is not None]
            result[key] = sum(values) / len(values) if values else None
            if key == "hf_temporal_mae":
                result["temporal_pairs"] = len(values)
        m = result["rgb_mse"]
        # JSON cannot represent infinity. Null for exact/empty, disambiguated.
        result["rgb_exact"] = m == 0.0 if m is not None else None
        result["psnr_db"] = -10.0 * math.log10(m) if m is not None and m > 0 else None
        return result
    return {"all": group(rows), "by_phase": {
        str(p): group([r for r in rows if int(r[phase_key]) == p]) for p in range(4)
    }}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def val_phase_rows(methods, *, sample_index: int, identity: dict) -> list[dict]:
    """Evaluate float model outputs (before PNG quantization), one view at a time."""
    videos = {name: video[0].detach().float().cpu().numpy().transpose(0, 2, 3, 1)
              for name, video in methods}
    length = len(videos["GT"])
    if any(v.shape != videos["GT"].shape for v in videos.values()):
        raise ValueError("All val13 methods must have identical dimensions")
    source_ids = identity["frame_indices"]
    if len(source_ids) != length:
        raise ValueError("Identity/output frame counts differ")
    pairs = [(name, "GT") for name in videos if name not in ("GT", "LQ-up")]
    pairs += [("M8A+Orig", "StageA+Orig"), ("M8A+M9A1", "M8A+Orig"),
              ("M8A+M9A1", "StageA+Orig")]
    if "StageA+M9A1" in videos:
        pairs.append(("StageA+M9A1", "StageA+Orig"))
    if "M8A+D76" in videos:
        pairs.append(("M8A+D76", "M8A+Orig"))
    rows, previous = [], {}
    for t in range(length):
        hf = {name: high_pass(video[t]) for name, video in videos.items()}
        for name, ref in pairs:
            error = hf[name] - hf[ref]
            old = previous.get((name, ref))
            rows.append({
                "sample": sample_index, "record_uid": identity.get("record_uid", ""),
                "local_frame": t, "source_frame_index": int(source_ids[t]),
                "phase": decoder_phase(t), "is_first_frame": t == 0,
                "pair": f"{name} vs {ref}",
                **compare_frames(videos[name][t], videos[ref][t], hf[name], hf[ref]),
                "hf_temporal_mae": mean_abs(error - old) if old is not None else None,
            })
            previous[(name, ref)] = error
    return rows


def write_val_phase_report(root: Path, rows: list[dict]) -> None:
    summary = {}
    for pair in dict.fromkeys(r["pair"] for r in rows):
        selected = [r for r in rows if r["pair"] == pair]
        # 13-frame views: exclude extra phase3 at t0 -> 3 frames/phase.
        balanced = [dict(r) for r in selected if r["local_frame"] >= 1]
        for r in balanced:
            if r["local_frame"] == 1:
                r["hf_temporal_mae"] = None  # do not include pair crossing t0
        summary[pair] = {"all_frames": aggregate(selected),
                         "exclude_first_frame": aggregate(balanced)}
    write_csv(root / "phase_per_frame.csv", rows)
    write_json(root / "phase_report.json", {
        "kind": "m10_val13_phase_diagnostic", "phase_formula": "(local_frame + 3) % 4",
        "range": "RGB [0,1], clamped float outputs before PNG quantization",
        "note": "GT errors are diagnostic; teacher matching cannot detect shared teacher flicker. "
                "Exclude-first is phase-count balancing, NOT proof of streaming steady state. "
                "Temporal errors are unwarped and assigned to the destination phase.",
        "pairs": summary,
    })
