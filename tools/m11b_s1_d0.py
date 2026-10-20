#!/usr/bin/env python3
"""M11B-S1-D0: phase-specific decoder diagnostics without model re-inference.

Subcommands:
  frames: compare lossless, same-origin A1 and Original Decoder PNG sequences.
  tgrow: inspect stride-2 TGrow effective two-phase kernels from safetensors.

These are diagnostic measurements, NOT proof of a causal flicker mechanism.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = (int(part) for part in value.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height in output pixels") from exc
    if min(x, y) < 0 or min(w, h) <= 0:
        raise argparse.ArgumentTypeError("ROI must have nonnegative origin and positive size")
    return x, y, w, h


def _png_list(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Expected PNG directory: {path}")
    files = list(path.glob("*.png"))
    if not files:
        raise ValueError(f"No PNG files: {path}")
    def key(p: Path):
        return (0, int(p.stem)) if p.stem.isdigit() else (1, p.name)
    return sorted(files, key=key)


def _prepare_folder(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _crop_rgb(path: Path, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    with Image.open(path) as picture:
        picture = picture.convert("RGB")
        if roi is not None:
            x, y, w, h = roi
            if x + w > picture.width or y + h > picture.height:
                raise ValueError(f"ROI {roi} exceeds source {path}: {picture.size}")
            picture = picture.crop((x, y, x + w, y + h))
        return np.asarray(picture, dtype=np.float32) / 255.0


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152
            + rgb[..., 2] * 0.0722)


def _blur(array: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    return cv2.GaussianBlur(
        array, (0, 0), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )


def _phase_means(rows: list[dict], fields: list[str]) -> dict[str, dict]:
    report = {}
    for p in range(4):
        subset = [r for r in rows if r["phase"] == p]
        report[str(p)] = {
            "count": len(subset),
            **{field: (float(np.mean([r[field] for r in subset]))
                       if subset and field in subset[0] else None)
               for field in fields},
        }
    return report


def _phase_spectrum(summary: dict, field: str) -> dict:
    means = [summary[str(p)].get(field) for p in range(4)]
    if any(v is None for v in means):
        return {"available": False}
    vector = np.asarray(means, dtype=np.float64)
    spectrum = np.fft.rfft(vector - vector.mean()) / 4.0
    return {
        "available": True,
        "four_phase_magnitude": float(abs(spectrum[1])),
        "two_phase_magnitude": float(abs(spectrum[2])),
        "definition": "DFT of four phase-conditioned frame means; NOT a motion-compensated flicker score",
    }


def _snapshot(path: Path, original: np.ndarray, a1: np.ndarray, index: int, phase: int) -> None:
    # Both sources are displayed at identical native-pixel geometry.
    images = [np.uint8(np.clip(im * 255.0 + 0.5, 0, 255)) for im in (original, a1)]
    height, width = images[0].shape[:2]
    canvas = Image.new("RGB", (width * 2, height + 32))
    draw = ImageDraw.Draw(canvas)
    for col, (name, frame) in enumerate(zip(("M11B + Original", "M11B + A1"), images)):
        canvas.paste(Image.fromarray(frame), (col * width, 32))
        draw.text((col * width + 8, 8), f"{name}  output={index} phase={phase}", fill="white")
    canvas.save(path)


def diagnose_frames(args: argparse.Namespace) -> dict:
    a1_files = _png_list(args.a1)
    orig_files = _png_list(args.original)
    if [p.name for p in a1_files] != [p.name for p in orig_files]:
        raise ValueError("A1 and Original PNG filenames differ; refusing implicit frame realignment")
    gt_files = None
    if args.gt is not None:
        gt_files = _png_list(args.gt)
        if [p.name for p in a1_files] != [p.name for p in gt_files]:
            raise ValueError("GT PNG filenames differ; refusing implicit frame realignment")
    total = len(a1_files)
    end = total - 1 if args.end is None else args.end
    if args.start < 0 or end < args.start or end >= total:
        raise ValueError(f"Requested inclusive range [{args.start}, {end}] outside [0, {total-1}]")
    if end - args.start + 1 < 8:
        raise ValueError("Use at least eight consecutive aligned frames for phase analysis")

    selected = set(args.snapshots)
    fields = ["luma_delta_a1_minus_original", "lf_delta_mae", "hf_delta_mae",
              "hf_energy_a1", "hf_energy_original", "luma_mean_a1", "luma_mean_original"]
    if gt_files is not None:
        fields += ["lf_a1_gt_mae", "lf_original_gt_mae",
                   "hf_a1_gt_mae", "hf_original_gt_mae",
                   "luma_a1_gt_bias", "luma_original_gt_bias"]

    rows = []
    _prepare_folder(args.output_dir)
    for i in range(args.start, end + 1):
        orig = _crop_rgb(orig_files[i], args.roi)
        a1 = _crop_rgb(a1_files[i], args.roi)
        if orig.shape != a1.shape:
            raise ValueError(f"Frame {i}: A1/Original resolution mismatch")
        ya, yo = _luma(a1), _luma(orig)
        low_a, low_o = _blur(ya, args.low_sigma), _blur(yo, args.low_sigma)
        high_a, high_o = ya - _blur(ya, args.high_sigma), yo - _blur(yo, args.high_sigma)
        phase = (i + args.phase_offset) % 4
        row = {
            "output_frame": i, "phase": phase, "filename": a1_files[i].name,
            "luma_delta_a1_minus_original": float((ya - yo).mean()),
            "lf_delta_mae": float(np.abs(low_a - low_o).mean()),
            "hf_delta_mae": float(np.abs(high_a - high_o).mean()),
            "hf_energy_a1": float(np.abs(high_a).mean()),
            "hf_energy_original": float(np.abs(high_o).mean()),
            "luma_mean_a1": float(ya.mean()), "luma_mean_original": float(yo.mean()),
        }
        if gt_files is not None:
            gt = _crop_rgb(gt_files[i], args.roi)
            if gt.shape != orig.shape:
                raise ValueError(f"Frame {i}: GT resolution mismatch")
            yg = _luma(gt)
            low_g = _blur(yg, args.low_sigma)
            high_g = yg - _blur(yg, args.high_sigma)
            row.update(
                lf_a1_gt_mae=float(np.abs(low_a - low_g).mean()),
                lf_original_gt_mae=float(np.abs(low_o - low_g).mean()),
                hf_a1_gt_mae=float(np.abs(high_a - high_g).mean()),
                hf_original_gt_mae=float(np.abs(high_o - high_g).mean()),
                luma_a1_gt_bias=float((ya - yg).mean()),
                luma_original_gt_bias=float((yo - yg).mean()),
            )
        rows.append(row)
        if i in selected:
            _snapshot(args.output_dir / f"native_roi_{i:06d}_p{phase}.png", orig, a1, i, phase)

    with (args.output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_phase = _phase_means(rows, fields)
    report = {
        "kind": "M11B-S1-D0-frames-v1",
        "input": {"a1": str(args.a1.resolve()), "original": str(args.original.resolve()),
                  "gt": None if args.gt is None else str(args.gt.resolve())},
        "output_frame_range_inclusive": [args.start, end],
        "source_total_frames": total,
        "phase_formula": f"(original_output_index + {args.phase_offset}) % 4",
        "roi_native_output_pixels": args.roi,
        "by_phase": by_phase,
        "periodicity": {
            "a1_minus_original_luma": _phase_spectrum(by_phase, "luma_delta_a1_minus_original"),
        },
        "interpretation": [
            "The same ordinal PNG filename and source origin are mandatory.",
            "A1-minus-Original measures decoder divergence, not absolute quality.",
            "Highpass energy is NOT a measure of correctness; highpass MAE without GT is only disagreement.",
            "Phase-conditioned differences can reflect scene dynamics: DFT is diagnostic, not causal evidence.",
            "Uncompensated metrics do not establish motion-aligned temporal stability.",
        ],
    }
    if gt_files is not None:
        report["periodicity"]["a1_gt_luma"] = _phase_spectrum(by_phase, "luma_a1_gt_bias")
        report["periodicity"]["original_gt_luma"] = _phase_spectrum(by_phase, "luma_original_gt_bias")
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


_TGROW_KEY = re.compile(r"^decoder\.(\d+)\.conv3d\.weight$")


def effective_tgrow(weight: np.ndarray) -> dict:
    """Exact constant-within-2-samples effective maps for TGrow stride=2.

    padding=1, kernel=3, nearest upsampling duplicates each input frame twice.
    The two outputs use (W_center+W_right) and (W_left+W_center).
    """
    if weight.ndim != 5 or tuple(weight.shape[2:]) != (3, 1, 1):
        raise ValueError(f"Expected TGrow Conv3d [C,C,3,1,1], got {weight.shape}")
    taps = np.asarray(weight, dtype=np.float64)[:, :, :, 0, 0]
    first = taps[:, :, 1] + taps[:, :, 2]
    second = taps[:, :, 0] + taps[:, :, 1]
    denominator = max(float((np.linalg.norm(first) + np.linalg.norm(second)) / 2), 1e-12)
    gain_first, gain_second = first.sum(axis=1), second.sum(axis=1)
    return {
        "channels_out": int(taps.shape[0]), "channels_in": int(taps.shape[1]),
        "relative_transfer_frobenius_gap": float(np.linalg.norm(first - second) / denominator),
        "mean_abs_dc_gain_gap": float(np.abs(gain_first - gain_second).mean()),
        "mean_signed_dc_gain_first": float(gain_first.mean()),
        "mean_signed_dc_gain_second": float(gain_second.mean()),
    }


def read_tgrow(path: Path) -> list[dict]:
    from safetensors import safe_open
    if not path.is_file():
        raise FileNotFoundError(path)
    layers = []
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in sorted(f.keys()):
            if _TGROW_KEY.fullmatch(key):
                matrix = f.get_tensor(key).float().numpy()
                layers.append({"key": key, **effective_tgrow(matrix)})
    if not layers:
        raise ValueError(f"No decoder.<idx>.conv3d.weight TGrow weights found: {path}")
    return layers


def diagnose_tgrow(args: argparse.Namespace) -> dict:
    original = read_tgrow(args.reae)
    a1 = read_tgrow(args.a1_model)
    if len(original) != 2 or len(a1) != 2:
        raise ValueError(f"Expected two stride-2 TGrow stages, got Original={len(original)} A1={len(a1)}")
    _prepare_folder(args.output_dir)
    report = {
        "kind": "M11B-S1-D0-tgrow-v1",
        "original_reae": {"weights": str(args.reae.resolve()), "layers": original},
        "a1": {"weights": str(args.a1_model.resolve()), "layers": a1},
        "scope": "Per-TGrow two-sample local transfer only; ignores intervening MemBlocks, nonlinearities, latent motion, and streaming state.",
        "caution": "Asymmetric weights do not by themselves prove flicker or phase0 causation.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("frames", help="Same-origin native PNG A1-versus-Original phase diagnostics")
    f.add_argument("--a1", type=Path, required=True)
    f.add_argument("--original", type=Path, required=True)
    f.add_argument("--gt", type=Path, default=None, help="Optional aligned HR PNG sequence")
    f.add_argument("--output-dir", type=Path, required=True)
    f.add_argument("--start", type=int, default=0, help="0-based output ordinal")
    f.add_argument("--end", type=int, default=None, help="Inclusive 0-based output ordinal")
    f.add_argument("--phase-offset", type=int, default=3)
    f.add_argument("--roi", type=parse_roi, default=None, help="Native output-pixel x,y,w,h")
    f.add_argument("--low-sigma", type=float, default=8.0)
    f.add_argument("--high-sigma", type=float, default=1.5)
    f.add_argument("--snapshots", type=lambda s: [int(v) for v in s.split(",") if v.strip()], default=[245, 248])
    w = sub.add_parser("tgrow", help="Local two-subphase response of TGrow kernels")
    w.add_argument("--reae", type=Path, required=True, help="base/reae.safetensors")
    w.add_argument("--a1-model", type=Path, required=True, help="A1/model.safetensors")
    w.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.mode == "frames":
        if not (args.low_sigma > 0 and args.high_sigma > 0):
            raise ValueError("Gaussian sigmas must be positive")
        result = diagnose_frames(args)
        print(json.dumps({"kind": result["kind"], "frames": result["output_frame_range_inclusive"],
                          "phase_summary": result["by_phase"], "output": str(args.output_dir)}, indent=2))
    else:
        result = diagnose_tgrow(args)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
