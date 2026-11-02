#!/usr/bin/env python3
"""Diagnose periodic reconstruction artifacts in Stage-B1 Tiny Decoder outputs.

This is an isolated read-only visual/statistical diagnostic. It compares one or
more TinyConditionalDecoder checkpoints on the same deterministic validation
samples and cached z_SR latents. For each decoder it measures spatial phase bias
in the visible residuals

    Tiny - ReAE teacher
    Tiny - GT

at periods 2, 4, 8, and 16 (configurable). ReAE - GT is measured once as a
control. The period-2 analysis additionally reports horizontal, vertical, and
checkerboard basis amplitudes, including per-frame mean-absolute/RMS values, so
a stable alternating bright/dark artifact can be distinguished from ordinary
image content.

The tool never modifies model weights, caches, trainers, or inference code and
never loads the Stage-A DiT.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated integers")
    return result


def _csv_periods(value: str) -> tuple[int, ...]:
    result = _csv_ints(value)
    if any(item < 2 for item in result):
        raise argparse.ArgumentTypeError("periods must be integers >= 2")
    return tuple(sorted(result))


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "decoder"


def _parse_decoder_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        raw_name, raw_path = value.split("=", 1)
        name = _safe_name(raw_name)
        path = Path(raw_path).expanduser()
    else:
        path = Path(value).expanduser()
        name = _safe_name(path.name)
    if not str(path):
        raise argparse.ArgumentTypeError("tiny-decoder path cannot be empty")
    return name, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tiny-decoder",
        type=_parse_decoder_spec,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat to compare multiple checkpoints on exactly the same samples.",
    )
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--views-per-record", type=int, default=1)
    parser.add_argument("--view-seed", type=int, default=9000001)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.0)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.0)
    parser.add_argument(
        "--sample-indices",
        type=_csv_ints,
        default=None,
        help="Cache positions to inspect. Default: all cached validation samples.",
    )
    parser.add_argument("--periods", type=_csv_periods, default=(2, 4, 8, 16))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument(
        "--no-phase-images",
        action="store_true",
        help="Write JSON statistics only; skip phase-map PNGs.",
    )
    return parser


def _move_pixels(batch: dict[str, object], device: torch.device, dtype: torch.dtype):
    result = dict(batch)
    for key in ("lr", "hr"):
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, dtype=dtype, non_blocking=True)
    return result


@dataclass
class _BasisAccumulator:
    total: float = 0.0
    total_abs: float = 0.0
    total_sq: float = 0.0
    count: int = 0

    def update(self, values: torch.Tensor) -> None:
        data = values.detach().double().cpu()
        self.total += float(data.sum().item())
        self.total_abs += float(data.abs().sum().item())
        self.total_sq += float(data.square().sum().item())
        self.count += int(data.numel())

    def finalize(self) -> dict[str, float | int]:
        count = max(self.count, 1)
        return {
            "mean": self.total / count,
            "mean_abs": self.total_abs / count,
            "rms": math.sqrt(max(self.total_sq / count, 0.0)),
            "count": self.count,
        }


class PhaseAccumulator:
    """Aggregate phase-conditioned residual statistics without storing videos."""

    def __init__(self, periods: Iterable[int]) -> None:
        self.periods = tuple(sorted(dict.fromkeys(int(value) for value in periods)))
        self.data: dict[int, dict[str, torch.Tensor]] = {}
        for period in self.periods:
            shape = (period, period)
            self.data[period] = {
                "sum": torch.zeros(shape, dtype=torch.float64),
                "abs_sum": torch.zeros(shape, dtype=torch.float64),
                "sq_sum": torch.zeros(shape, dtype=torch.float64),
                "count": torch.zeros(shape, dtype=torch.float64),
                "rgb_sum": torch.zeros((period, period, 3), dtype=torch.float64),
            }
        self.basis = {
            "dc": _BasisAccumulator(),
            "horizontal": _BasisAccumulator(),
            "vertical": _BasisAccumulator(),
            "checkerboard": _BasisAccumulator(),
        }
        self.frames = 0
        self.elements = 0

    @staticmethod
    def _luma(residual: torch.Tensor) -> torch.Tensor:
        weights = residual.new_tensor(LUMA_WEIGHTS).reshape(1, 1, 3, 1, 1)
        return (residual * weights).sum(dim=2)

    def update(self, residual: torch.Tensor) -> None:
        if residual.ndim != 5 or int(residual.shape[2]) != 3:
            raise ValueError(f"residual must be [B,T,3,H,W], got {tuple(residual.shape)}")
        residual = residual.detach().float()
        luma = self._luma(residual)
        batch, frames, _, height, width = residual.shape
        self.frames += int(batch * frames)
        self.elements += int(residual.numel())

        for period in self.periods:
            usable_h = (int(height) // period) * period
            usable_w = (int(width) // period) * period
            if usable_h <= 0 or usable_w <= 0:
                raise ValueError(
                    f"period={period} exceeds residual geometry {height}x{width}"
                )
            luma_crop = luma[..., :usable_h, :usable_w]
            nh, nw = usable_h // period, usable_w // period
            phase = luma_crop.reshape(batch, frames, nh, period, nw, period)
            # [B,T,nh,p,nw,p] -> aggregate B,T,nh,nw -> [p,p]
            phase_sum = phase.sum(dim=(0, 1, 2, 4)).double().cpu()
            phase_abs = phase.abs().sum(dim=(0, 1, 2, 4)).double().cpu()
            phase_sq = phase.square().sum(dim=(0, 1, 2, 4)).double().cpu()
            count_value = float(batch * frames * nh * nw)

            rgb_crop = residual[..., :usable_h, :usable_w]
            rgb_phase = rgb_crop.reshape(
                batch, frames, 3, nh, period, nw, period
            )
            rgb_sum = rgb_phase.sum(dim=(0, 1, 3, 5)).permute(1, 2, 0).double().cpu()

            current = self.data[period]
            current["sum"] += phase_sum
            current["abs_sum"] += phase_abs
            current["sq_sum"] += phase_sq
            current["count"] += count_value
            current["rgb_sum"] += rgb_sum

            if period == 2:
                per_frame = phase.mean(dim=(2, 4))  # [B,T,2,2]
                g00 = per_frame[..., 0, 0]
                g01 = per_frame[..., 0, 1]
                g10 = per_frame[..., 1, 0]
                g11 = per_frame[..., 1, 1]
                self.basis["dc"].update((g00 + g01 + g10 + g11) * 0.25)
                self.basis["horizontal"].update(
                    (g00 - g01 + g10 - g11) * 0.25
                )
                self.basis["vertical"].update(
                    (g00 + g01 - g10 - g11) * 0.25
                )
                self.basis["checkerboard"].update(
                    (g00 - g01 - g10 + g11) * 0.25
                )

    def finalize(self) -> dict[str, object]:
        period_results: dict[str, object] = {}
        means_by_period: dict[int, torch.Tensor] = {}
        global_rms_by_period: dict[int, float] = {}

        for period in self.periods:
            current = self.data[period]
            count = current["count"].clamp_min(1.0)
            mean = current["sum"] / count
            mean_abs = current["abs_sum"] / count
            rms = torch.sqrt((current["sq_sum"] / count).clamp_min(0.0))
            rgb_mean = current["rgb_sum"] / count.unsqueeze(-1)
            means_by_period[period] = mean
            global_count = float(current["count"].sum().item())
            global_sq = float(current["sq_sum"].sum().item())
            global_rms = math.sqrt(max(global_sq / max(global_count, 1.0), 0.0))
            global_rms_by_period[period] = global_rms

            phase_std = float(mean.std(unbiased=False).item())
            phase_range = float((mean.max() - mean.min()).item())
            period_results[str(period)] = {
                "phase_mean_luma": mean.tolist(),
                "phase_mean_abs_luma": mean_abs.tolist(),
                "phase_rms_luma": rms.tolist(),
                "phase_mean_rgb": rgb_mean.tolist(),
                "phase_mean_std": phase_std,
                "phase_mean_range": phase_range,
                "global_residual_rms_luma": global_rms,
                "phase_strength_vs_residual_rms": phase_std / max(global_rms, 1e-12),
                "used_values_per_phase": int(current["count"][0, 0].item()),
            }

        for period in self.periods:
            result = period_results[str(period)]
            parent = period // 2
            if period % 2 == 0 and parent in means_by_period:
                mean = means_by_period[period]
                parent_mean = means_by_period[parent]
                y = torch.arange(period) % parent
                x = torch.arange(period) % parent
                tiled_parent = parent_mean[y[:, None], x[None, :]]
                novel = mean - tiled_parent
                novel_std = float(novel.std(unbiased=False).item())
                result["new_phase_std_beyond_parent"] = novel_std
                result["new_phase_strength_vs_residual_rms"] = novel_std / max(
                    global_rms_by_period[period], 1e-12
                )
            else:
                result["new_phase_std_beyond_parent"] = float(
                    means_by_period[period].std(unbiased=False).item()
                )
                result["new_phase_strength_vs_residual_rms"] = result[
                    "phase_strength_vs_residual_rms"
                ]

        basis = (
            {name: accumulator.finalize() for name, accumulator in self.basis.items()}
            if 2 in self.periods
            else {}
        )
        return {
            "frames": self.frames,
            "elements": self.elements,
            "periods": period_results,
            "period2_basis": basis,
        }


def _phase_map_from_result(result: dict[str, object], period: int) -> np.ndarray:
    periods = result["periods"]
    if not isinstance(periods, dict):
        raise TypeError("invalid phase result")
    payload = periods[str(period)]
    if not isinstance(payload, dict):
        raise TypeError("invalid period payload")
    return np.asarray(payload["phase_mean_luma"], dtype=np.float64)


def _save_phase_image(grid: np.ndarray, path: Path, *, cell_pixels: int = 40) -> None:
    if grid.ndim != 2:
        raise ValueError("phase grid must be 2D")
    max_abs = float(np.max(np.abs(grid))) if grid.size else 0.0
    if max_abs <= 1e-12:
        normalized = np.full(grid.shape, 127, dtype=np.uint8)
    else:
        normalized = np.clip(127.5 + 127.5 * grid / max_abs, 0, 255).astype(np.uint8)
    image = Image.fromarray(normalized, mode="L")
    width = max(int(grid.shape[1] * cell_pixels), int(grid.shape[1]))
    height = max(int(grid.shape[0] * cell_pixels), int(grid.shape[0]))
    image = image.resize((width, height), resample=Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _compact_console(name: str, residual_name: str, result: dict[str, object]) -> str:
    pieces = [f"[{name}:{residual_name}]"]
    periods = result.get("periods", {})
    if isinstance(periods, dict):
        for period in sorted(int(key) for key in periods):
            payload = periods[str(period)]
            if isinstance(payload, dict):
                pieces.append(
                    f"p{period}:phase_std={float(payload['phase_mean_std']):.6g} "
                    f"new={float(payload['new_phase_std_beyond_parent']):.6g} "
                    f"strength={float(payload['phase_strength_vs_residual_rms']):.4f}"
                )
    basis = result.get("period2_basis", {})
    if isinstance(basis, dict) and "checkerboard" in basis:
        cb = basis["checkerboard"]
        h = basis["horizontal"]
        v = basis["vertical"]
        if isinstance(cb, dict) and isinstance(h, dict) and isinstance(v, dict):
            pieces.append(
                "p2_basis(mean_abs): "
                f"checker={float(cb['mean_abs']):.6g} "
                f"horizontal={float(h['mean_abs']):.6g} "
                f"vertical={float(v['mean_abs']):.6g}"
            )
    return " | ".join(pieces)


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_length <= 0 or args.crop_size <= 0 or args.scale <= 0:
        raise ValueError("clip-length, crop-size and scale must be positive")
    if args.clip_length % 4 != 1:
        raise ValueError("SwiftVR diagnostic clips must satisfy T=4k+1")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for name, probability in (
        ("horizontal-flip-probability", args.horizontal_flip_probability),
        ("vertical-flip-probability", args.vertical_flip_probability),
    ):
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")

    decoder_specs: list[tuple[str, Path]] = list(args.tiny_decoder)
    names = [name for name, _ in decoder_specs]
    if len(set(names)) != len(names):
        raise ValueError(f"tiny-decoder names must be unique, got {names}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = DTYPES[args.dtype]
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)

    base = args.base_checkpoint.expanduser().resolve()
    cache = TinyDecoderLatentCache(args.val_cache)
    path_root = args.path_root.expanduser().resolve()
    base_dataset = TripletVideoDataset(
        args.val_manifest,
        split=args.val_split,
        training=True,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        load_hq=False,
        horizontal_flip_probability=args.horizontal_flip_probability,
        vertical_flip_probability=args.vertical_flip_probability,
        drop_short_sequences=True,
        path_root=path_root,
        verify_paths=args.verify_paths,
    )
    views = DeterministicTripletViewDataset(
        base_dataset,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
    )
    cache.validate_dataset(
        manifests=args.val_manifest,
        split=args.val_split,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
        horizontal_flip_probability=args.horizontal_flip_probability,
        vertical_flip_probability=args.vertical_flip_probability,
        dataset_length=len(views),
    )

    cached_indices = cache.selected_indices()
    if args.sample_indices is None:
        selected_positions = tuple(range(len(cached_indices)))
    else:
        selected_positions = tuple(args.sample_indices)
    for position in selected_positions:
        if position >= len(cached_indices):
            raise IndexError(
                f"sample index {position} exceeds cache sample count {len(cached_indices)}"
            )
    selected_dataset_indices = [cached_indices[position] for position in selected_positions]
    loader = DataLoader(
        Subset(views, selected_dataset_indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    reae = ReAE(str(base / args.reae_filename)).to(device=device, dtype=dtype).eval()
    for parameter in reae.parameters():
        parameter.requires_grad_(False)

    decoders: dict[str, TinyConditionalDecoder] = {}
    decoder_paths: dict[str, str] = {}
    for name, raw_path in decoder_specs:
        path = raw_path.expanduser().resolve()
        model = TinyConditionalDecoder.from_pretrained(path, device=device, dtype=dtype).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        decoders[name] = model
        decoder_paths[name] = str(path)

    periods = tuple(args.periods)
    control = PhaseAccumulator(periods)
    accumulators: dict[str, dict[str, PhaseAccumulator]] = {
        name: {
            "tiny_minus_reae": PhaseAccumulator(periods),
            "tiny_minus_gt": PhaseAccumulator(periods),
        }
        for name in decoders
    }

    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    processed_samples = 0
    with torch.inference_mode():
        for batch_cpu in loader:
            moved = _move_pixels(dict(batch_cpu), device, dtype)
            prepared = prepare_training_batch(moved)
            lq_input = prepared["lq_input"]
            target = prepared["target"]
            if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise TypeError("validation batch is missing lq_input/target")
            z_sr = cache.load_batch(batch_cpu, device=device, dtype=dtype)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype if autocast_enabled else torch.float32,
                enabled=autocast_enabled,
            ):
                teacher = decode_reae_clip(
                    reae,
                    z_sr,
                    output_frames=int(target.shape[1]),
                    clamp=True,
                )
                predictions = {
                    name: decoder(
                        z_sr,
                        lq_input,
                        output_frames=int(target.shape[1]),
                        clamp=True,
                    )
                    for name, decoder in decoders.items()
                }

            target_f = target.float().clamp(0.0, 1.0)
            teacher_f = teacher.float().clamp(0.0, 1.0)
            control.update(teacher_f - target_f)
            for name, prediction in predictions.items():
                prediction_f = prediction.float().clamp(0.0, 1.0)
                accumulators[name]["tiny_minus_reae"].update(prediction_f - teacher_f)
                accumulators[name]["tiny_minus_gt"].update(prediction_f - target_f)
            processed_samples += int(target.shape[0])

    control_result = control.finalize()
    decoder_results: dict[str, dict[str, object]] = {}
    for name, residuals in accumulators.items():
        decoder_results[name] = {
            residual_name: accumulator.finalize()
            for residual_name, accumulator in residuals.items()
        }

    report: dict[str, object] = {
        "diagnostic": "tiny_decoder_spatial_phase_bias_v1",
        "base_checkpoint": str(base),
        "val_cache": str(cache.root),
        "val_manifests": [str(path.expanduser().resolve()) for path in args.val_manifest],
        "decoder_paths": decoder_paths,
        "sample_positions": list(selected_positions),
        "dataset_indices": list(selected_dataset_indices),
        "processed_samples": processed_samples,
        "periods": list(periods),
        "dtype": args.dtype,
        "control_reae_minus_gt": control_result,
        "decoders": decoder_results,
        "interpretation_notes": {
            "phase_mean_std": "Std. dev. of the fixed signed luma residual across spatial phases.",
            "phase_strength_vs_residual_rms": "phase_mean_std divided by total luma residual RMS; larger means more error is locked to a spatial phase.",
            "new_phase_std_beyond_parent": "For p>2, phase bias remaining after subtracting the nested p/2 pattern; helps separate new 4/8/16-period structure from inherited 2-period checkerboard.",
            "period2_checkerboard": "For phase means g00,g01,g10,g11: (g00-g01-g10+g11)/4. Mean-absolute/RMS are also accumulated per frame so sign changes do not cancel.",
            "control": "Compare Tiny-ReAE against ReAE-GT; a strong Tiny-only phase pattern is evidence for decoder reconstruction bias rather than the source latent/teacher alone.",
        },
    }

    (output / "phase_bias_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    if not args.no_phase_images:
        image_root = output / "phase_maps"
        all_results: list[tuple[str, str, dict[str, object]]] = [
            ("control", "reae_minus_gt", control_result)
        ]
        for name, residuals in decoder_results.items():
            for residual_name, result in residuals.items():
                if isinstance(result, dict):
                    all_results.append((name, residual_name, result))
        for name, residual_name, result in all_results:
            for period in periods:
                grid = _phase_map_from_result(result, period)
                _save_phase_image(
                    grid,
                    image_root / name / residual_name / f"period_{period:02d}_mean_luma.png",
                )

    print(_compact_console("control", "reae_minus_gt", control_result), flush=True)
    for name, residuals in decoder_results.items():
        for residual_name, result in residuals.items():
            if isinstance(result, dict):
                print(_compact_console(name, residual_name, result), flush=True)
    print(f"Wrote {output / 'phase_bias_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
