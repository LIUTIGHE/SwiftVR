#!/usr/bin/env python3
"""Compare spatial phase bias across canonical and resize-conv Tiny decoders.

This is an isolated read-only Stage-B1 diagnostic. It evaluates any mixture of
canonical TinyConditionalDecoder and ResizeConvTinyConditionalDecoder checkpoints
on the same deterministic validation samples and cached z_SR latents. For every
variant it measures Tiny-ReAE and Tiny-GT residual phase bias at periods 2/4/8/16
(or a configurable set), including the parent-subtracted ``new`` component used by
our earlier checkerboard diagnosis. ReAE-GT is measured once as a control.

For canonical PixelShuffle checkpoints only, the tool additionally hooks the final
12-channel convolution and reports the RGB x 4 subpixel-phase basis imbalance.
Resize-conv checkpoints intentionally have no such metric because their shared
32->3 head has no independent PixelShuffle phases.

No model weights, caches, trainers, or inference paths are modified, and the
Stage-A DiT is never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder
from swiftvr.models.tiny_conditional_decoder_resize_conv import (
    ResizeConvTinyConditionalDecoder,
)
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache
from tools.diagnose_tiny_decoder_layer_phase import (
    _BasisAccumulator as PrePixelBasisAccumulator,
    _prepixel_phase_values,
)
from tools.diagnose_tiny_decoder_phase_bias import (
    PhaseAccumulator,
    _compact_console,
    _phase_map_from_result,
    _save_phase_image,
)


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
SUPPORTED_DECODER_CLASSES = {
    "TinyConditionalDecoder": TinyConditionalDecoder,
    "ResizeConvTinyConditionalDecoder": ResizeConvTinyConditionalDecoder,
}


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


def _safe_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise argparse.ArgumentTypeError("decoder label cannot be empty")
    return cleaned


def _parse_decoder_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--decoder expects LABEL=PATH")
    raw_label, raw_path = value.split("=", 1)
    label = _safe_label(raw_label)
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("decoder path cannot be empty")
    return label, Path(raw_path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--decoder",
        type=_parse_decoder_spec,
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat to compare canonical and/or resize-conv Tiny checkpoints.",
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
        help="Write JSON/console statistics only; skip normalized phase-map PNGs.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.clip_length <= 0 or args.crop_size <= 0 or args.scale <= 0:
        raise ValueError("clip-length, crop-size and scale must be positive")
    if args.clip_length % 4 != 1:
        raise ValueError("SwiftVR diagnostic clips must satisfy T=4k+1")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for name, value in (
        ("horizontal-flip-probability", args.horizontal_flip_probability),
        ("vertical-flip-probability", args.vertical_flip_probability),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    labels = [label for label, _ in args.decoder]
    if len(labels) != len(set(labels)):
        raise ValueError(f"decoder labels must be unique, got {labels}")


def _move_pixels(batch: Mapping[str, object], device: torch.device, dtype: torch.dtype):
    result = dict(batch)
    for key in ("lr", "hr"):
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, dtype=dtype, non_blocking=True)
    return result


def _checkpoint_class_name(root: Path) -> str:
    config_path = root.expanduser().resolve() / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    class_name = str(config.get("class_name", ""))
    if class_name not in SUPPORTED_DECODER_CLASSES:
        raise ValueError(
            f"Unsupported Tiny decoder class_name={class_name!r} in {config_path}; "
            f"supported={sorted(SUPPORTED_DECODER_CLASSES)}"
        )
    return class_name


def _load_decoder(root: Path, *, device: torch.device, dtype: torch.dtype):
    resolved = root.expanduser().resolve()
    class_name = _checkpoint_class_name(resolved)
    cls = SUPPORTED_DECODER_CLASSES[class_name]
    model = cls.from_pretrained(resolved, device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, class_name


class _CanonicalPrePixelAudit:
    """Collect final 12-channel RGB/subpixel phase imbalance for canonical Tiny."""

    def __init__(self, model: TinyConditionalDecoder) -> None:
        if type(model) is not TinyConditionalDecoder:
            raise TypeError("pre-PixelShuffle audit is canonical Tiny only")
        if not isinstance(model.decoder[-1], torch.nn.Conv2d):
            raise TypeError("canonical Tiny must end with Conv2d")
        if int(model.decoder[-1].out_channels) != 3 * model.patch_size**2:
            raise ValueError("unexpected canonical pre-PixelShuffle channel count")
        self.accumulator = PrePixelBasisAccumulator()
        self.patch_size = int(model.patch_size)

        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                self.accumulator.update(_prepixel_phase_values(output, self.patch_size))

        self.handle = model.decoder[-1].register_forward_hook(hook)

    def close(self) -> None:
        self.handle.remove()

    def summary(self) -> dict[str, object]:
        return self.accumulator.summary()


def _print_prepixel(label: str, result: Mapping[str, object]) -> None:
    def value(name: str, field: str) -> float:
        payload = result.get(name, {})
        return float(payload.get(field, float("nan"))) if isinstance(payload, Mapping) else float("nan")

    print(
        f"[{label}:pre_pixelshuffle_subpixel] "
        f"checker(mean_abs/rms)={value('checkerboard','mean_abs'):.6g}/{value('checkerboard','rms'):.6g} "
        f"horizontal={value('horizontal','mean_abs'):.6g} "
        f"vertical={value('vertical','mean_abs'):.6g}",
        flush=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = DTYPES[args.dtype]
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

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
    selected_positions = (
        tuple(range(len(cached_indices)))
        if args.sample_indices is None
        else tuple(args.sample_indices)
    )
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

    decoders: dict[str, torch.nn.Module] = {}
    decoder_metadata: dict[str, dict[str, str]] = {}
    prepixel_audits: dict[str, _CanonicalPrePixelAudit] = {}
    for label, raw_path in args.decoder:
        path = raw_path.expanduser().resolve()
        model, class_name = _load_decoder(path, device=device, dtype=dtype)
        decoders[label] = model
        decoder_metadata[label] = {"path": str(path), "class_name": class_name}
        if class_name == "TinyConditionalDecoder":
            prepixel_audits[label] = _CanonicalPrePixelAudit(model)

    periods = tuple(args.periods)
    control = PhaseAccumulator(periods)
    accumulators: dict[str, dict[str, PhaseAccumulator]] = {
        label: {
            "tiny_minus_reae": PhaseAccumulator(periods),
            "tiny_minus_gt": PhaseAccumulator(periods),
        }
        for label in decoders
    }

    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    processed_samples = 0
    try:
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
                        label: decoder(
                            z_sr,
                            lq_input,
                            output_frames=int(target.shape[1]),
                            clamp=True,
                        )
                        for label, decoder in decoders.items()
                    }

                target_f = target.float().clamp(0.0, 1.0)
                teacher_f = teacher.float().clamp(0.0, 1.0)
                control.update(teacher_f - target_f)
                for label, prediction in predictions.items():
                    prediction_f = prediction.float().clamp(0.0, 1.0)
                    accumulators[label]["tiny_minus_reae"].update(
                        prediction_f - teacher_f
                    )
                    accumulators[label]["tiny_minus_gt"].update(
                        prediction_f - target_f
                    )
                processed_samples += int(target.shape[0])
    finally:
        for audit in prepixel_audits.values():
            audit.close()

    control_result = control.finalize()
    decoder_results: dict[str, dict[str, object]] = {}
    for label, residuals in accumulators.items():
        decoder_results[label] = {
            residual_name: accumulator.finalize()
            for residual_name, accumulator in residuals.items()
        }
        if label in prepixel_audits:
            decoder_results[label]["pre_pixelshuffle_rgb_subpixel_basis"] = (
                prepixel_audits[label].summary()
            )

    report: dict[str, object] = {
        "diagnostic": "tiny_decoder_variant_spatial_phase_bias_v1",
        "base_checkpoint": str(base),
        "val_cache": str(cache.root),
        "val_manifests": [str(path.expanduser().resolve()) for path in args.val_manifest],
        "decoders": decoder_metadata,
        "sample_positions": list(selected_positions),
        "dataset_indices": list(selected_dataset_indices),
        "processed_samples": processed_samples,
        "periods": list(periods),
        "dtype": args.dtype,
        "control_reae_minus_gt": control_result,
        "results": decoder_results,
        "interpretation_notes": {
            "new_phase_std_beyond_parent": "For p>2, residual phase structure after subtracting the nested p/2 pattern. This is the primary quantity for testing whether p8/p16 remain after p2 removal.",
            "period2_checkerboard": "Mean-absolute/RMS are accumulated per frame, so alternating sign across frames does not cancel.",
            "canonical_prepixel": "Only canonical PixelShuffle Tiny has four independent RGB subpixel phases. ResizeConv deliberately reports no pre-PixelShuffle phase metric.",
            "phase_images": "Each PNG is normalized independently for visibility; compare JSON/console magnitudes, not PNG contrast, across variants.",
        },
    }
    summary_path = output / "variant_phase_bias_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if not args.no_phase_images:
        image_root = output / "phase_maps"
        all_results: list[tuple[str, str, dict[str, object]]] = [
            ("control", "reae_minus_gt", control_result)
        ]
        for label, residuals in decoder_results.items():
            for residual_name in ("tiny_minus_reae", "tiny_minus_gt"):
                result = residuals.get(residual_name)
                if isinstance(result, dict):
                    all_results.append((label, residual_name, result))
        for label, residual_name, result in all_results:
            for period in periods:
                _save_phase_image(
                    _phase_map_from_result(result, period),
                    image_root / label / residual_name / f"period_{period:02d}_mean_luma.png",
                )

    print(_compact_console("control", "reae_minus_gt", control_result), flush=True)
    for label, residuals in decoder_results.items():
        for residual_name in ("tiny_minus_reae", "tiny_minus_gt"):
            result = residuals[residual_name]
            if isinstance(result, dict):
                print(_compact_console(label, residual_name, result), flush=True)
        prepixel = residuals.get("pre_pixelshuffle_rgb_subpixel_basis")
        if isinstance(prepixel, Mapping):
            _print_prepixel(label, prepixel)
        else:
            print(f"[{label}:pre_pixelshuffle_subpixel] N/A (phase-shared resize-conv head)", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
