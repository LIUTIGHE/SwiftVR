#!/usr/bin/env python3
"""Trace where spatial phase locking appears inside the Stage-B1 Tiny Decoder.

This is an isolated read-only diagnostic. It never changes model weights, caches,
training code, or inference code. For one or more TinyConditionalDecoder
checkpoints it:

1. captures the concatenated decoder input and the output of every spatial
   reconstruction layer;
2. measures 2x2 phase-locked activation imbalance after removing each
   feature-map/channel spatial DC component;
3. audits the final 12 pre-PixelShuffle channels as RGB x four subpixel phases
   using PyTorch PixelShuffle(2) channel ordering; and
4. reports the visible Tiny-ReAE and Tiny-GT 2x2 residual basis amplitudes.

The main use is to locate the first decoder stage where a persistent checkerboard
signature increases, and to separate an early upsampling-path artifact from a
final subpixel-head phase imbalance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.reae import TGrow
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "decoder"


def _parse_decoder_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        raw_name, raw_path = value.split("=", 1)
        return _safe_name(raw_name), Path(raw_path).expanduser()
    path = Path(value).expanduser()
    return _safe_name(path.name), path


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated integers")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tiny-decoder",
        type=_parse_decoder_spec,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat to compare multiple Tiny checkpoints on identical samples.",
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--verify-paths", action="store_true")
    return parser


def _move_pixels(batch: Mapping[str, object], device: torch.device, dtype: torch.dtype):
    result = dict(batch)
    for key in ("lr", "hr"):
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, dtype=dtype, non_blocking=True)
    return result


def _basis_from_phase2(phase: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return 2x2 orthogonal basis coefficients from [...,2,2] phase means."""
    if tuple(phase.shape[-2:]) != (2, 2):
        raise ValueError(f"expected [...,2,2], got {tuple(phase.shape)}")
    p00 = phase[..., 0, 0]
    p01 = phase[..., 0, 1]
    p10 = phase[..., 1, 0]
    p11 = phase[..., 1, 1]
    return {
        "dc": (p00 + p01 + p10 + p11) / 4.0,
        "horizontal": (p00 - p01 + p10 - p11) / 4.0,
        "vertical": (p00 + p01 - p10 - p11) / 4.0,
        "checkerboard": (p00 - p01 - p10 + p11) / 4.0,
    }


def _feature_phase2_values(feature: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Per-map/channel phase coefficients after subtracting each spatial mean.

    Input is [N,C,H,W]. The returned basis tensors are [N,C]. ``feature_rms`` is
    also [N,C] and measures the spatially centered feature energy used for the
    normalized phase-locking ratios.
    """
    if feature.ndim != 4:
        raise ValueError(f"feature must be [N,C,H,W], got {tuple(feature.shape)}")
    if feature.shape[-2] < 2 or feature.shape[-1] < 2:
        raise ValueError("feature spatial size must be at least 2x2")
    x = feature.detach().float()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    phase = torch.stack(
        [
            torch.stack(
                [x[..., y::2, xoff::2].mean(dim=(-2, -1)) for xoff in range(2)],
                dim=-1,
            )
            for y in range(2)
        ],
        dim=-2,
    )
    rms = x.square().mean(dim=(-2, -1)).sqrt()
    return _basis_from_phase2(phase), rms


def _residual_phase2_values(residual: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-video-frame/RGB 2x2 basis values for residual [B,T,3,H,W]."""
    if residual.ndim != 5 or int(residual.shape[2]) != 3:
        raise ValueError(f"residual must be [B,T,3,H,W], got {tuple(residual.shape)}")
    x = residual.detach().float()
    phase = torch.stack(
        [
            torch.stack(
                [x[..., y::2, xoff::2].mean(dim=(-2, -1)) for xoff in range(2)],
                dim=-1,
            )
            for y in range(2)
        ],
        dim=-2,
    )
    return _basis_from_phase2(phase)


def _prepixel_phase_values(prepixel: torch.Tensor, patch_size: int = 2) -> dict[str, torch.Tensor]:
    """Audit pre-PixelShuffle channels using PyTorch channel ordering.

    For patch_size=2, input [N,12,H,W] is interpreted as
    [N, RGB=3, phase=4, H, W], where phase index is i*2+j and maps to output
    spatial phase (i,j). Returns basis tensors [N,3] from per-phase spatial means.
    """
    if prepixel.ndim != 4:
        raise ValueError(f"prepixel must be [N,C,H,W], got {tuple(prepixel.shape)}")
    r = int(patch_size)
    if r != 2:
        raise ValueError("this diagnostic currently expects PixelShuffle patch_size=2")
    expected = 3 * r * r
    if int(prepixel.shape[1]) != expected:
        raise ValueError(f"expected {expected} pre-PixelShuffle channels, got {prepixel.shape[1]}")
    x = prepixel.detach().float().reshape(prepixel.shape[0], 3, r * r, *prepixel.shape[-2:])
    phase_means = x.mean(dim=(-2, -1)).reshape(prepixel.shape[0], 3, r, r)
    return _basis_from_phase2(phase_means)


@dataclass
class _ScalarAccumulator:
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_signed: float = 0.0
    count: int = 0

    def update(self, value: torch.Tensor) -> None:
        x = value.detach().double().cpu().reshape(-1)
        self.sum_abs += float(x.abs().sum().item())
        self.sum_sq += float(x.square().sum().item())
        self.sum_signed += float(x.sum().item())
        self.count += int(x.numel())

    def summary(self) -> dict[str, float | int]:
        denom = max(self.count, 1)
        return {
            "count": self.count,
            "mean": self.sum_signed / denom,
            "mean_abs": self.sum_abs / denom,
            "rms": math.sqrt(max(self.sum_sq / denom, 0.0)),
        }


class _FeatureAccumulator:
    def __init__(self) -> None:
        self.basis = {name: _ScalarAccumulator() for name in ("dc", "horizontal", "vertical", "checkerboard")}
        self.energy = _ScalarAccumulator()
        self.calls = 0
        self.shape: tuple[int, ...] | None = None

    def update(self, feature: torch.Tensor) -> None:
        values, rms = _feature_phase2_values(feature)
        for name, tensor in values.items():
            self.basis[name].update(tensor)
        self.energy.update(rms)
        self.calls += 1
        self.shape = tuple(int(v) for v in feature.shape[1:])

    def summary(self) -> dict[str, object]:
        basis = {name: acc.summary() for name, acc in self.basis.items()}
        feature_rms = self.energy.summary()
        denom = max(float(feature_rms["rms"]), 1e-12)
        return {
            "calls": self.calls,
            "last_chw": list(self.shape or ()),
            "feature_rms": feature_rms,
            "basis": basis,
            "normalized_rms": {
                name: float(basis[name]["rms"]) / denom
                for name in ("horizontal", "vertical", "checkerboard")
            },
        }


class _BasisAccumulator:
    def __init__(self) -> None:
        self.basis = {name: _ScalarAccumulator() for name in ("dc", "horizontal", "vertical", "checkerboard")}

    def update(self, values: Mapping[str, torch.Tensor]) -> None:
        for name, tensor in values.items():
            self.basis[name].update(tensor)

    def summary(self) -> dict[str, object]:
        return {name: acc.summary() for name, acc in self.basis.items()}


def _semantic_layer_names(model: TinyConditionalDecoder) -> dict[int, str]:
    """Name the frozen Stage-B1 decoder layout and fail loudly if it changes."""
    layers = list(model.decoder)
    if len(layers) != 21:
        raise ValueError(f"Expected 21 Tiny decoder layers, got {len(layers)}")
    expected_upsamples = [index for index, layer in enumerate(layers) if isinstance(layer, nn.Upsample)]
    expected_tgrow = [index for index, layer in enumerate(layers) if isinstance(layer, TGrow)]
    if expected_upsamples != [5, 10, 15] or expected_tgrow != [6, 11, 16]:
        raise ValueError(
            "Tiny decoder layer layout changed; expected Upsample=[5,10,15] and "
            f"TGrow=[6,11,16], got Upsample={expected_upsamples}, TGrow={expected_tgrow}"
        )
    return {
        1: "input_conv",
        2: "input_relu",
        4: "stage0_blocks_end",
        5: "upsample1_nearest",
        6: "upsample1_tgrow",
        7: "upsample1_transition_conv",
        9: "stage1_blocks_end",
        10: "upsample2_nearest",
        11: "upsample2_tgrow",
        12: "upsample2_transition_conv",
        14: "stage2_blocks_end",
        15: "upsample3_nearest",
        16: "upsample3_tgrow",
        17: "upsample3_transition_conv",
        18: "stage3_blocks_end",
        19: "output_relu",
        20: "pre_pixelshuffle_conv12",
    }


class _LayerTrace:
    def __init__(self, model: TinyConditionalDecoder) -> None:
        self.model = model
        self.names = _semantic_layer_names(model)
        self.features: dict[str, _FeatureAccumulator] = {
            "decoder_concat_input": _FeatureAccumulator(),
            **{name: _FeatureAccumulator() for name in self.names.values()},
        }
        self.prepixel = _BasisAccumulator()
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

        def pre_hook(_module, inputs):
            hidden = inputs[0]
            if isinstance(hidden, torch.Tensor):
                self.features["decoder_concat_input"].update(hidden)

        # _apply_video_stack invokes each Sequential child directly, so a hook on
        # the Sequential container itself would never fire. Layer 0 (Clamp) sees
        # the flattened concat input before any learned decoder operation.
        self.handles.append(model.decoder[0].register_forward_pre_hook(pre_hook))
        for index, name in self.names.items():
            layer = model.decoder[index]

            def hook(_module, _inputs, output, *, layer_name=name, layer_index=index):
                if not isinstance(output, torch.Tensor):
                    return
                self.features[layer_name].update(output)
                if layer_index == 20:
                    self.prepixel.update(_prepixel_phase_values(output, model.patch_size))

            self.handles.append(layer.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self) -> dict[str, object]:
        return {
            "layers": {name: acc.summary() for name, acc in self.features.items()},
            "pre_pixelshuffle_rgb_subpixel_basis": self.prepixel.summary(),
        }


def _build_dataset(args: argparse.Namespace, cache: TinyDecoderLatentCache):
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
        path_root=args.path_root.expanduser().resolve(),
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
        positions = tuple(range(len(cached_indices)))
    else:
        positions = tuple(args.sample_indices)
        for position in positions:
            if position >= len(cached_indices):
                raise IndexError(f"sample index {position} exceeds cache sample count {len(cached_indices)}")
    selected = [cached_indices[position] for position in positions]
    return Subset(views, selected), positions


def _run_decoder(
    *,
    name: str,
    root: Path,
    reae: ReAE,
    cache: TinyDecoderLatentCache,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    model = TinyConditionalDecoder.from_pretrained(root, device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trace = _LayerTrace(model)
    tiny_reae = _BasisAccumulator()
    tiny_gt = _BasisAccumulator()
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    samples = 0

    try:
        with torch.inference_mode():
            for batch_cpu in loader:
                moved = _move_pixels(dict(batch_cpu), device, dtype)
                prepared = prepare_training_batch(moved)
                lq_input = prepared["lq_input"]
                target = prepared["target"]
                if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
                    raise TypeError("Validation batch is missing lq_input/target")
                z_sr = cache.load_batch(batch_cpu, device=device, dtype=dtype)
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype if autocast_enabled else torch.float32,
                    enabled=autocast_enabled,
                ):
                    teacher = decode_reae_clip(
                        reae, z_sr, output_frames=int(target.shape[1]), clamp=False
                    )
                    prediction = model(
                        z_sr, lq_input, output_frames=int(target.shape[1]), clamp=False
                    )
                tiny_reae.update(_residual_phase2_values(prediction - teacher))
                tiny_gt.update(_residual_phase2_values(prediction - target))
                samples += int(target.shape[0])
    finally:
        trace.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = trace.summary()
    result.update(
        {
            "name": name,
            "checkpoint": str(root),
            "samples": samples,
            "visible_residual_basis": {
                "tiny_minus_reae": tiny_reae.summary(),
                "tiny_minus_gt": tiny_gt.summary(),
            },
        }
    )
    return result


def _print_summary(result: Mapping[str, object]) -> None:
    name = str(result["name"])
    layers = result["layers"]
    print(f"[{name}] layer phase trace", flush=True)
    for layer_name, payload in layers.items():
        checker = float(payload["basis"]["checkerboard"]["rms"])
        norm = payload["normalized_rms"]
        chw = payload["last_chw"]
        print(
            f"  {layer_name:28s} CHW={chw} "
            f"checker_rms={checker:.6g} checker_norm={float(norm['checkerboard']):.6g} "
            f"h_norm={float(norm['horizontal']):.6g} v_norm={float(norm['vertical']):.6g}",
            flush=True,
        )
    pre = result["pre_pixelshuffle_rgb_subpixel_basis"]
    print(
        "  pre_pixelshuffle basis(mean_abs/rms): "
        f"checker={float(pre['checkerboard']['mean_abs']):.6g}/{float(pre['checkerboard']['rms']):.6g} "
        f"horizontal={float(pre['horizontal']['mean_abs']):.6g}/{float(pre['horizontal']['rms']):.6g} "
        f"vertical={float(pre['vertical']['mean_abs']):.6g}/{float(pre['vertical']['rms']):.6g}",
        flush=True,
    )
    visible = result["visible_residual_basis"]
    for key in ("tiny_minus_reae", "tiny_minus_gt"):
        basis = visible[key]
        print(
            f"  {key} p2(mean_abs): "
            f"checker={float(basis['checkerboard']['mean_abs']):.6g} "
            f"horizontal={float(basis['horizontal']['mean_abs']):.6g} "
            f"vertical={float(basis['vertical']['mean_abs']):.6g}",
            flush=True,
        )


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.clip_length <= 0 or args.crop_size <= 0 or args.scale <= 0:
        raise ValueError("batch-size, clip-length, crop-size and scale must be positive")
    if args.clip_length % 4 != 1:
        raise ValueError("SwiftVR diagnostic clips must satisfy T=4k+1")
    names = [name for name, _ in args.tiny_decoder]
    if len(names) != len(set(names)):
        raise ValueError(f"Tiny decoder names must be unique, got {names}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = DTYPES[args.dtype]
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = TinyDecoderLatentCache(args.val_cache)
    dataset, positions = _build_dataset(args, cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    base = args.base_checkpoint.expanduser().resolve()
    reae = ReAE(str(base / args.reae_filename)).to(device=device, dtype=dtype).eval()
    for parameter in reae.parameters():
        parameter.requires_grad_(False)

    report: dict[str, object] = {
        "diagnostic": "tiny_decoder_layer_phase_trace",
        "base_checkpoint": str(base),
        "val_cache": str(cache.root),
        "sample_positions": list(positions),
        "dtype": args.dtype,
        "decoders": {},
    }
    for name, root in args.tiny_decoder:
        resolved = root.expanduser().resolve()
        result = _run_decoder(
            name=name,
            root=resolved,
            reae=reae,
            cache=cache,
            loader=loader,
            device=device,
            dtype=dtype,
        )
        report["decoders"][name] = result
        _print_summary(result)

    path = output / "layer_phase_summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[saved] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
