#!/usr/bin/env python3
"""Trace spatial phase locking inside ResizeConv Stage-B1 Tiny Decoder variants.

This is an isolated read-only diagnostic for ResizeConvTinyConditionalDecoder
checkpoints. It complements ``diagnose_tiny_decoder_layer_phase.py`` (which is
specific to the canonical PixelShuffle topology) and is intended for controlled
comparisons such as ResizeConv head-only R4 versus decoder-tail recovery E9.

For each checkpoint, the tool captures:

* decoder concat input;
* Decoder-S0/S1/S2/S3 block endpoints;
* all three spatial/temporal transition components;
* the final decoder output ReLU;
* the spatially resized feature immediately before the shared RGB head; and
* the shared RGB output-head response.

At every captured [N,C,H,W] tensor it measures current-resolution 2x2 phase
locking after subtracting each feature-map/channel spatial DC component. This
lets us test whether the strong output-space p8 artifact corresponds to a
persistent current-resolution p2 mode in Decoder-S2 and whether tail recovery
changes that mode. Visible Tiny-ReAE/Tiny-GT p2 residual basis values are also
reported for continuity with the earlier diagnostics.

The diagnostic never changes weights, caches, trainers, or inference code and
never loads the Stage-A DiT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.reae import TGrow
from swiftvr.models.tiny_conditional_decoder_resize_conv import (
    ResizeConvTinyConditionalDecoder,
)
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.forward import decode_reae_clip, prepare_training_batch
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache

# Reuse the already-validated phase estimators/accumulators from the canonical
# layer tracer. Only checkpoint loading and topology hooks differ here.
from tools.diagnose_tiny_decoder_layer_phase import (
    DTYPES,
    _BasisAccumulator,
    _FeatureAccumulator,
    _csv_ints,
    _move_pixels,
    _parse_decoder_spec,
    _residual_phase2_values,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--decoder",
        type=_parse_decoder_spec,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat to compare ResizeConv checkpoints on identical samples.",
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


def _semantic_layer_names(model: ResizeConvTinyConditionalDecoder) -> dict[int, str]:
    """Return semantic names for the formal (2,2,2,1) ResizeConv decoder layout."""
    layers = list(model.decoder)
    if tuple(model.blocks_per_stage) != (2, 2, 2, 1):
        raise ValueError(
            "ResizeConv layer diagnostic expects formal blocks_per_stage=(2,2,2,1), "
            f"got {model.blocks_per_stage}"
        )
    if len(layers) != 20:
        raise ValueError(f"Expected 20 ResizeConv decoder trunk layers, got {len(layers)}")
    upsample_indices = [
        index for index, layer in enumerate(layers) if isinstance(layer, nn.Upsample)
    ]
    tgrow_indices = [
        index for index, layer in enumerate(layers) if isinstance(layer, TGrow)
    ]
    if upsample_indices != [5, 10, 15] or tgrow_indices != [6, 11, 16]:
        raise ValueError(
            "ResizeConv Tiny decoder layout changed; expected Upsample=[5,10,15] "
            "and TGrow=[6,11,16], got "
            f"Upsample={upsample_indices}, TGrow={tgrow_indices}"
        )
    if not isinstance(model.output_head, nn.Conv2d):
        raise TypeError("ResizeConv output_head must be Conv2d")
    if int(model.output_head.out_channels) != 3:
        raise ValueError(
            f"ResizeConv output_head must produce RGB, got {model.output_head.out_channels}"
        )
    return {
        1: "input_conv",
        2: "input_relu",
        4: "decoder_s0_blocks_end",
        5: "transition01_nearest",
        6: "transition01_tgrow",
        7: "transition01_conv",
        9: "decoder_s1_blocks_end",
        10: "transition12_nearest",
        11: "transition12_tgrow",
        12: "transition12_conv",
        14: "decoder_s2_blocks_end",
        15: "transition23_nearest",
        16: "transition23_tgrow",
        17: "transition23_conv",
        18: "decoder_s3_blocks_end",
        19: "decoder_output_relu",
    }


class _ResizeConvLayerTrace:
    def __init__(self, model: ResizeConvTinyConditionalDecoder) -> None:
        self.model = model
        self.names = _semantic_layer_names(model)
        feature_names = [
            "decoder_concat_input",
            *self.names.values(),
            "resize_x2_pre_head",
            "output_head_rgb",
        ]
        self.features = {name: _FeatureAccumulator() for name in feature_names}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

        def decoder_pre_hook(_module, inputs):
            hidden = inputs[0]
            if isinstance(hidden, torch.Tensor):
                self.features["decoder_concat_input"].update(hidden)

        # _apply_video_stack invokes decoder children directly, so hook layer 0.
        self.handles.append(model.decoder[0].register_forward_pre_hook(decoder_pre_hook))

        for index, name in self.names.items():
            layer = model.decoder[index]

            def layer_hook(_module, _inputs, output, *, layer_name=name):
                if isinstance(output, torch.Tensor):
                    self.features[layer_name].update(output)

            self.handles.append(layer.register_forward_hook(layer_hook))

        # ResizeConv.forward performs _resize(flat) and then calls output_head.
        # Therefore output_head's input is exactly the nearest-upsampled x2 feature.
        def head_pre_hook(_module, inputs):
            hidden = inputs[0]
            if isinstance(hidden, torch.Tensor):
                self.features["resize_x2_pre_head"].update(hidden)

        def head_hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                self.features["output_head_rgb"].update(output)

        self.handles.append(model.output_head.register_forward_pre_hook(head_pre_hook))
        self.handles.append(model.output_head.register_forward_hook(head_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self) -> dict[str, object]:
        return {"layers": {name: acc.summary() for name, acc in self.features.items()}}


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
                raise IndexError(
                    f"sample index {position} exceeds cache sample count {len(cached_indices)}"
                )
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
    model = ResizeConvTinyConditionalDecoder.from_pretrained(
        root, device=device, dtype=dtype
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trace = _ResizeConvLayerTrace(model)
    tiny_reae = _BasisAccumulator()
    tiny_gt = _BasisAccumulator()
    autocast_enabled = device.type == "cuda" and dtype in (
        torch.float16,
        torch.bfloat16,
    )
    samples = 0

    try:
        with torch.inference_mode():
            for batch_cpu in loader:
                moved = _move_pixels(dict(batch_cpu), device, dtype)
                prepared = prepare_training_batch(moved)
                lq_input = prepared["lq_input"]
                target = prepared["target"]
                if not isinstance(lq_input, torch.Tensor) or not isinstance(
                    target, torch.Tensor
                ):
                    raise TypeError("Validation batch is missing lq_input/target")
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
                        clamp=False,
                    )
                    prediction = model(
                        z_sr,
                        lq_input,
                        output_frames=int(target.shape[1]),
                        clamp=False,
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


def _layer_scalar(payload: Mapping[str, object], key: str) -> float:
    normalized = payload["normalized_rms"]
    if not isinstance(normalized, Mapping):
        raise TypeError("invalid normalized_rms payload")
    return float(normalized[key])


def _comparison(base: Mapping[str, object], other: Mapping[str, object]) -> dict[str, object]:
    """Compute other/base ratios for current-resolution phase locking per layer."""
    base_layers = base["layers"]
    other_layers = other["layers"]
    if not isinstance(base_layers, Mapping) or not isinstance(other_layers, Mapping):
        raise TypeError("invalid layer summaries")
    if tuple(base_layers) != tuple(other_layers):
        raise ValueError("Compared ResizeConv traces do not share layer names/order")

    layers: dict[str, object] = {}
    for name in base_layers:
        left = base_layers[name]
        right = other_layers[name]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise TypeError(f"invalid layer payload for {name}")
        values = {}
        for basis in ("checkerboard", "horizontal", "vertical"):
            before = _layer_scalar(left, basis)
            after = _layer_scalar(right, basis)
            values[basis] = {
                "base": before,
                "other": after,
                "ratio_other_over_base": after / max(before, 1e-12),
                "relative_change": (after - before) / max(before, 1e-12),
            }
        layers[str(name)] = values
    return {
        "base": str(base["name"]),
        "other": str(other["name"]),
        "layers": layers,
    }


def _print_summary(result: Mapping[str, object]) -> None:
    name = str(result["name"])
    layers = result["layers"]
    if not isinstance(layers, Mapping):
        raise TypeError("invalid layer summary")
    print(f"[{name}] ResizeConv layer phase trace", flush=True)
    for layer_name, raw_payload in layers.items():
        if not isinstance(raw_payload, Mapping):
            continue
        basis = raw_payload["basis"]
        norm = raw_payload["normalized_rms"]
        if not isinstance(basis, Mapping) or not isinstance(norm, Mapping):
            continue
        checker = basis["checkerboard"]
        if not isinstance(checker, Mapping):
            continue
        print(
            f"  {str(layer_name):28s} CHW={raw_payload['last_chw']} "
            f"checker_rms={float(checker['rms']):.6g} "
            f"checker_norm={float(norm['checkerboard']):.6g} "
            f"h_norm={float(norm['horizontal']):.6g} "
            f"v_norm={float(norm['vertical']):.6g}",
            flush=True,
        )
    visible = result["visible_residual_basis"]
    if isinstance(visible, Mapping):
        for key in ("tiny_minus_reae", "tiny_minus_gt"):
            current = visible[key]
            if not isinstance(current, Mapping):
                continue
            print(
                f"  {key} p2(mean_abs): "
                f"checker={float(current['checkerboard']['mean_abs']):.6g} "
                f"horizontal={float(current['horizontal']['mean_abs']):.6g} "
                f"vertical={float(current['vertical']['mean_abs']):.6g}",
                flush=True,
            )


def _print_pair_comparison(value: Mapping[str, object]) -> None:
    print(
        f"[comparison] {value['base']} -> {value['other']} "
        "(checker_norm ratio; <1 means less current-resolution p2 locking)",
        flush=True,
    )
    layers = value["layers"]
    if not isinstance(layers, Mapping):
        return
    focus = (
        "decoder_s1_blocks_end",
        "transition12_conv",
        "decoder_s2_blocks_end",
        "transition23_nearest",
        "transition23_tgrow",
        "transition23_conv",
        "decoder_s3_blocks_end",
        "resize_x2_pre_head",
        "output_head_rgb",
    )
    for name in focus:
        payload = layers.get(name)
        if not isinstance(payload, Mapping):
            continue
        checker = payload["checkerboard"]
        if not isinstance(checker, Mapping):
            continue
        print(
            f"  {name:28s} "
            f"base={float(checker['base']):.6g} "
            f"other={float(checker['other']):.6g} "
            f"ratio={float(checker['ratio_other_over_base']):.4f}",
            flush=True,
        )


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.clip_length <= 0 or args.crop_size <= 0 or args.scale <= 0:
        raise ValueError("batch-size, clip-length, crop-size and scale must be positive")
    if args.clip_length % 4 != 1:
        raise ValueError("SwiftVR diagnostic clips must satisfy T=4k+1")
    names = [name for name, _ in args.decoder]
    if len(names) != len(set(names)):
        raise ValueError(f"ResizeConv decoder names must be unique, got {names}")

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

    results: dict[str, dict[str, object]] = {}
    for name, root in args.decoder:
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
        results[name] = result
        _print_summary(result)

    comparisons: list[dict[str, object]] = []
    if len(results) >= 2:
        ordered = list(results.values())
        anchor = ordered[0]
        for other in ordered[1:]:
            current = _comparison(anchor, other)
            comparisons.append(current)
            _print_pair_comparison(current)

    report: dict[str, object] = {
        "diagnostic": "tiny_decoder_resize_conv_layer_phase_trace_v1",
        "base_checkpoint": str(base),
        "val_cache": str(cache.root),
        "sample_positions": list(positions),
        "dtype": args.dtype,
        "decoders": results,
        "comparisons": comparisons,
        "interpretation_notes": {
            "normalized_rms": "Per-layer current-resolution 2x2 basis RMS divided by centered feature RMS; compare the same layer across checkpoints, not arbitrary layers.",
            "decoder_s2_mapping": "A current-resolution p2 mode at Decoder-S2 (96x96 for the standard diagnostic crop) can map through the remaining two spatial x2 operations to an output-space p8 pattern.",
            "nearest": "Nearest x2 duplicates each source value into a 2x2 block, so current-resolution p2 basis can collapse at the immediate nearest output while the pattern moves to a coarser period.",
            "comparison": "The first decoder passed on the command line is the anchor; pairwise ratios below 1 mean less same-layer current-resolution p2 locking after recovery.",
        },
    }
    path = output / "resize_conv_layer_phase_summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[saved] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
