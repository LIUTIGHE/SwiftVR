#!/usr/bin/env python3
"""Unified SwiftVR compression-progress MAC/FLOP profiler.

The tool deliberately composes already-validated accounting paths instead of
introducing a new FLOP convention:

* Encoder and prompt-free/no-time Transformer MACs are counted from a real
  steady-state streaming MIDDLE chunk with ``RuntimeMacCounter``.
* The original ReAE decoder is counted by the same runtime path.
* Structurally slim ReAE decoders use the validated analytical estimator from
  ``profile_reae_slim_macs.py`` because B1 decoder checkpoints are decoder-only
  and are not yet the object executed by ``StreamingTAE``.

Primary reporting convention is GMAC per emitted RGB output frame.  GFLOPs use
1 MAC = 2 FLOPs, matching the Stage-A/B profiling convention.

Typical B2-A use::

    python tools/profile_compression_progress.py \
      --input assets/example.mp4 \
      --reae-checkpoint checkpoints_prompt_free_no_time \
      --transformer-checkpoint outputs/b2a/wan13_init \
      --decoder slim100

A trained B2-A snapshot can be passed in place of ``wan13_init``; changing
weights without changing architecture will (correctly) leave MACs unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.models import ReAE
from swiftvr.models.reae_slim_decoder import (
    AGGRESSIVE_CHANNELS,
    SLIM100_CHANNELS,
    TEACHER_CHANNELS,
)
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from tools.profile_b2_dit_mac_anatomy import architecture_summary
from tools.profile_reae_slim_macs import estimate_reae_decoder_macs
from tools.profile_stage_a_streaming_macs import (
    DTYPES,
    PromptFreeMiddleSession,
    _prepare_video_geometry,
    _run_until_counted_middle,
)
from tools.runtime_macs import RuntimeMacCounter


# Canonical values already established by the validated 1080p/24-frame profiling
# work in this branch.  They are reference rows only; the current row is measured
# from the checkpoints supplied to this tool.
CANONICAL_BASELINE_1080P = {
    "label": "Original SwiftVR",
    "encoder_gmac_per_frame": 42.278,
    "transformer_gmac_per_frame": 2519.386,
    "decoder_gmac_per_frame": 343.108,
}
CANONICAL_STAGE_A_1080P = {
    "label": "Stage A: prompt-free + no-time",
    "encoder_gmac_per_frame": 42.278,
    "transformer_gmac_per_frame": 2182.43137536,
    "decoder_gmac_per_frame": 343.108,
}


def parse_resolution(value: str) -> tuple[int, int]:
    width, sep, height = value.lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    try:
        width_i, height_i = int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must contain integers") from exc
    if width_i <= 0 or height_i <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width_i, height_i


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Video long enough to reach a streaming MIDDLE chunk.")
    p.add_argument("--reae-checkpoint", type=Path, required=True,
                   help="Checkpoint root providing the current ReAE encoder/base decoder.")
    p.add_argument("--transformer-checkpoint", type=Path, required=True,
                   help="Checkpoint root containing the current prompt-free/no-time transformer/.")
    p.add_argument(
        "--decoder",
        choices=("full", "slim100", "aggressive", "checkpoint"),
        default="full",
        help=(
            "Decoder assembled into the current compute total. 'full' uses the runtime "
            "ReAE decoder count; slim presets/checkpoint use the validated analytical B1 count."
        ),
    )
    p.add_argument("--decoder-checkpoint", type=Path, default=None,
                   help="SlimReAEDecoder directory with config.json; requires --decoder checkpoint.")
    p.add_argument("--resolution", type=parse_resolution, default=(1920, 1080))
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--clip-len", type=int, default=24)
    p.add_argument("--dit-overlap", type=int, default=0)
    p.add_argument("--warmup-middle", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=tuple(DTYPES), default="float16")
    p.add_argument("--attention-backend", choices=("sdpa",), default="sdpa")
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--output-json", type=Path,
                   default=Path("outputs/compression_progress_macs.json"))
    return p


def _totalize(row: Mapping[str, object]) -> dict[str, object]:
    result = dict(row)
    total = (
        float(result["encoder_gmac_per_frame"])
        + float(result["transformer_gmac_per_frame"])
        + float(result["decoder_gmac_per_frame"])
    )
    result["total_gmac_per_frame"] = total
    result["gflops_per_frame_1mac_2flops"] = 2.0 * total
    return result


def _canonical_rows() -> list[dict[str, object]]:
    original = _totalize(CANONICAL_BASELINE_1080P)
    stage_a = _totalize(CANONICAL_STAGE_A_1080P)
    slim = estimate_reae_decoder_macs(SLIM100_CHANNELS)
    stage_a_slim = _totalize(
        {
            "label": "Stage A + B1 Slim100",
            "encoder_gmac_per_frame": CANONICAL_STAGE_A_1080P["encoder_gmac_per_frame"],
            "transformer_gmac_per_frame": CANONICAL_STAGE_A_1080P["transformer_gmac_per_frame"],
            "decoder_gmac_per_frame": float(slim["total_gmac"]),
        }
    )
    baseline = float(original["total_gmac_per_frame"])
    for row in (original, stage_a, stage_a_slim):
        current = float(row["total_gmac_per_frame"])
        row["vs_original_ratio"] = current / baseline
        row["vs_original_reduction_percent"] = 100.0 * (1.0 - current / baseline)
        row["speedup_from_compute_ratio"] = baseline / current
    return [original, stage_a, stage_a_slim]


def _read_slim_decoder_channels(path: Path) -> tuple[int, int, int, int]:
    config_path = path.expanduser().resolve() / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing slim decoder config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    channels = config.get("channels")
    if not isinstance(channels, list) or len(channels) != 4:
        raise ValueError(f"Slim decoder config has invalid channels: {channels!r}")
    values = tuple(int(value) for value in channels)
    if any(value <= 0 for value in values):
        raise ValueError(f"Slim decoder channels must be positive: {values}")
    return values  # type: ignore[return-value]


def _decoder_profile(
    args: argparse.Namespace,
    *,
    runtime_full_decoder_gmac: float,
    compute_height: int,
    compute_width: int,
) -> dict[str, object]:
    if args.decoder == "full":
        return {
            "mode": "runtime_full_reae",
            "channels": list(TEACHER_CHANNELS),
            "gmac_per_frame": float(runtime_full_decoder_gmac),
        }
    if args.decoder == "slim100":
        channels = tuple(SLIM100_CHANNELS)
        source = "preset:slim100"
    elif args.decoder == "aggressive":
        channels = tuple(AGGRESSIVE_CHANNELS)
        source = "preset:aggressive"
    else:
        if args.decoder_checkpoint is None:
            raise ValueError("--decoder checkpoint requires --decoder-checkpoint")
        channels = _read_slim_decoder_channels(args.decoder_checkpoint)
        source = str(args.decoder_checkpoint.expanduser().resolve())

    estimate = estimate_reae_decoder_macs(
        channels,
        output_height=compute_height,
        output_width=compute_width,
    )
    return {
        "mode": "analytical_structured_slim_reae",
        "source": source,
        "channels": list(channels),
        "gmac_per_frame": float(estimate["total_gmac"]),
        "components_gmac": estimate["components_gmac"],
    }


def _profile_current(args: argparse.Namespace) -> dict[str, object]:
    if args.clip_len <= 0 or args.clip_len % 4 != 0:
        raise ValueError("--clip-len must be a positive multiple of 4")
    if args.dit_overlap != 0:
        raise ValueError("Canonical compression comparison currently requires --dit-overlap 0")
    if args.warmup_middle < 0:
        raise ValueError("--warmup-middle must be non-negative")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.input = args.input.expanduser().resolve()
    reae_root = args.reae_checkpoint.expanduser().resolve()
    transformer_root = args.transformer_checkpoint.expanduser().resolve()
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    geometry = _prepare_video_geometry(args)

    reae = ReAE(str(reae_root / args.reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(transformer_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    architecture = architecture_summary(transformer)
    if not architecture["time_condition_folded"] or architecture["has_condition_embedder"]:
        raise RuntimeError(
            "Current transformer must be the folded prompt-free/no-time architecture"
        )

    reae.to(device=device, dtype=dtype).eval()
    transformer.to(device=device, dtype=dtype).eval()
    transformer.prepare_for_inference(
        attention_backend=args.attention_backend,
        use_torch_compile=False,
    )
    session = PromptFreeMiddleSession(
        reae,
        transformer,
        device=device,
        dtype=dtype,
        out_w=int(geometry["out_w"]),
        out_h=int(geometry["out_h"]),
        pad_w=int(geometry["pad_w"]),
        pad_h=int(geometry["pad_h"]),
        overlap=args.dit_overlap,
    )

    counter = RuntimeMacCounter()
    counter.add_module("encoder", reae.encoder)
    counter.add_module("transformer", transformer)
    counter.add_module("decoder", reae.decoder)
    try:
        runtime = _run_until_counted_middle(
            args,
            geometry,
            session.step,
            counter,
            label="current_prompt_free_no_time",
            expect_cross_attention=False,
        )
    finally:
        counter.close()

    roots = runtime["macs"]["by_root_gmacs_per_output_frame"]
    decoder = _decoder_profile(
        args,
        runtime_full_decoder_gmac=float(roots["decoder"]),
        compute_height=int(geometry["compute_h"]),
        compute_width=int(geometry["compute_w"]),
    )
    row = _totalize(
        {
            "label": "Current assembled model",
            "encoder_gmac_per_frame": float(roots["encoder"]),
            "transformer_gmac_per_frame": float(roots["transformer"]),
            "decoder_gmac_per_frame": float(decoder["gmac_per_frame"]),
        }
    )
    return {
        "geometry": geometry,
        "transformer_architecture": architecture,
        "decoder": decoder,
        "runtime_full_reae_roots_gmac_per_frame": dict(roots),
        "runtime_profile": runtime,
        "row": row,
    }


def _print_row(row: Mapping[str, object], baseline_gmac: float) -> None:
    total = float(row["total_gmac_per_frame"])
    gflops = float(row["gflops_per_frame_1mac_2flops"])
    print(f"{str(row['label']):30s} "
          f"Enc {float(row['encoder_gmac_per_frame']):8.3f} | "
          f"DiT {float(row['transformer_gmac_per_frame']):9.3f} | "
          f"Dec {float(row['decoder_gmac_per_frame']):8.3f} | "
          f"Total {total:9.3f} GMAC | {gflops:9.3f} GFLOPs | "
          f"{baseline_gmac / total:5.2f}x")


def main() -> int:
    args = build_parser().parse_args()
    current = _profile_current(args)
    geometry = current["geometry"]
    canonical = (
        tuple(args.resolution) == (1920, 1080)
        and int(args.clip_len) == 24
        and int(args.dit_overlap) == 0
        and int(geometry["compute_w"]) == 1920
        and int(geometry["compute_h"]) == 1088
    )

    references = _canonical_rows() if canonical else []
    if references:
        baseline_gmac = float(references[0]["total_gmac_per_frame"])
    else:
        # The canonical original reference is not resolution-portable.  For a
        # noncanonical run, compare only against the current model itself.
        baseline_gmac = float(current["row"]["total_gmac_per_frame"])

    row = dict(current["row"])
    row["vs_original_ratio"] = float(row["total_gmac_per_frame"]) / baseline_gmac
    row["vs_original_reduction_percent"] = 100.0 * (
        1.0 - float(row["total_gmac_per_frame"]) / baseline_gmac
    )
    row["speedup_from_compute_ratio"] = baseline_gmac / float(row["total_gmac_per_frame"])
    current["row"] = row

    report = {
        "format_version": 1,
        "kind": "swiftvr_compression_progress_macs",
        "mac_convention": "1 MAC = multiply-accumulate; GFLOPs view uses 2 FLOPs/MAC",
        "input": str(args.input),
        "reae_checkpoint": str(args.reae_checkpoint.expanduser().resolve()),
        "transformer_checkpoint": str(args.transformer_checkpoint.expanduser().resolve()),
        "target_resolution": list(args.resolution),
        "internal_compute_resolution": [int(geometry["compute_w"]), int(geometry["compute_h"])],
        "clip_len": int(args.clip_len),
        "dit_overlap": int(args.dit_overlap),
        "canonical_1080p_reference_available": canonical,
        "reference_rows": references,
        "current": current,
        "reporting_note": (
            "Encoder/Transformer/full-decoder values come from the real streaming MIDDLE "
            "runtime counter. Structured slim decoder values use the validated B1 analytical "
            "estimator at the same internal output geometry. Compute ratio is not latency."
        ),
    }

    print("\n================ SwiftVR compression progress ================")
    print(f"Target / compute resolution : {args.resolution[0]}x{args.resolution[1]} / "
          f"{geometry['compute_w']}x{geometry['compute_h']}")
    print(f"Transformer checkpoint      : {args.transformer_checkpoint}")
    print(f"Decoder                     : {args.decoder} {current['decoder']['channels']}")
    arch = current["transformer_architecture"]
    print(
        "Transformer architecture : "
        f"L={arch['num_layers']} D={arch['inner_dim']} "
        f"heads={arch['num_attention_heads']}x{arch['attention_head_dim']} "
        f"FFN={arch['ffn_dim']} adapter={arch['adapter_dim']}"
    )
    print("---------------------------------------------------------------")
    if references:
        for ref in references:
            _print_row(ref, baseline_gmac)
    else:
        print("Canonical Original/Stage-A rows omitted for noncanonical geometry.")
    _print_row(row, baseline_gmac)
    print("---------------------------------------------------------------")
    if references:
        print(
            f"Current vs Original       : {row['speedup_from_compute_ratio']:.3f}x less compute, "
            f"{row['vs_original_reduction_percent']:.2f}% MAC reduction"
        )
    print("===============================================================")

    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
