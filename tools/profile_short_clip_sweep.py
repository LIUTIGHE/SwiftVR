#!/usr/bin/env python3
"""Profile how SwiftVR DiT compute changes with RGB clip length.

This profiler answers a different question from ``profile_stage_a_streaming_macs.py``.
The older tool measures one steady-state MIDDLE streaming chunk.  This tool sweeps
whole ``4k+1`` clip lengths (default: 9/17/33/49/81 RGB frames) and reports the
DiT cost normalized by *RGB output frames*.

Why the DiT is profiled directly on synthetic latents
------------------------------------------------------
ReAE deterministically maps a ``4k+1`` RGB clip to ``(T+3)/4`` latent frames.
The values of the latent tensor do not affect MAC counts, while materializing an
81-frame 1920x1088 RGB clip and running the whole ReAE stack at once needlessly
inflates activation memory.  We therefore execute the real Stage-A prompt-free /
no-time DiT on the exact latent shapes and count its real Linear/Conv/SDPA MACs.

For an end-to-end compute estimate the report also adds reference per-padded-frame
ReAE encoder/decoder GMACs.  The default decoder value is the original ReAE
(343.10750208 GMAC/padded RGB frame).  Use ``--decoder-gmac-per-padded-frame
98.2228992`` for Slim-100 or ``86.79211008`` for the aggressive slim decoder.
These encoder/decoder values are analytical/reference additions; DiT MACs are
runtime-counted exactly for each swept shape.

Latency and peak-VRAM fields are DiT-only.  This isolates the temporal-length
variable that can change nonlinearly through attention, rather than mixing in
video I/O and a particular ReAE streaming chunk implementation.

Canonical MAC convention follows ``tools/runtime_macs.py`` and uses SDPA so the
QK/AV attention matmuls are observable.  FLOPs are additionally reported using
1 MAC = 2 FLOPs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.models.transformer import _WindowIndexCache, _WindowRuntimeMetaCache
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from tools.runtime_macs import RuntimeMacCounter


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
DEFAULT_LENGTHS = (9, 17, 33, 49, 81)
DEFAULT_ENCODER_GMAC_PER_PADDED_FRAME = 42.278
DEFAULT_DECODER_GMAC_PER_PADDED_FRAME = 343.10750208


def parse_resolution(value: str) -> tuple[int, int]:
    width, sep, height = str(value).lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    try:
        width_i, height_i = int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must contain integers") from exc
    if width_i <= 0 or height_i <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width_i, height_i


def parse_lengths(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one clip length is required")
    invalid = [item for item in values if item <= 0 or item % 4 != 1]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"all RGB clip lengths must satisfy T=4k+1; invalid={invalid}"
        )
    return values


def aligned_size(size: int, multiple: int = 32) -> int:
    return int(size) + ((int(multiple) - int(size) % int(multiple)) % int(multiple))


def latent_frames(rgb_frames: int) -> int:
    rgb_frames = int(rgb_frames)
    if rgb_frames <= 0 or rgb_frames % 4 != 1:
        raise ValueError(f"RGB frames must satisfy T=4k+1, got {rgb_frames}")
    return (rgb_frames + 3) // 4


def latent_spatial_shape(compute_width: int, compute_height: int) -> tuple[int, int]:
    # ReAE spatial contract: pixel-unshuffle(2) followed by three stride-2 convs.
    if compute_width % 16 or compute_height % 16:
        raise ValueError(
            f"compute geometry must be divisible by 16, got {compute_width}x{compute_height}"
        )
    return compute_height // 16, compute_width // 16


def estimate_pipeline_compute(
    *,
    rgb_frames: int,
    dit_clip_gmac: float,
    encoder_gmac_per_padded_frame: float,
    decoder_gmac_per_padded_frame: float,
) -> dict[str, float | int]:
    """Add reference ReAE compute to one exact DiT clip MAC measurement."""
    frames = int(rgb_frames)
    padded = frames + 3
    encoder_clip = float(encoder_gmac_per_padded_frame) * padded
    decoder_clip = float(decoder_gmac_per_padded_frame) * padded
    total_clip = encoder_clip + float(dit_clip_gmac) + decoder_clip
    return {
        "rgb_frames": frames,
        "padded_rgb_frames": padded,
        "padding_overhead_ratio": padded / frames,
        "encoder_clip_gmac_estimate": encoder_clip,
        "decoder_clip_gmac_estimate": decoder_clip,
        "pipeline_clip_gmac_estimate": total_clip,
        "encoder_gmac_per_output_frame_estimate": encoder_clip / frames,
        "decoder_gmac_per_output_frame_estimate": decoder_clip / frames,
        "pipeline_gmac_per_output_frame_estimate": total_clip / frames,
        "pipeline_gflops_per_output_frame_if_1mac_2flops": 2.0 * total_clip / frames,
    }


def _clear_window_caches() -> None:
    _WindowIndexCache.clear()
    _WindowRuntimeMetaCache.clear()


def _validate_counter(summary: dict[str, object]) -> None:
    if int(summary.get("total_macs", 0)) <= 0:
        raise RuntimeError("DiT MAC counter produced zero MACs")
    errors = summary.get("count_errors")
    if isinstance(errors, list) and errors:
        raise RuntimeError(f"MAC counting diagnostics are non-empty: {errors}")
    calls = summary.get("calls_by_type")
    if not isinstance(calls, dict) or int(calls.get("self_attn_qk", 0)) <= 0:
        raise RuntimeError("Shifted-window self-attention was not observed by MAC counter")
    if int(calls.get("cross_attn_qk", 0)) != 0:
        raise RuntimeError("Prompt-free/no-time DiT unexpectedly executed cross-attention")


def _dit_forward(transformer, latent: torch.Tensor) -> torch.Tensor:
    output = transformer(latent)
    sample = getattr(output, "sample", None)
    if not isinstance(sample, torch.Tensor):
        raise TypeError("Prompt-free/no-time transformer did not return .sample tensor")
    if tuple(sample.shape) != tuple(latent.shape):
        raise RuntimeError(
            f"DiT output shape mismatch: input={tuple(latent.shape)} output={tuple(sample.shape)}"
        )
    return sample


def _cuda_timed_forward(transformer, latent: torch.Tensor) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    prediction = _dit_forward(transformer, latent)
    # Materialize the one-step denoised latent too; this elementwise subtraction
    # is intentionally outside the MAC convention but belongs to runtime.
    denoised = latent - prediction
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    del prediction, denoised
    return elapsed_ms


def _cpu_timed_forward(transformer, latent: torch.Tensor) -> float:
    import time

    started = time.perf_counter()
    prediction = _dit_forward(transformer, latent)
    denoised = latent - prediction
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    del prediction, denoised
    return elapsed_ms


def _summary_ratio(value: float, reference: float) -> float:
    if reference == 0:
        return float("nan")
    return float(value) / float(reference)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--resolution", type=parse_resolution, default=(1920, 1080))
    parser.add_argument("--lengths", type=parse_lengths, default=DEFAULT_LENGTHS)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa",),
        default="sdpa",
        help="Canonical MAC counting requires SDPA so QK/AV MACs are observable.",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--encoder-gmac-per-padded-frame",
        type=float,
        default=DEFAULT_ENCODER_GMAC_PER_PADDED_FRAME,
    )
    parser.add_argument(
        "--decoder-gmac-per-padded-frame",
        type=float,
        default=DEFAULT_DECODER_GMAC_PER_PADDED_FRAME,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/short_clip_sweep.json"),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.encoder_gmac_per_padded_frame < 0 or args.decoder_gmac_per_padded_frame < 0:
        raise ValueError("encoder/decoder reference GMACs must be non-negative")


def _print_table(records: Iterable[dict[str, object]], reference_length: int) -> None:
    print("\n================ SwiftVR temporal-length sweep ================")
    print(
        " RGB  Lat | DiT GMAC/clip  DiT GMAC/f  Attn GMAC/f  Attn%  "
        "Est.Total GMAC/f  GFLOPs/f | ms/clip  ms/f    FPS   PeakGB | vsRef"
    )
    print("-" * 139)
    for record in records:
        print(
            f"{int(record['rgb_frames']):4d} {int(record['latent_frames']):4d} | "
            f"{float(record['dit_clip_gmac']):13.3f} "
            f"{float(record['dit_gmac_per_output_frame']):11.3f} "
            f"{float(record['self_attention_gmac_per_output_frame']):11.3f} "
            f"{100.0 * float(record['self_attention_fraction_of_dit']):5.1f}% "
            f"{float(record['pipeline_gmac_per_output_frame_estimate']):17.3f} "
            f"{float(record['pipeline_gflops_per_output_frame_if_1mac_2flops']):9.3f} | "
            f"{float(record['latency_ms_per_clip_median']):7.1f} "
            f"{float(record['latency_ms_per_output_frame_median']):6.2f} "
            f"{float(record['fps_from_median_latency']):6.2f} "
            f"{float(record.get('peak_allocated_gb', float('nan'))):7.2f} | "
            f"{float(record['dit_gmac_per_frame_ratio_vs_reference']):5.3f}"
        )
    print(f"Reference length for ratios: {reference_length} RGB frames")
    print("Latency/VRAM are DiT-only; end-to-end GMAC/f adds analytical ReAE reference costs.")
    print("===============================================================\n")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = DTYPES[args.dtype]
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    checkpoint = args.checkpoint.expanduser().resolve()
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(checkpoint),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    transformer.to(device=device, dtype=dtype).eval()
    transformer.prepare_for_inference(
        attention_backend=args.attention_backend,
        use_torch_compile=False,
    )

    out_w, out_h = args.resolution
    compute_w = aligned_size(out_w, 32)
    compute_h = aligned_size(out_h, 32)
    latent_h, latent_w = latent_spatial_shape(compute_w, compute_h)
    latent_channels = int(getattr(transformer.config, "in_channels", 48))
    lengths = tuple(int(value) for value in args.lengths)
    reference_length = max(lengths)

    records: list[dict[str, object]] = []
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    for rgb_frames in lengths:
        _clear_window_caches()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        lat_frames = latent_frames(rgb_frames)
        latent = torch.randn(
            1,
            latent_channels,
            lat_frames,
            latent_h,
            latent_w,
            device=device,
            dtype=dtype,
        )

        # Warm-up builds window metadata and lets backend kernels settle before
        # latency/VRAM measurement.  Outputs are immediately released.
        with torch.inference_mode():
            for _ in range(args.warmup):
                prediction = _dit_forward(transformer, latent)
                denoised = latent - prediction
                del prediction, denoised
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Exact runtime MAC count for the real prepared DiT graph.
        counter = RuntimeMacCounter()
        counter.add_module("transformer", transformer)
        try:
            with torch.inference_mode(), counter.count(reset=True):
                prediction = _dit_forward(transformer, latent)
                denoised = latent - prediction
            del prediction, denoised
            macs = counter.summary(emitted_frames=rgb_frames)
            _validate_counter(macs)
        finally:
            counter.close()

        by_type = macs.get("by_type_gmacs", {})
        if not isinstance(by_type, dict):
            raise RuntimeError("MAC summary lacks by_type_gmacs")
        attn_clip_gmac = float(by_type.get("self_attn_qk", 0.0)) + float(
            by_type.get("self_attn_av", 0.0)
        )
        dit_clip_gmac = float(macs["total_gmacs"])
        dit_gmac_per_frame = dit_clip_gmac / rgb_frames
        attn_gmac_per_frame = attn_clip_gmac / rgb_frames
        attention_fraction = attn_clip_gmac / dit_clip_gmac if dit_clip_gmac > 0 else 0.0

        pipeline = estimate_pipeline_compute(
            rgb_frames=rgb_frames,
            dit_clip_gmac=dit_clip_gmac,
            encoder_gmac_per_padded_frame=args.encoder_gmac_per_padded_frame,
            decoder_gmac_per_padded_frame=args.decoder_gmac_per_padded_frame,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
            baseline_alloc = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_alloc = baseline_reserved = 0

        timings_ms: list[float] = []
        with torch.inference_mode():
            for _ in range(args.repeats):
                if device.type == "cuda":
                    timings_ms.append(_cuda_timed_forward(transformer, latent))
                else:
                    timings_ms.append(_cpu_timed_forward(transformer, latent))

        median_ms = float(statistics.median(timings_ms))
        mean_ms = float(statistics.mean(timings_ms))
        if device.type == "cuda":
            peak_alloc = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_alloc = peak_reserved = 0

        record: dict[str, object] = {
            "rgb_frames": rgb_frames,
            "latent_frames": lat_frames,
            "latent_shape_bcfhw": [1, latent_channels, lat_frames, latent_h, latent_w],
            "dit_clip_gmac": dit_clip_gmac,
            "dit_gmac_per_output_frame": dit_gmac_per_frame,
            "dit_gflops_per_output_frame_if_1mac_2flops": 2.0 * dit_gmac_per_frame,
            "self_attention_clip_gmac": attn_clip_gmac,
            "self_attention_gmac_per_output_frame": attn_gmac_per_frame,
            "self_attention_fraction_of_dit": attention_fraction,
            "operator_gmacs": by_type,
            "latency_repeats_ms": timings_ms,
            "latency_ms_per_clip_median": median_ms,
            "latency_ms_per_clip_mean": mean_ms,
            "latency_ms_per_output_frame_median": median_ms / rgb_frames,
            "fps_from_median_latency": 1000.0 * rgb_frames / max(median_ms, 1e-12),
            "baseline_allocated_gb": baseline_alloc / (1024**3),
            "baseline_reserved_gb": baseline_reserved / (1024**3),
            "peak_allocated_gb": peak_alloc / (1024**3),
            "peak_reserved_gb": peak_reserved / (1024**3),
            "incremental_peak_allocated_gb": max(0, peak_alloc - baseline_alloc) / (1024**3),
            **pipeline,
            "mac_count_diagnostics": macs.get("count_errors", []),
        }
        records.append(record)
        print(
            f"[T={rgb_frames:2d} -> F={lat_frames:2d}] "
            f"DiT={dit_gmac_per_frame:.3f} GMAC/f, "
            f"attn={attn_gmac_per_frame:.3f} GMAC/f ({100*attention_fraction:.1f}%), "
            f"median={median_ms/rgb_frames:.2f} ms/f",
            flush=True,
        )

        del latent
        _clear_window_caches()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reference = next(item for item in records if int(item["rgb_frames"]) == reference_length)
    ref_dit_frame = float(reference["dit_gmac_per_output_frame"])
    ref_attn_frame = float(reference["self_attention_gmac_per_output_frame"])
    ref_pipeline_frame = float(reference["pipeline_gmac_per_output_frame_estimate"])
    ref_latency_frame = float(reference["latency_ms_per_output_frame_median"])
    ref_peak = float(reference.get("peak_allocated_gb", 0.0))

    for record in records:
        record["reference_rgb_frames"] = reference_length
        record["dit_gmac_per_frame_ratio_vs_reference"] = _summary_ratio(
            float(record["dit_gmac_per_output_frame"]), ref_dit_frame
        )
        record["attention_gmac_per_frame_ratio_vs_reference"] = _summary_ratio(
            float(record["self_attention_gmac_per_output_frame"]), ref_attn_frame
        )
        record["pipeline_gmac_per_frame_ratio_vs_reference"] = _summary_ratio(
            float(record["pipeline_gmac_per_output_frame_estimate"]), ref_pipeline_frame
        )
        record["latency_per_frame_ratio_vs_reference"] = _summary_ratio(
            float(record["latency_ms_per_output_frame_median"]), ref_latency_frame
        )
        record["peak_allocated_ratio_vs_reference"] = _summary_ratio(
            float(record.get("peak_allocated_gb", 0.0)), ref_peak
        )

    report = {
        "status": "PASS",
        "measurement": "one_shot_whole_clip_temporal_shape_sweep",
        "checkpoint": str(checkpoint),
        "transformer_subfolder": args.transformer_subfolder,
        "requested_resolution_wh": [out_w, out_h],
        "compute_resolution_wh": [compute_w, compute_h],
        "latent_spatial_hw": [latent_h, latent_w],
        "lengths": list(lengths),
        "reference_length": reference_length,
        "dtype": args.dtype,
        "device": str(device),
        "attention_backend": args.attention_backend,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "encoder_gmac_per_padded_frame_reference": args.encoder_gmac_per_padded_frame,
        "decoder_gmac_per_padded_frame_reference": args.decoder_gmac_per_padded_frame,
        "latency_vram_scope": "DiT only",
        "pipeline_compute_scope": "runtime-counted DiT + analytical/reference ReAE encoder/decoder",
        "mac_convention": (
            "RuntimeMacCounter Linear/Conv/attention matmul MACs; FLOPs use 1 MAC=2 FLOPs"
        ),
        "records": records,
    }

    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _print_table(records, reference_length)
    print(f"Saved sweep report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
