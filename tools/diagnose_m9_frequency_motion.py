#!/usr/bin/env python3
"""M9-D1: diagnose Decoder76 error across spatial frequency and motion.

This tool reuses the z_SR cache written by ``diagnose_m9_decoder_streaming.py``.
It does not run the encoder or Transformer again.  The cached latents are decoded
through the original ReAE decoder and Decoder76 using the same canonical streaming
path, then the compact-decoder error is split into low/high spatial frequencies.

For frame t, with O_t = OriginalDecoder(z_t), C_t = Decoder76(z_t):

    e_L(t) = LP(C_t) - LP(O_t)
    e_H(t) = HP(C_t) - HP(O_t)

Temporal decoder error is measured as |e_*(t)-e_*(t-1)|.  To avoid confusing
"more scene motion" with "worse temporal modelling", the tool also reports a
relative error divided by the corresponding temporal energy of the original
decoder.  Motion is a cheap diagnostic proxy: mean absolute grayscale difference
between adjacent LQ frames after downsampling.

The main outputs are:
  * per_frame.csv
  * report.json with motion quartiles and Pearson correlations

This is a diagnostic, not a quality metric or training objective.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from swiftvr.io import (
    crop_spatial_padding_ntchw,
    get_video_info,
    iter_video_clips_fixed_scheme,
)
from swiftvr.models import ReAE
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder
from swiftvr.streaming import StreamingTAE
from swiftvr.streaming.chunk import ChunkSpec, ChunkType


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _gaussian_kernel(kernel_size: int, sigma: float, *, device, dtype) -> torch.Tensor:
    kernel_size = int(kernel_size)
    sigma = float(sigma)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("gaussian kernel size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("gaussian sigma must be positive")
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    k2 = torch.outer(k, k)
    return k2.to(dtype=dtype)


def _lowpass(video: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Gaussian low-pass for [B,T,C,H,W], computed channel-wise."""
    if video.ndim != 5:
        raise ValueError(f"expected [B,T,C,H,W], got {tuple(video.shape)}")
    b, t, c, h, w = video.shape
    pad = int(kernel.shape[-1] // 2)
    flat = video.reshape(b * t, c, h, w)
    weight = kernel.view(1, 1, *kernel.shape).expand(c, 1, -1, -1)
    flat = F.pad(flat, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(flat, weight, groups=c)
    return out.reshape(b, t, c, h, w)


def _pearson(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    x = torch.tensor(list(xs), dtype=torch.float64)
    y = torch.tensor(list(ys), dtype=torch.float64)
    if x.numel() < 2 or x.numel() != y.numel():
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if float(denom) == 0.0:
        return None
    return float((x * y).sum() / denom)


def _mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0.0):
        return None
    return float(a / b)


def _motion_quartiles(rows: list[dict[str, object]]) -> dict[str, object]:
    temporal = [row for row in rows if row.get("motion") is not None]
    temporal.sort(key=lambda row: float(row["motion"]))
    n = len(temporal)
    if n < 4:
        raise ValueError("D1 requires at least four temporal frame pairs")

    groups: list[list[dict[str, object]]] = []
    for q in range(4):
        start = q * n // 4
        end = (q + 1) * n // 4
        groups.append(temporal[start:end])

    keys = (
        "motion",
        "lf_spatial_mae",
        "hf_spatial_mae",
        "lf_temporal_error",
        "hf_temporal_error",
        "lf_reference_temporal_energy",
        "hf_reference_temporal_energy",
        "lf_temporal_relative",
        "hf_temporal_relative",
    )
    result: dict[str, object] = {}
    for index, group in enumerate(groups, start=1):
        result[f"Q{index}"] = {
            "frames": len(group),
            "frame_min": min(int(row["frame"]) for row in group),
            "frame_max": max(int(row["frame"]) for row in group),
            **{key: _mean(group, key) for key in keys},
        }

    q1 = result["Q1"]
    q4 = result["Q4"]
    assert isinstance(q1, dict) and isinstance(q4, dict)
    result["Q4_over_Q1"] = {
        key: _ratio(q4.get(key), q1.get(key))
        for key in (
            "lf_spatial_mae",
            "hf_spatial_mae",
            "lf_temporal_error",
            "hf_temporal_error",
            "lf_temporal_relative",
            "hf_temporal_relative",
        )
    }
    return result


def _motion_scores(
    input_path: Path,
    *,
    total_frames: int,
    clip_len: int,
    motion_width: int,
) -> list[float | None]:
    raw_total, lq_h, lq_w, _ = get_video_info(input_path)
    if raw_total < total_frames:
        raise ValueError(
            f"input has {raw_total} usable frames but D0 report expects {total_frames}"
        )
    motion_width = int(motion_width)
    if motion_width <= 0:
        raise ValueError("motion-width must be positive")
    motion_height = max(1, int(round(lq_h * motion_width / lq_w)))

    scores: list[float | None] = []
    previous = None
    seen = 0
    for _spec, raw in iter_video_clips_fixed_scheme(
        input_path,
        clip_len=clip_len,
        total_frames=total_frames,
        crop_h=lq_h,
        crop_w=lq_w,
    ):
        # [T,H,W,3] uint8 -> inexpensive downsampled grayscale in [0,1].
        x = raw.permute(0, 3, 1, 2).float().div_(255.0)
        x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = F.interpolate(
            x,
            size=(motion_height, motion_width),
            mode="bilinear",
            align_corners=False,
        )
        for index in range(int(x.shape[0])):
            current = x[index]
            if previous is None:
                scores.append(None)
            else:
                scores.append(float((current - previous).abs().mean()))
            previous = current
            seen += 1

    if seen != total_frames or len(scores) != total_frames:
        raise RuntimeError(
            f"motion proxy produced {len(scores)} frames, expected {total_frames}"
        )
    return scores


def _load_latent_records(d0_dir: Path) -> list[tuple[ChunkSpec, torch.Tensor]]:
    latent_dir = d0_dir / "latents"
    manifest_path = latent_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"invalid D0 latent manifest: {manifest_path}")

    records = []
    for item in sorted(manifest, key=lambda value: int(value["clip_idx"])):
        spec = ChunkSpec(
            ctype=ChunkType(str(item["ctype"])),
            frame_start=int(item["frame_start"]),
            frame_count=int(item["frame_count"]),
            b=int(item["b"]),
            clip_idx=int(item["clip_idx"]),
            is_first_decode=bool(item["is_first_decode"]),
        )
        payload = _load_torch(latent_dir / str(item["file"]))
        if not isinstance(payload, dict) or "z_sr" not in payload:
            raise ValueError(f"invalid latent payload: {item['file']}")
        z = payload["z_sr"]
        if not isinstance(z, torch.Tensor) or z.ndim != 5:
            raise ValueError(f"invalid z_sr tensor in {item['file']}")
        records.append((spec, z.contiguous()))
    return records


def _analyse_batch(
    original: torch.Tensor,
    compact: torch.Tensor,
    *,
    kernel: torch.Tensor,
    start_frame: int,
    motion_scores: list[float | None],
    previous: dict[str, torch.Tensor | None],
    relative_epsilon: float,
) -> list[dict[str, object]]:
    # Frequency analysis is intentionally float32, even when decoder inference is BF16.
    original = original.float()
    compact = compact.float()
    low_o = _lowpass(original, kernel)
    low_c = _lowpass(compact, kernel)
    high_o = original - low_o
    high_c = compact - low_c
    err_l = low_c - low_o
    err_h = high_c - high_o

    rows: list[dict[str, object]] = []
    for local in range(int(original.shape[1])):
        frame = start_frame + local
        cur_low_o = low_o[:, local]
        cur_high_o = high_o[:, local]
        cur_err_l = err_l[:, local]
        cur_err_h = err_h[:, local]

        row: dict[str, object] = {
            "frame": frame,
            "motion": motion_scores[frame],
            "lf_spatial_mae": float(cur_err_l.abs().mean()),
            "hf_spatial_mae": float(cur_err_h.abs().mean()),
            "lf_temporal_error": None,
            "hf_temporal_error": None,
            "lf_reference_temporal_energy": None,
            "hf_reference_temporal_energy": None,
            "lf_temporal_relative": None,
            "hf_temporal_relative": None,
        }

        prev_low_o = previous["low_o"]
        prev_high_o = previous["high_o"]
        prev_err_l = previous["err_l"]
        prev_err_h = previous["err_h"]
        if prev_low_o is not None:
            lf_temp = float((cur_err_l - prev_err_l).abs().mean())
            hf_temp = float((cur_err_h - prev_err_h).abs().mean())
            lf_ref = float((cur_low_o - prev_low_o).abs().mean())
            hf_ref = float((cur_high_o - prev_high_o).abs().mean())
            row.update(
                {
                    "lf_temporal_error": lf_temp,
                    "hf_temporal_error": hf_temp,
                    "lf_reference_temporal_energy": lf_ref,
                    "hf_reference_temporal_energy": hf_ref,
                    "lf_temporal_relative": lf_temp / max(lf_ref, relative_epsilon),
                    "hf_temporal_relative": hf_temp / max(hf_ref, relative_epsilon),
                }
            )

        # Only one frame of history is required for the pairwise temporal diagnostic.
        previous["low_o"] = cur_low_o.detach()
        previous["high_o"] = cur_high_o.detach()
        previous["err_l"] = cur_err_l.detach()
        previous["err_h"] = cur_err_h.detach()
        rows.append(row)
    return rows


@torch.inference_mode()
def run_d1(
    *,
    d0_dir: Path,
    report: dict[str, object],
    device: torch.device,
    dtype: torch.dtype,
    gaussian_kernel: int,
    gaussian_sigma: float,
    analysis_batch_frames: int,
    motion_width: int,
    relative_epsilon: float,
) -> list[dict[str, object]]:
    base_reae = Path(str(report["base_reae"])).expanduser().resolve()
    decoder76 = Path(str(report["decoder76_checkpoint"])).expanduser().resolve()
    input_path = Path(str(report["input"])).expanduser().resolve()
    total_frames = int(report["total_frames"])
    clip_len = int(report["clip_len"])
    requested = report["requested_output_resolution"]
    padded = report["padded_output_resolution"]
    if not isinstance(requested, list) or not isinstance(padded, list):
        raise ValueError("D0 report is missing output geometry")
    pad_w = int(padded[0]) - int(requested[0])
    pad_h = int(padded[1]) - int(requested[1])

    if not base_reae.is_file():
        raise FileNotFoundError(base_reae)
    if not decoder76.is_dir():
        raise FileNotFoundError(decoder76)

    records = _load_latent_records(d0_dir)
    motion_scores = _motion_scores(
        input_path,
        total_frames=total_frames,
        clip_len=clip_len,
        motion_width=motion_width,
    )

    reae = ReAE(str(base_reae)).to(device=device, dtype=dtype).eval()
    compact = SlimReAEDecoder.from_pretrained(
        decoder76,
        device=device,
        dtype=dtype,
    ).eval()
    if tuple(compact.channels) != (128, 96, 64, 64):
        raise ValueError(f"D1 expects Decoder76 [128,96,64,64], got {compact.channels}")

    original_stream = StreamingTAE(reae)
    compact_stream = StreamingTAE(compact)
    original_stream.reset()
    compact_stream.reset()

    kernel = _gaussian_kernel(
        gaussian_kernel,
        gaussian_sigma,
        device=device,
        dtype=torch.float32,
    )
    analysis_batch_frames = int(analysis_batch_frames)
    if analysis_batch_frames <= 0:
        raise ValueError("analysis-batch-frames must be positive")

    previous: dict[str, torch.Tensor | None] = {
        "low_o": None,
        "high_o": None,
        "err_l": None,
        "err_h": None,
    }
    rows: list[dict[str, object]] = []
    cursor = 0
    for spec, z_cpu in records:
        z = z_cpu.to(device=device, dtype=dtype)
        original = original_stream.decode_chunk_fixed(z, spec)
        compact_out = compact_stream.decode_chunk_fixed(z, spec)
        if original is None or compact_out is None:
            raise RuntimeError(f"decoder buffered D0 chunk {spec.clip_idx}")
        original = crop_spatial_padding_ntchw(original, pad_h, pad_w)
        compact_out = crop_spatial_padding_ntchw(compact_out, pad_h, pad_w)
        if original.shape != compact_out.shape:
            raise RuntimeError(
                f"decoder shape mismatch in chunk {spec.clip_idx}: "
                f"{tuple(original.shape)} vs {tuple(compact_out.shape)}"
            )

        count = int(original.shape[1])
        for start in range(0, count, analysis_batch_frames):
            end = min(start + analysis_batch_frames, count)
            rows.extend(
                _analyse_batch(
                    original[:, start:end],
                    compact_out[:, start:end],
                    kernel=kernel,
                    start_frame=cursor + start,
                    motion_scores=motion_scores,
                    previous=previous,
                    relative_epsilon=relative_epsilon,
                )
            )
        print(
            f"[D1] chunk {spec.clip_idx}:{spec.ctype.value} "
            f"frames={cursor}..{cursor + count - 1}",
            flush=True,
        )
        cursor += count
        del z, original, compact_out

    if cursor != total_frames or len(rows) != total_frames:
        raise RuntimeError(
            f"D1 produced {len(rows)} rows / {cursor} frames; expected {total_frames}"
        )
    return rows


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--d0-dir",
        type=Path,
        required=True,
        help="D0 output directory containing report.json and latents/.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=tuple(DTYPES), default=None)
    p.add_argument("--gaussian-kernel", type=int, default=7)
    p.add_argument("--gaussian-sigma", type=float, default=1.5)
    p.add_argument("--analysis-batch-frames", type=int, default=4)
    p.add_argument("--motion-width", type=int, default=320)
    p.add_argument("--relative-epsilon", type=float, default=1e-6)
    p.add_argument("--output-dir", type=Path, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    d0_dir = args.d0_dir.expanduser().resolve()
    report_path = d0_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("kind") != "m9_d0_decoder_whole_vs_fixed_streaming":
        raise ValueError(f"unexpected D0 report kind: {report.get('kind')!r}")
    if not report.get("decoder76_checkpoint"):
        raise ValueError("D0 report does not contain Decoder76")

    dtype_name = args.dtype or str(report.get("dtype", "bfloat16"))
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype_name!r}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.relative_epsilon <= 0:
        raise ValueError("relative-epsilon must be positive")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else d0_dir / "d1_frequency_motion"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_d1(
        d0_dir=d0_dir,
        report=report,
        device=device,
        dtype=DTYPES[dtype_name],
        gaussian_kernel=args.gaussian_kernel,
        gaussian_sigma=args.gaussian_sigma,
        analysis_batch_frames=args.analysis_batch_frames,
        motion_width=args.motion_width,
        relative_epsilon=args.relative_epsilon,
    )
    _save_csv(output_dir / "per_frame.csv", rows)

    temporal = [row for row in rows if row["motion"] is not None]
    correlations = {
        key: _pearson(
            (float(row["motion"]) for row in temporal),
            (float(row[key]) for row in temporal),
        )
        for key in (
            "lf_spatial_mae",
            "hf_spatial_mae",
            "lf_temporal_error",
            "hf_temporal_error",
            "lf_temporal_relative",
            "hf_temporal_relative",
        )
    }
    quartiles = _motion_quartiles(rows)
    report_out = {
        "kind": "m9_d1_frequency_x_motion",
        "source_d0": str(d0_dir),
        "input": report["input"],
        "base_reae": report["base_reae"],
        "decoder76_checkpoint": report["decoder76_checkpoint"],
        "dtype": dtype_name,
        "frames": len(rows),
        "temporal_pairs": len(temporal),
        "frequency_split": {
            "operator": "gaussian_lowpass / residual_highpass",
            "kernel_size": int(args.gaussian_kernel),
            "sigma": float(args.gaussian_sigma),
        },
        "motion_proxy": {
            "operator": "mean_abs_adjacent_LQ_grayscale_difference",
            "downsample_width": int(args.motion_width),
        },
        "global_means": {
            key: _mean(rows, key)
            for key in (
                "lf_spatial_mae",
                "hf_spatial_mae",
                "lf_temporal_error",
                "hf_temporal_error",
                "lf_temporal_relative",
                "hf_temporal_relative",
            )
        },
        "pearson_motion_correlation": correlations,
        "motion_quartiles": quartiles,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report_out, indent=2), encoding="utf-8"
    )

    q4q1 = quartiles["Q4_over_Q1"]
    print("\n========== M9-D1 Frequency x Motion ==========")
    print(f"frames / temporal pairs: {len(rows)} / {len(temporal)}")
    print(
        "corr motion -> temporal relative: "
        f"LF={correlations['lf_temporal_relative']} "
        f"HF={correlations['hf_temporal_relative']}"
    )
    print(
        "Q4/Q1 temporal relative: "
        f"LF={q4q1['lf_temporal_relative']} "
        f"HF={q4q1['hf_temporal_relative']}"
    )
    print(
        "Q4/Q1 spatial MAE: "
        f"LF={q4q1['lf_spatial_mae']} "
        f"HF={q4q1['hf_spatial_mae']}"
    )
    print(f"Saved: {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
