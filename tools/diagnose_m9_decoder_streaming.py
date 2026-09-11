#!/usr/bin/env python3
"""M9-D0: compare training-style whole-clip and deployment streaming decodes.

The canonical encoder + M8-A MoE DiT run once. Their z_SR chunks are cached and
then reused unchanged by the original ReAE decoder and M8 Decoder76.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from swiftvr import SwiftVRPromptFreeNoTimePipeline
from swiftvr.io import (
    append_chunk_to_png_dir,
    crop_spatial_padding_ntchw,
    get_video_info,
    iter_video_clips_fixed_scheme,
    preprocess_clip_uint8,
)
from swiftvr.models import ReAE
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder
from swiftvr.models.transformer_prompt_free_no_time_moe import (
    WanTransformer3DModelPromptFreeNoTimeMoE,
)
from swiftvr.streaming import StreamingTAE
from swiftvr.streaming.chunk import ChunkType
from swiftvr.training.forward import decode_reae_clip


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _parse_resolution(value: str) -> tuple[int, int]:
    w, sep, h = value.lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    return int(w), int(h)


def _normalize_total_frames(raw_total: int, limit: int) -> int:
    total = min(int(raw_total), int(limit))
    total = 4 * ((total - 1) // 4) + 1
    if total < 5:
        raise ValueError("D0 requires at least 5 usable frames")
    return total


def _spec_dict(spec) -> dict[str, object]:
    return {
        "clip_idx": int(spec.clip_idx),
        "ctype": spec.ctype.value,
        "frame_start": int(spec.frame_start),
        "frame_count": int(spec.frame_count),
        "b": int(spec.b),
        "is_first_decode": bool(spec.is_first_decode),
    }


def _rows(
    ref: torch.Tensor,
    pred: torch.Tensor,
    *,
    start: int,
    clip_idx: int,
    chunk_type: str,
) -> list[dict[str, object]]:
    if ref.shape != pred.shape:
        raise ValueError(f"metric shape mismatch: {ref.shape} vs {pred.shape}")
    result = []
    for local in range(int(ref.shape[1])):
        diff = pred[0, local].float() - ref[0, local].float()
        mae = float(diff.abs().mean())
        mse = float(diff.square().mean())
        result.append(
            {
                "frame": start + local,
                "clip_idx": clip_idx,
                "chunk_type": chunk_type,
                "chunk_local_frame": local,
                "within_2_after_chunk_start": int(clip_idx > 0 and local < 2),
                "mae": mae,
                "rmse": math.sqrt(mse),
                "max_abs": float(diff.abs().max()),
                "psnr_db": math.inf if mse == 0.0 else -10.0 * math.log10(mse),
            }
        )
    return result


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    def mean(key: str, selected=rows):
        values = [float(row[key]) for row in selected]
        return sum(values) / len(values) if values else None

    boundary = [r for r in rows if int(r["within_2_after_chunk_start"])]
    interior = [r for r in rows if not int(r["within_2_after_chunk_start"])]
    b_mae, i_mae = mean("mae", boundary), mean("mae", interior)
    return {
        "frames": len(rows),
        "mean_mae": mean("mae"),
        "mean_rmse": mean("rmse"),
        "max_abs": max(float(r["max_abs"]) for r in rows),
        "boundary_2frame_mean_mae": b_mae,
        "interior_mean_mae": i_mae,
        "boundary_to_interior_mae_ratio": (
            b_mae / i_mae if b_mae is not None and i_mae not in (None, 0.0) else None
        ),
        "exact_frame_fraction": (
            sum(float(r["max_abs"]) == 0.0 for r in rows) / len(rows)
        ),
    }


def _save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if math.isinf(float(row["psnr_db"])):
                row["psnr_db"] = "inf"
            writer.writerow(row)


@torch.inference_mode()
def _capture_latents(
    pipe,
    input_path: Path,
    *,
    total_frames: int,
    clip_len: int,
    lq_h: int,
    lq_w: int,
    out_h: int,
    out_w: int,
    pad_h: int,
    pad_w: int,
    output_dir: Path,
):
    pipe.tae_stream.reset()
    pipe.dit_stream.reset()
    pipe.dit_stream.overlap = 0
    n_lat = clip_len // 4
    prev_lq_cpu = None
    records = []
    manifest = []
    latent_dir = output_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    for spec, raw in iter_video_clips_fixed_scheme(
        input_path, clip_len, total_frames, lq_h, lq_w
    ):
        clip = preprocess_clip_uint8(
            raw.to(pipe.device),
            out_h,
            out_w,
            pipe.upscale_mode,
            pad_h,
            pad_w,
            pipe.dtype,
        )
        z_lq = pipe.tae_stream.encode_chunk_fixed(clip, spec)
        if spec.ctype == ChunkType.LAST:
            z_sr = pipe.dit_stream.denoise_last_chunk(
                z_lq,
                spec,
                pipe.prompt_emb,
                prev_lq_cpu,
                n_lat,
                pipe.device,
                pipe.dtype,
            )
        else:
            z_bcfhw = z_lq.permute(0, 2, 1, 3, 4).contiguous()
            z_den = pipe.dit_stream.denoise(z_bcfhw, pipe.prompt_emb)
            z_sr = z_den.permute(0, 2, 1, 3, 4).contiguous()
            prev_lq_cpu = z_bcfhw[:, :, -n_lat:].detach().cpu().clone()

        z_cpu = z_sr.detach().cpu().contiguous()
        record = _spec_dict(spec)
        file_name = f"chunk_{spec.clip_idx:03d}_{spec.ctype.value}.pt"
        torch.save({"z_sr": z_cpu, "spec": record}, latent_dir / file_name)
        manifest.append(
            {**record, "file": file_name, "z_sr_shape": list(z_cpu.shape)}
        )
        records.append((spec, z_cpu))
        print(
            f"[latent] {spec.clip_idx}:{spec.ctype.value} "
            f"input={spec.frame_count} z_sr={tuple(z_cpu.shape)}",
            flush=True,
        )
        del clip, z_lq, z_sr

    (latent_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return records


@torch.inference_mode()
def _run_decoder(
    name: str,
    model,
    records,
    *,
    total_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    pad_h: int,
    pad_w: int,
    output_dir: Path,
    save_png: bool,
):
    model.to(device=device, dtype=dtype).eval()
    z_all = torch.cat([z for _, z in records], dim=1).to(device, dtype)
    whole = decode_reae_clip(
        model, z_all, output_frames=total_frames, clamp=True
    )
    whole = crop_spatial_padding_ntchw(whole, pad_h, pad_w).cpu()
    del z_all
    if device.type == "cuda":
        torch.cuda.empty_cache()

    root = output_dir / name
    if save_png:
        append_chunk_to_png_dir(whole, root / "whole_png", start_idx=0)

    stream = StreamingTAE(model)
    stream.reset()
    rows = []
    cursor = 0
    for spec, z_cpu in records:
        pred = stream.decode_chunk_fixed(z_cpu.to(device, dtype), spec)
        if pred is None:
            raise RuntimeError(f"{name} buffered decoder chunk {spec.clip_idx}")
        pred = crop_spatial_padding_ntchw(pred, pad_h, pad_w).cpu()
        count = int(pred.shape[1])
        ref = whole[:, cursor : cursor + count]
        rows += _rows(
            ref,
            pred,
            start=cursor,
            clip_idx=spec.clip_idx,
            chunk_type=spec.ctype.value,
        )
        if save_png:
            append_chunk_to_png_dir(pred, root / "streaming_png", start_idx=cursor)
        print(
            f"[{name}] {spec.clip_idx}:{spec.ctype.value} "
            f"frames={cursor}..{cursor + count - 1}",
            flush=True,
        )
        cursor += count

    if cursor != total_frames:
        raise RuntimeError(f"{name} emitted {cursor} frames, expected {total_frames}")

    result = _summary(rows)
    _save_rows(root / "whole_vs_streaming.csv", rows)
    (root / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--transformer-checkpoint", type=Path, required=True)
    p.add_argument("--decoder76-checkpoint", type=Path, default=None)
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--resolution", type=_parse_resolution, default=(960, 540))
    p.add_argument("--upscale", type=int, default=3)
    p.add_argument("--clip-len", type=int, default=24)
    p.add_argument("--max-frames", type=int, default=81)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument(
        "--attention-backend",
        default="sdpa",
        choices=("sdpa", "flash_attn_2", "flash_attn_3", "sageattention", "xformers"),
    )
    p.add_argument("--save-png", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m9_d0_decoder_streaming"),
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_len <= 0 or args.clip_len % 4:
        raise ValueError("--clip-len must be a positive multiple of 4")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = DTYPES[args.dtype]

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input.expanduser().resolve()
    base_root = args.base_checkpoint.expanduser().resolve()
    transformer_root = args.transformer_checkpoint.expanduser().resolve()
    reae_path = base_root / args.reae_filename
    raw_total, lq_h, lq_w, _ = get_video_info(input_path)
    total_frames = _normalize_total_frames(raw_total, args.max_frames)

    reae = ReAE(str(reae_path))
    transformer = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
        str(transformer_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    pipe = SwiftVRPromptFreeNoTimePipeline(reae, transformer)
    pipe.to(
        device,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        torch_compile=False,
    )
    reae.decoder.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    req_w, req_h = args.resolution
    out_h, out_w, pad_h, pad_w = pipe._target_size(
        lq_h, lq_w, (req_w, req_h), args.upscale
    )
    print(
        f"D0 frames={total_frames}, LQ={lq_w}x{lq_h}, "
        f"output={out_w}x{out_h}, padded={out_w + pad_w}x{out_h + pad_h}",
        flush=True,
    )
    records = _capture_latents(
        pipe,
        input_path,
        total_frames=total_frames,
        clip_len=args.clip_len,
        lq_h=lq_h,
        lq_w=lq_w,
        out_h=out_h,
        out_w=out_w,
        pad_h=pad_h,
        pad_w=pad_w,
        output_dir=output_dir,
    )

    transformer.to("cpu")
    reae.encoder.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summaries = {
        "original": _run_decoder(
            "original",
            reae,
            records,
            total_frames=total_frames,
            device=device,
            dtype=dtype,
            pad_h=pad_h,
            pad_w=pad_w,
            output_dir=output_dir,
            save_png=args.save_png,
        )
    }
    decoder_path = None
    if args.decoder76_checkpoint is not None:
        decoder_path = args.decoder76_checkpoint.expanduser().resolve()
        slim = SlimReAEDecoder.from_pretrained(decoder_path, device="cpu")
        if tuple(slim.channels) != (128, 96, 64, 64):
            raise ValueError(f"expected Decoder76 [128,96,64,64], got {slim.channels}")
        if slim.patch_size != reae.patch_size or slim.frames_to_trim != reae.frames_to_trim:
            raise ValueError("Decoder76 ReAE temporal/spatial contract mismatch")
        summaries["decoder76"] = _run_decoder(
            "decoder76",
            slim,
            records,
            total_frames=total_frames,
            device=device,
            dtype=dtype,
            pad_h=pad_h,
            pad_w=pad_w,
            output_dir=output_dir,
            save_png=args.save_png,
        )

    report = {
        "kind": "m9_d0_decoder_whole_vs_fixed_streaming",
        "input": str(input_path),
        "base_reae": str(reae_path),
        "transformer_checkpoint": str(transformer_root),
        "decoder76_checkpoint": None if decoder_path is None else str(decoder_path),
        "dtype": args.dtype,
        "total_frames": total_frames,
        "clip_len": args.clip_len,
        "requested_output_resolution": [out_w, out_h],
        "padded_output_resolution": [out_w + pad_w, out_h + pad_h],
        "summaries": summaries,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\n========== M9-D0 ==========")
    for name, values in summaries.items():
        print(
            f"{name:10s} MAE={values['mean_mae']:.6g} "
            f"RMSE={values['mean_rmse']:.6g} max={values['max_abs']:.6g} "
            f"boundary/interior={values['boundary_to_interior_mae_ratio']}"
        )
    print(f"Saved: {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
