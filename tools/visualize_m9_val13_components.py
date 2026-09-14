#!/usr/bin/env python3
"""Visualize exact val13 component differences for the M9 compression lineage.

The script reconstructs the deterministic validation views recorded by an M9-A1
training run and compares:

1. GT;
2. Stage-A D3072 + original ReAE decoder (uncompressed teacher path);
3. cached M8-A D1024/L20 MoE + original ReAE decoder (Transformer-only change);
4. cached M8-A + M9-A1 factorized decoder (current ~500G path);
5. optional cached M8-A + Decoder76 E8 (old ~500G decoder baseline).

M8-A z_SR is read from the exact validation cache used by M9-A1 training.
Stage-A z_SR is recomputed on the same deterministic 13-frame / 384x384 views.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_tiny_decoder_formal_ddp as formal
from swiftvr.models import ReAE
from swiftvr.models.m9_factorized_decoder import M9A1FactorizedReAEDecoder
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training.distillation import distillation_sample_identity
from swiftvr.training.forward import (
    decode_reae_clip,
    encode_reae_clip,
    forward_prompt_free_no_time_training,
    prepare_prompt_free_no_time_transformer_for_training,
    prepare_training_batch,
)
from swiftvr.training.stage3 import VideoMetricAccumulator
from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--a1-run",
        type=Path,
        required=True,
        help="M9-A1 training run containing run_config.json.",
    )
    p.add_argument(
        "--a1-checkpoint",
        type=Path,
        required=True,
        help="M9-A1 tiny_decoder directory, typically best epoch99 checkpoint.",
    )
    p.add_argument(
        "--stagea-checkpoint",
        type=Path,
        required=True,
        help="Materialized Stage-A D3072 checkpoint root containing transformer/.",
    )
    p.add_argument(
        "--decoder76-checkpoint",
        type=Path,
        default=None,
        help="Optional Decoder76 E8 tiny_decoder directory.",
    )
    p.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Override frozen ReAE checkpoint root; default comes from run_config.json.",
    )
    p.add_argument(
        "--path-root",
        type=Path,
        default=Path("."),
        help="Dataset path root used by the original training launch if manifests use relative paths.",
    )
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--stagea-subfolder", default="transformer")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--frame-indices",
        default="0,6,12",
        help="Comma-separated frames used for static contact sheets.",
    )
    p.add_argument("--panel-size", type=int, default=256)
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--verify-paths", action="store_true")
    return p


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _required(config: Mapping[str, object], key: str):
    if key not in config:
        raise KeyError(f"run_config.json is missing {key!r}")
    return config[key]


def _build_val_dataset(
    config: Mapping[str, object],
    *,
    path_root: Path,
    verify_paths: bool,
):
    val_cache = TinyDecoderLatentCache(Path(str(_required(config, "val_cache"))))
    manifests_raw = _required(config, "val_manifests")
    if not isinstance(manifests_raw, list) or not manifests_raw:
        raise ValueError("run_config val_manifests must be a non-empty list")
    manifests = [Path(str(value)) for value in manifests_raw]
    dataset = formal._cache_subset(
        manifests,
        val_cache,
        split=str(_required(config, "val_split")),
        path_root=path_root.expanduser().resolve(),
        clip_length=int(_required(config, "clip_length")),
        crop_size=int(_required(config, "val_crop_size")),
        scale=int(_required(config, "scale")),
        views_per_record=int(_required(config, "val_views_per_record")),
        view_seed=int(_required(config, "val_view_seed")),
        hflip=float(_required(config, "val_horizontal_flip_probability")),
        vflip=float(_required(config, "val_vertical_flip_probability")),
        verify_paths=bool(verify_paths),
    )
    if len(dataset) != 13:
        raise ValueError(
            f"Expected the formal val13 subset, got {len(dataset)} views. "
            "Check --a1-run and --path-root."
        )
    return dataset, val_cache


def _loader(dataset):
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )


def _tensor_frame_image(frame: torch.Tensor, *, size: int | None = None) -> Image.Image:
    value = (
        frame.detach()
        .float()
        .clamp(0, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    image = Image.fromarray(value, mode="RGB")
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def _save_video_pngs(video: torch.Tensor, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if video.ndim != 5 or int(video.shape[0]) != 1:
        raise ValueError(f"Expected [1,T,3,H,W], got {tuple(video.shape)}")
    for index in range(int(video.shape[1])):
        _tensor_frame_image(video[0, index]).save(root / f"{index:05d}.png")


def _labeled_tile(frame: torch.Tensor, label: str, panel_size: int) -> Image.Image:
    image = _tensor_frame_image(frame, size=panel_size)
    band = 24
    canvas = Image.new("RGB", (panel_size, panel_size + band), "white")
    canvas.paste(image, (0, band))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), label, fill="black")
    return canvas


def _make_contact_sheet(
    methods: list[tuple[str, torch.Tensor]],
    frame_indices: list[int],
    output: Path,
    *,
    panel_size: int,
) -> None:
    band = 24
    width = panel_size * len(methods)
    height = (panel_size + band) * len(frame_indices)
    sheet = Image.new("RGB", (width, height), "white")
    for row, frame_index in enumerate(frame_indices):
        for col, (label, video) in enumerate(methods):
            tile = _labeled_tile(video[0, frame_index], label, panel_size)
            y = row * (panel_size + band)
            sheet.paste(tile, (col * panel_size, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _make_comparison_frames(
    methods: list[tuple[str, torch.Tensor]],
    output_dir: Path,
    *,
    panel_size: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = min(int(video.shape[1]) for _, video in methods)
    for frame_index in range(frames):
        tiles = [
            _labeled_tile(video[0, frame_index], label, panel_size)
            for label, video in methods
        ]
        canvas = Image.new(
            "RGB",
            (panel_size * len(tiles), panel_size + 24),
            "white",
        )
        for col, tile in enumerate(tiles):
            canvas.paste(tile, (col * panel_size, 0))
        canvas.save(output_dir / f"{frame_index:05d}.png")
    return frames


def _encode_mp4(frame_dir: Path, output: Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[visual] ffmpeg not found; comparison PNG sequence retained.", flush=True)
        return False
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(float(fps)),
        "-i",
        str(frame_dir / "%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "16",
        str(output),
    ]
    subprocess.run(command, check=True)
    return True


def _metric_dict(accumulator: VideoMetricAccumulator) -> dict[str, float | int]:
    return dict(accumulator.compute())


@torch.inference_mode()
def _stagea_reference_pass(
    dataset,
    *,
    base_root: Path,
    stagea_root: Path,
    reae_filename: str,
    transformer_subfolder: str,
    device: torch.device,
    dtype: torch.dtype,
    attention_backend: str,
):
    reae = ReAE(str(base_root / reae_filename)).to(device=device, dtype=dtype).eval()
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(stagea_root),
        subfolder=transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype)
    prepare_prompt_free_no_time_transformer_for_training(
        transformer,
        attention_backend=attention_backend,
    )
    transformer.eval()

    references: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    lq_inputs: list[torch.Tensor] = []
    identities: list[dict[str, object]] = []
    stagea_gt = VideoMetricAccumulator()

    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    for sample_index, batch_cpu in enumerate(_loader(dataset)):
        moved = formal._move_pixels(batch_cpu, device, dtype)
        prepared = prepare_training_batch(moved)
        lq_input = prepared["lq_input"]
        target = prepared["target"]
        if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("val13 batch is missing lq_input/target")

        with torch.autocast(
            device_type=device.type,
            dtype=dtype if autocast_enabled else torch.float32,
            enabled=autocast_enabled,
        ):
            z_lq = encode_reae_clip(reae, lq_input)
            velocity = forward_prompt_free_no_time_training(
                transformer,
                z_lq.permute(0, 2, 1, 3, 4).contiguous(),
            )
            z_stagea = (
                z_lq.permute(0, 2, 1, 3, 4).contiguous() - velocity
            ).permute(0, 2, 1, 3, 4).contiguous()
            stagea_rgb = decode_reae_clip(
                reae,
                z_stagea,
                output_frames=int(target.shape[1]),
                clamp=True,
            )

        stagea_gt.update(stagea_rgb, target, clamp=True)
        references.append(stagea_rgb.float().cpu())
        targets.append(target.float().clamp(0, 1).cpu())
        lq_inputs.append(lq_input.float().clamp(0, 1).cpu())
        identities.append(dict(distillation_sample_identity(batch_cpu, 0)))
        print(f"[Stage-A] val view {sample_index + 1:02d}/13", flush=True)

    transformer.to("cpu")
    reae.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return references, targets, lq_inputs, identities, _metric_dict(stagea_gt)


@torch.inference_mode()
def _compressed_pass(
    dataset,
    val_cache: TinyDecoderLatentCache,
    stagea_references: list[torch.Tensor],
    targets: list[torch.Tensor],
    *,
    base_root: Path,
    reae_filename: str,
    a1_checkpoint: Path,
    decoder76_checkpoint: Path | None,
    device: torch.device,
    dtype: torch.dtype,
):
    reae = ReAE(str(base_root / reae_filename)).to(device=device, dtype=dtype).eval()
    a1 = M9A1FactorizedReAEDecoder.from_pretrained(
        a1_checkpoint,
        device=device,
        dtype=dtype,
    ).eval()
    decoder76 = None
    if decoder76_checkpoint is not None:
        decoder76 = SlimReAEDecoder.from_pretrained(
            decoder76_checkpoint,
            device=device,
            dtype=dtype,
        ).eval()

    m8orig_gt = VideoMetricAccumulator()
    a1_gt = VideoMetricAccumulator()
    m8orig_stagea = VideoMetricAccumulator()
    a1_m8orig = VideoMetricAccumulator()
    a1_stagea = VideoMetricAccumulator()
    d76_gt = VideoMetricAccumulator() if decoder76 is not None else None
    d76_m8orig = VideoMetricAccumulator() if decoder76 is not None else None

    outputs: list[dict[str, torch.Tensor]] = []
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    for sample_index, batch_cpu in enumerate(_loader(dataset)):
        target = targets[sample_index].to(device=device, dtype=dtype)
        stagea = stagea_references[sample_index].to(device=device, dtype=dtype)
        z_m8a = val_cache.load_batch(batch_cpu, device=device, dtype=dtype)

        with torch.autocast(
            device_type=device.type,
            dtype=dtype if autocast_enabled else torch.float32,
            enabled=autocast_enabled,
        ):
            m8orig = decode_reae_clip(
                reae,
                z_m8a,
                output_frames=int(target.shape[1]),
                clamp=True,
            )
            a1_rgb = a1(
                z_m8a,
                output_frames=int(target.shape[1]),
                clamp=True,
            )
            d76_rgb = (
                decoder76(z_m8a, output_frames=int(target.shape[1]), clamp=True)
                if decoder76 is not None
                else None
            )

        m8orig_gt.update(m8orig, target, clamp=True)
        a1_gt.update(a1_rgb, target, clamp=True)
        m8orig_stagea.update(m8orig, stagea, clamp=True)
        a1_m8orig.update(a1_rgb, m8orig, clamp=True)
        a1_stagea.update(a1_rgb, stagea, clamp=True)
        if d76_rgb is not None:
            assert d76_gt is not None and d76_m8orig is not None
            d76_gt.update(d76_rgb, target, clamp=True)
            d76_m8orig.update(d76_rgb, m8orig, clamp=True)

        item = {
            "m8a_original": m8orig.float().cpu(),
            "m8a_m9a1": a1_rgb.float().cpu(),
        }
        if d76_rgb is not None:
            item["m8a_decoder76"] = d76_rgb.float().cpu()
        outputs.append(item)
        print(f"[M8] val view {sample_index + 1:02d}/13", flush=True)

    metrics: dict[str, object] = {
        "m8a_original_vs_gt": _metric_dict(m8orig_gt),
        "m8a_m9a1_vs_gt": _metric_dict(a1_gt),
        "m8a_original_vs_stagea_original": _metric_dict(m8orig_stagea),
        "m9a1_vs_m8a_original_same_latent": _metric_dict(a1_m8orig),
        "m8a_m9a1_vs_stagea_original": _metric_dict(a1_stagea),
    }
    if d76_gt is not None and d76_m8orig is not None:
        metrics["m8a_decoder76_vs_gt"] = _metric_dict(d76_gt)
        metrics["decoder76_vs_m8a_original_same_latent"] = _metric_dict(d76_m8orig)

    reae.to("cpu")
    a1.to("cpu")
    if decoder76 is not None:
        decoder76.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs, metrics


def main() -> int:
    args = build_parser().parse_args()
    if args.panel_size <= 0:
        raise ValueError("--panel-size must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = DTYPES[args.dtype]

    run_root = args.a1_run.expanduser().resolve()
    config = _read_json(run_root / formal.RUN_CONFIG_FILENAME)
    base_root = (
        args.base_checkpoint.expanduser().resolve()
        if args.base_checkpoint is not None
        else Path(str(_required(config, "base_checkpoint"))).expanduser().resolve()
    )
    stagea_root = args.stagea_checkpoint.expanduser().resolve()
    a1_checkpoint = args.a1_checkpoint.expanduser().resolve()
    decoder76_checkpoint = (
        None
        if args.decoder76_checkpoint is None
        else args.decoder76_checkpoint.expanduser().resolve()
    )
    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    frame_indices = [int(value) for value in args.frame_indices.split(",") if value.strip()]
    dataset, val_cache = _build_val_dataset(
        config,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
    )
    print(
        json.dumps(
            {
                "val_views": len(dataset),
                "val_cache": str(val_cache.root),
                "val_cache_source_checkpoint": val_cache.metadata.get("source_checkpoint"),
                "stagea_checkpoint": str(stagea_root),
                "a1_checkpoint": str(a1_checkpoint),
                "decoder76_checkpoint": (
                    None if decoder76_checkpoint is None else str(decoder76_checkpoint)
                ),
                "base_checkpoint": str(base_root),
                "frame_indices": frame_indices,
            },
            indent=2,
        ),
        flush=True,
    )

    stagea_refs, targets, lq_inputs, identities, stagea_metrics = _stagea_reference_pass(
        dataset,
        base_root=base_root,
        stagea_root=stagea_root,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.stagea_subfolder,
        device=device,
        dtype=dtype,
        attention_backend=args.attention_backend,
    )
    compressed_outputs, compressed_metrics = _compressed_pass(
        dataset,
        val_cache,
        stagea_refs,
        targets,
        base_root=base_root,
        reae_filename=args.reae_filename,
        a1_checkpoint=a1_checkpoint,
        decoder76_checkpoint=decoder76_checkpoint,
        device=device,
        dtype=dtype,
    )

    all_metrics: dict[str, object] = {
        "stagea_original_vs_gt": stagea_metrics,
        **compressed_metrics,
    }

    for sample_index in range(len(dataset)):
        sample_root = output_root / f"sample_{sample_index:02d}"
        methods: list[tuple[str, torch.Tensor]] = [
            ("GT", targets[sample_index]),
            ("LQ-up", lq_inputs[sample_index]),
            ("StageA+Orig", stagea_refs[sample_index]),
            ("M8A+Orig", compressed_outputs[sample_index]["m8a_original"]),
            ("M8A+M9A1", compressed_outputs[sample_index]["m8a_m9a1"]),
        ]
        if "m8a_decoder76" in compressed_outputs[sample_index]:
            methods.append(
                ("M8A+D76", compressed_outputs[sample_index]["m8a_decoder76"])
            )

        for label, video in methods:
            safe = (
                label.lower()
                .replace("+", "_")
                .replace("-", "_")
                .replace(" ", "_")
            )
            _save_video_pngs(video, sample_root / safe)

        max_frame = min(int(video.shape[1]) for _, video in methods) - 1
        selected = [index for index in frame_indices if 0 <= index <= max_frame]
        if not selected:
            selected = [max_frame // 2]
        _make_contact_sheet(
            methods,
            selected,
            sample_root / "contact_sheet.png",
            panel_size=args.panel_size,
        )
        comparison_dir = sample_root / "comparison_frames"
        _make_comparison_frames(
            methods,
            comparison_dir,
            panel_size=args.panel_size,
        )
        _encode_mp4(
            comparison_dir,
            sample_root / "comparison.mp4",
            args.fps,
        )

        identity = identities[sample_index]
        (sample_root / "identity.json").write_text(
            json.dumps(identity, indent=2, default=str),
            encoding="utf-8",
        )

    report = {
        "kind": "m9_val13_component_visual_comparison",
        "val_views": len(dataset),
        "run_config": str(run_root / formal.RUN_CONFIG_FILENAME),
        "m8a_val_cache": str(val_cache.root),
        "m8a_cache_source_checkpoint": val_cache.metadata.get("source_checkpoint"),
        "stagea_checkpoint": str(stagea_root),
        "base_checkpoint": str(base_root),
        "a1_checkpoint": str(a1_checkpoint),
        "decoder76_checkpoint": (
            None if decoder76_checkpoint is None else str(decoder76_checkpoint)
        ),
        "metrics": all_metrics,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n========== M9 val13 component comparison ==========")
    for name, values in all_metrics.items():
        if isinstance(values, Mapping):
            print(
                f"{name:40s} "
                f"PSNR={values.get('psnr')} "
                f"SSIM={values.get('ssim')} "
                f"MAE={values.get('mae')}"
            )
    print(f"Saved: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
