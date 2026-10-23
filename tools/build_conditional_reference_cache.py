#!/usr/bin/env python3
"""Cache the original fixed-prompt SwiftVR conditional reference on validation clips."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer import WanTransformer3DModel
from swiftvr.training.forward import prepare_training_batch
from swiftvr.training.reference import (
    CACHE_FORMAT_VERSION,
    CACHE_METADATA_FILENAME,
    batch_sample_identity,
    conditional_reference_forward,
    sha256_file,
)
from swiftvr.training.stage3 import VideoMetricAccumulator, temporal_difference_mse


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--prompt-embedding-filename", default="prompt_embedding.safetensors")
    parser.add_argument("--prompt-key", default="prompt_emb")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _move_batch(batch: dict[str, object], device: torch.device, dtype: torch.dtype):
    result = dict(batch)
    for key in ("lr", "hq", "hr"):
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, dtype=dtype, non_blocking=True)
    return result


def main() -> int:
    args = build_parser().parse_args()
    if args.max_samples <= 0 or args.batch_size <= 0:
        raise ValueError("max-samples and batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = DTYPES[args.dtype]

    reference_root = args.reference_checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Reference cache is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    samples_dir = output / "samples"
    samples_dir.mkdir()

    dataset = TripletVideoDataset(
        args.val_manifest,
        split=args.val_split,
        training=False,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        drop_short_sequences=True,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    prompt_path = reference_root / args.prompt_embedding_filename
    prompt_payload = load_file(str(prompt_path), device="cpu")
    if args.prompt_key not in prompt_payload:
        raise KeyError(f"{prompt_path} does not contain key {args.prompt_key!r}")
    prompt_embedding = prompt_payload[args.prompt_key]

    reae = ReAE(str(reference_root / args.reae_filename)).to(device=device, dtype=dtype).eval()
    transformer = WanTransformer3DModel.from_pretrained(
        str(reference_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype).eval()
    transformer.prepare_for_inference(attention_backend=args.attention_backend)

    metric = VideoMetricAccumulator()
    pixel_sum = 0.0
    temporal_sum = 0.0
    saved_samples: list[dict[str, object]] = []
    processed = 0
    started = time.perf_counter()

    for batch_cpu in loader:
        if processed >= args.max_samples:
            break
        batch = _move_batch(batch_cpu, device, dtype)
        prepared = prepare_training_batch(batch)
        lq_input = prepared["lq_input"]
        target = prepared["target"]
        if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("Prepared validation batch is missing lq_input/target")
        reference = conditional_reference_forward(
            reae=reae,
            transformer=transformer,
            prompt_embedding=prompt_embedding,
            lq_input=lq_input,
            output_frames=int(target.shape[1]),
            timestep=args.timestep,
        )
        prediction = reference["prediction"]
        velocity = reference["velocity"]

        for local_index in range(int(prediction.shape[0])):
            if processed >= args.max_samples:
                break
            identity = batch_sample_identity(batch_cpu, local_index)
            pred_i = prediction[local_index : local_index + 1]
            target_i = target[local_index : local_index + 1]
            velocity_i = velocity[local_index]
            metric.update(pred_i, target_i, clamp=True)
            pixel_l1 = float(torch.nn.functional.l1_loss(pred_i.float(), target_i.float()).item())
            temporal_mse = float(temporal_difference_mse(pred_i, target_i).item())
            pixel_sum += pixel_l1
            temporal_sum += temporal_mse

            relative_file = f"samples/{processed:05d}_{identity['key']}.safetensors"
            save_file(
                {
                    "prediction": pred_i[0].detach().to(device="cpu", dtype=torch.float16).contiguous(),
                    "velocity": velocity_i.detach().to(device="cpu", dtype=torch.float16).contiguous(),
                },
                str(output / relative_file),
            )
            saved_samples.append(
                {
                    **identity,
                    "file": relative_file,
                    "pixel_l1_vs_gt": pixel_l1,
                    "temporal_mse_vs_gt": temporal_mse,
                }
            )
            processed += 1
            print(
                f"cached {processed}/{min(args.max_samples, len(dataset))}: "
                f"{identity['record_uid']}",
                flush=True,
            )

    if processed == 0:
        raise RuntimeError("Reference cache produced no samples")
    reference_gt = metric.compute()
    reference_gt["pixel_l1"] = pixel_sum / processed
    reference_gt["temporal_mse"] = temporal_sum / processed
    manifests = [path.expanduser().resolve() for path in args.val_manifest]
    metadata: dict[str, object] = {
        "format_version": CACHE_FORMAT_VERSION,
        "kind": "swiftvr_fixed_empty_prompt_conditional_reference",
        "reference_checkpoint": str(reference_root),
        "prompt_embedding_file": str(prompt_path),
        "prompt_embedding_sha256": sha256_file(prompt_path),
        "prompt_key": args.prompt_key,
        "timestep": float(args.timestep),
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "val_manifests": [str(path) for path in manifests],
        "val_manifest_sha256": {str(path): sha256_file(path) for path in manifests},
        "val_split": args.val_split,
        "clip_length": int(args.clip_length),
        "crop_size": int(args.crop_size),
        "scale": int(args.scale),
        "sample_count": processed,
        "reference_gt": reference_gt,
        "elapsed_seconds": time.perf_counter() - started,
        "samples": saved_samples,
    }
    _write_json(output / CACHE_METADATA_FILENAME, metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
