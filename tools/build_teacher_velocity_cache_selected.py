#!/usr/bin/env python3
"""Cache a deterministic selected subset of conditional SwiftVR teacher velocities."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset

from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer import WanTransformer3DModel
from swiftvr.training.distillation import (
    TEACHER_CACHE_FORMAT_VERSION,
    TEACHER_CACHE_METADATA_FILENAME,
    DeterministicTripletViewDataset,
    conditional_teacher_velocity,
    distillation_sample_identity,
)
from swiftvr.training.distillation_generalization import (
    SELECTION_MODES,
    select_distillation_indices,
    selected_indices_sha256,
)
from swiftvr.training.forward import prepare_training_batch
from swiftvr.training.reference import sha256_file

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--views-per-record", type=int, default=1)
    parser.add_argument("--view-seed", type=int, default=0)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.0)
    parser.add_argument(
        "--selection-mode",
        choices=tuple(sorted(SELECTION_MODES)),
        default="all",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if args.views_per_record <= 0 or args.batch_size <= 0:
        raise ValueError("views-per-record and batch-size must be positive")
    for name, value in (
        ("horizontal-flip-probability", args.horizontal_flip_probability),
        ("vertical-flip-probability", args.vertical_flip_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = DTYPES[args.dtype]
    if (
        dtype == torch.bfloat16
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    reference_root = args.reference_checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Teacher cache is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir()

    base_dataset = TripletVideoDataset(
        args.manifest,
        split=args.split,
        training=True,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        horizontal_flip_probability=args.horizontal_flip_probability,
        vertical_flip_probability=args.vertical_flip_probability,
        drop_short_sequences=True,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
    )
    full_dataset = DeterministicTripletViewDataset(
        base_dataset,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
    )
    selected_indices = select_distillation_indices(
        len(full_dataset),
        max_samples=args.max_samples,
        mode=args.selection_mode,
        seed=args.selection_seed,
    )
    dataset = Subset(full_dataset, list(selected_indices))
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

    reae_path = reference_root / args.reae_filename
    reae = ReAE(str(reae_path)).to(device=device, dtype=dtype).eval()
    transformer = WanTransformer3DModel.from_pretrained(
        str(reference_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype).eval()
    transformer.prepare_for_inference(attention_backend=args.attention_backend)

    saved_samples: list[dict[str, object]] = []
    processed = 0
    started = time.perf_counter()
    for batch_cpu in loader:
        batch = _move_batch(batch_cpu, device, dtype)
        prepared = prepare_training_batch(batch)
        lq_input = prepared["lq_input"]
        if not isinstance(lq_input, torch.Tensor):
            raise TypeError("Prepared batch is missing lq_input")
        teacher = conditional_teacher_velocity(
            reae=reae,
            transformer=transformer,
            prompt_embedding=prompt_embedding,
            lq_input=lq_input,
            timestep=args.timestep,
        )
        velocities = teacher["velocity"]

        for local_index in range(int(velocities.shape[0])):
            identity = distillation_sample_identity(batch_cpu, local_index)
            velocity = velocities[local_index].detach().to(
                device="cpu", dtype=torch.float16
            ).contiguous()
            relative_file = (
                f"samples/{identity['distillation_index']:08d}_"
                f"{identity['key']}.safetensors"
            )
            save_file({"velocity": velocity}, str(output / relative_file))
            saved_samples.append({**identity, "file": relative_file})
            processed += 1
            print(
                f"cached {processed}/{len(selected_indices)}: "
                f"dataset_index={identity['distillation_index']} "
                f"{identity['record_uid']} view={identity['view_index']}",
                flush=True,
            )

    if processed != len(selected_indices):
        raise RuntimeError(f"Cached {processed}, expected {len(selected_indices)}")

    manifests = [path.expanduser().resolve() for path in args.manifest]
    metadata: dict[str, object] = {
        "format_version": TEACHER_CACHE_FORMAT_VERSION,
        "kind": "swiftvr_endpoint_teacher_velocity",
        "reference_checkpoint": str(reference_root),
        "prompt_embedding_file": str(prompt_path),
        "prompt_embedding_sha256": sha256_file(prompt_path),
        "prompt_key": args.prompt_key,
        "reae_file": str(reae_path),
        "reae_sha256": sha256_file(reae_path),
        "timestep": float(args.timestep),
        "dtype": args.dtype,
        "storage_dtype": "float16",
        "attention_backend": args.attention_backend,
        "manifests": [str(path) for path in manifests],
        "manifest_sha256": {str(path): sha256_file(path) for path in manifests},
        "split": args.split,
        "clip_length": int(args.clip_length),
        "crop_size": int(args.crop_size),
        "scale": int(args.scale),
        "views_per_record": int(args.views_per_record),
        "view_seed": int(args.view_seed),
        "horizontal_flip_probability": float(args.horizontal_flip_probability),
        "vertical_flip_probability": float(args.vertical_flip_probability),
        "base_record_count": len(base_dataset),
        "full_dataset_length": len(full_dataset),
        "selection_mode": args.selection_mode,
        "selection_seed": int(args.selection_seed),
        "selected_indices": list(selected_indices),
        "selected_indices_sha256": selected_indices_sha256(selected_indices),
        "sample_count": processed,
        "elapsed_seconds": time.perf_counter() - started,
        "samples": saved_samples,
    }
    _write_json(output / TEACHER_CACHE_METADATA_FILENAME, metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
