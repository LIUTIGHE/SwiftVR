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
    SOURCE_IDENTITY_METHOD,
    record_source_uid,
    select_distillation_indices,
    selected_indices_sha256,
)
from swiftvr.training.forward import prepare_training_batch
from swiftvr.training.input_pipeline import dataloader_worker_kwargs
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
        help=(
            "source_balanced covers distinct resolved HR sources before taking "
            "additional degradation records or views from the same source"
        ),
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
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
    worker_kwargs = dataloader_worker_kwargs(
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
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
    path_root = args.path_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Teacher cache is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir()

    # HQ is not consumed by the endpoint teacher-velocity forward. Keep HR loaded
    # because prepare_training_batch uses the target geometry to construct the
    # exact same LQ input used by distillation training.
    base_dataset = TripletVideoDataset(
        args.manifest,
        split=args.split,
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
    full_dataset = DeterministicTripletViewDataset(
        base_dataset,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
    )
    record_source_uids = tuple(record_source_uid(record) for record in base_dataset.records)
    selected_indices = select_distillation_indices(
        len(full_dataset),
        max_samples=args.max_samples,
        mode=args.selection_mode,
        seed=args.selection_seed,
        source_uids=record_source_uids,
        views_per_record=args.views_per_record,
    )
    selected_record_indices = {
        int(index) // int(args.views_per_record) for index in selected_indices
    }
    selected_source_uids = {
        record_source_uids[index] for index in selected_record_indices
    }
    dataset = Subset(full_dataset, list(selected_indices))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
        **worker_kwargs,
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
            distillation_index = int(identity["distillation_index"])
            record_index = distillation_index // int(args.views_per_record)
            if record_index >= len(base_dataset.records):
                raise RuntimeError(
                    f"distillation_index={distillation_index} maps beyond "
                    f"{len(base_dataset.records)} records"
                )
            record = base_dataset.records[record_index]
            source_uid = record_source_uids[record_index]
            velocity = velocities[local_index].detach().to(
                device="cpu", dtype=torch.float16
            ).contiguous()
            relative_file = (
                f"samples/{distillation_index:08d}_"
                f"{identity['key']}.safetensors"
            )
            save_file({"velocity": velocity}, str(output / relative_file))
            saved_samples.append(
                {
                    **identity,
                    "source_uid": source_uid,
                    "source_hr_first": record.hr_paths[0],
                    "source_hr_last": record.hr_paths[-1],
                    "file": relative_file,
                }
            )
            processed += 1
            print(
                f"cached {processed}/{len(selected_indices)}: "
                f"dataset_index={distillation_index} "
                f"{identity['record_uid']} view={identity['view_index']} "
                f"source={source_uid[:12]}",
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
        "path_root": str(path_root),
        "split": args.split,
        "clip_length": int(args.clip_length),
        "crop_size": int(args.crop_size),
        "scale": int(args.scale),
        "views_per_record": int(args.views_per_record),
        "view_seed": int(args.view_seed),
        "horizontal_flip_probability": float(args.horizontal_flip_probability),
        "vertical_flip_probability": float(args.vertical_flip_probability),
        "base_record_count": len(base_dataset),
        "unique_source_count": len(set(record_source_uids)),
        "full_dataset_length": len(full_dataset),
        "selection_mode": args.selection_mode,
        "selection_seed": int(args.selection_seed),
        "selected_indices": list(selected_indices),
        "selected_indices_sha256": selected_indices_sha256(selected_indices),
        "selected_record_count": len(selected_record_indices),
        "selected_unique_source_count": len(selected_source_uids),
        "sample_count": len(saved_samples),
        "samples": saved_samples,
    }
    _write_json(output / TEACHER_CACHE_METADATA_FILENAME, metadata)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "cache_root": str(output),
                "selected_samples": len(saved_samples),
                "selected_records": len(selected_record_indices),
                "selected_unique_sources": len(selected_source_uids),
                "full_dataset_length": len(full_dataset),
                "selection_mode": args.selection_mode,
                "selected_indices_sha256": metadata["selected_indices_sha256"],
                "storage_dtype": metadata["storage_dtype"],
                "elapsed_seconds": elapsed,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
