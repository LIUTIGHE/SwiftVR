#!/usr/bin/env python3
"""Cache deterministic long-run Stage-A ``z_SR`` latents for Tiny Decoder training."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training import cast_trainable_parameters, load_delta_checkpoint
from swiftvr.training.distillation import (
    DeterministicTripletViewDataset,
    SwiftVRVelocityDistillationForward,
    distillation_sample_identity,
)
from swiftvr.training.distillation_generalization import (
    SELECTION_MODES,
    SOURCE_IDENTITY_METHOD,
    record_source_uid,
    select_distillation_indices,
    selected_indices_sha256,
)
from swiftvr.training.forward import encode_reae_clip, prepare_training_batch
from swiftvr.training.input_pipeline import dataloader_worker_kwargs
from swiftvr.training.reference import extract_transformer_sample, sha256_file
from swiftvr.training.tiny_decoder_cache import (
    TINY_DECODER_CACHE_FORMAT_VERSION,
    TINY_DECODER_CACHE_METADATA_FILENAME,
)


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--views-per-record", type=int, default=2)
    parser.add_argument("--view-seed", type=int, default=20260805)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.0)
    parser.add_argument(
        "--selection-mode",
        choices=tuple(sorted(SELECTION_MODES)),
        default="all",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _move_batch(batch: dict[str, object], device: torch.device, dtype: torch.dtype):
    result = dict(batch)
    for key in ("lr", "hr"):
        value = result.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, dtype=dtype, non_blocking=True)
    return result


def _load_source(
    *,
    base: Path,
    delta: Path,
    device: torch.device,
    dtype: torch.dtype,
    reae_filename: str,
    transformer_subfolder: str,
    attention_backend: str,
) -> SwiftVRVelocityDistillationForward:
    reae = ReAE(str(base / reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(base),
        subfolder=transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    for parameter in reae.parameters():
        parameter.requires_grad_(False)
    for name, parameter in transformer.named_parameters():
        parameter.requires_grad_("prompt_free_adapter" in name)
    source = SwiftVRVelocityDistillationForward(
        reae,
        transformer,
        attention_backend=attention_backend,
        prepare_transformer=False,
    )
    cast_trainable_parameters(source, dtype=torch.float32)
    load_delta_checkpoint(delta, source, strict=True)
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    source.to(device=device)
    source.reae.eval()
    source.transformer.eval()
    source.transformer.prepare_for_inference(
        attention_backend=attention_backend,
        use_torch_compile=False,
    )
    return source


def main() -> int:
    args = build_parser().parse_args()
    if args.views_per_record <= 0 or args.batch_size <= 0:
        raise ValueError("views-per-record and batch-size must be positive")
    if args.crop_size <= 0 or args.scale <= 0 or args.clip_length <= 0:
        raise ValueError("clip-length, crop-size and scale must be positive")
    if args.clip_length % 4 != 1:
        raise ValueError("SwiftVR cache clip-length must satisfy T=4k+1")
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
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")

    base = args.base_checkpoint.expanduser().resolve()
    delta = args.source_checkpoint.expanduser().resolve()
    path_root = args.path_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    source_weights = delta / "trainable.safetensors"
    source_metadata = delta / "metadata.json"
    if not source_weights.is_file() or not source_metadata.is_file():
        raise FileNotFoundError(
            f"source-checkpoint must contain trainable.safetensors and metadata.json: {delta}"
        )
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Tiny-decoder latent cache is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir()

    worker_kwargs = dataloader_worker_kwargs(
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
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
    source_uids = tuple(record_source_uid(record) for record in base_dataset.records)
    selected_indices = select_distillation_indices(
        len(full_dataset),
        max_samples=args.max_samples,
        mode=args.selection_mode,
        seed=args.selection_seed,
        source_uids=source_uids,
        views_per_record=args.views_per_record,
    )
    selected_record_indices = {
        int(index) // int(args.views_per_record) for index in selected_indices
    }
    selected_source_uids = {source_uids[index] for index in selected_record_indices}
    loader = DataLoader(
        Subset(full_dataset, list(selected_indices)),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
        **worker_kwargs,
    )

    source = _load_source(
        base=base,
        delta=delta,
        device=device,
        dtype=dtype,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
        attention_backend=args.attention_backend,
    )

    saved_samples: list[dict[str, object]] = []
    processed = 0
    started = time.perf_counter()
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode():
        for batch_cpu in loader:
            batch = _move_batch(batch_cpu, device, dtype)
            prepared = prepare_training_batch(batch)
            lq_input = prepared["lq_input"]
            if not isinstance(lq_input, torch.Tensor):
                raise TypeError("Prepared batch is missing lq_input")
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=autocast_enabled,
            ):
                z_lq_ntchw = encode_reae_clip(
                    source.reae,
                    lq_input,
                    require_4k_plus_1=True,
                )
                z_lq = z_lq_ntchw.permute(0, 2, 1, 3, 4).contiguous()
                velocity = extract_transformer_sample(
                    source.transformer(z_lq, return_dict=True)
                )
                if velocity.shape != z_lq.shape:
                    raise RuntimeError(
                        f"Source velocity shape {tuple(velocity.shape)} != latent {tuple(z_lq.shape)}"
                    )
                z_sr = (z_lq - velocity).permute(0, 2, 1, 3, 4).contiguous()

            for local_index in range(int(z_sr.shape[0])):
                identity = distillation_sample_identity(batch_cpu, local_index)
                distillation_index = int(identity["distillation_index"])
                record_index = distillation_index // int(args.views_per_record)
                record = base_dataset.records[record_index]
                source_uid = source_uids[record_index]
                latent = z_sr[local_index].detach().to(
                    device="cpu",
                    dtype=torch.float16,
                ).contiguous()
                relative_file = (
                    f"samples/{distillation_index:08d}_{identity['key']}.safetensors"
                )
                save_file({"z_sr": latent}, str(output / relative_file))
                saved_samples.append(
                    {
                        **identity,
                        "source_uid": source_uid,
                        "source_hr_first": record.hr_paths[0],
                        "source_hr_last": record.hr_paths[-1],
                        "z_sr_shape": list(latent.shape),
                        "file": relative_file,
                    }
                )
                processed += 1
                if processed == 1 or processed % 32 == 0 or processed == len(selected_indices):
                    print(
                        f"cached {processed}/{len(selected_indices)}: "
                        f"dataset_index={distillation_index} "
                        f"{identity['record_uid']} view={identity['view_index']}",
                        flush=True,
                    )

    if processed != len(selected_indices):
        raise RuntimeError(f"Cached {processed}, expected {len(selected_indices)}")
    manifests = [path.expanduser().resolve() for path in args.manifest]
    reae_path = base / args.reae_filename
    transformer_config = base / args.transformer_subfolder / "config.json"
    metadata: dict[str, object] = {
        "format_version": TINY_DECODER_CACHE_FORMAT_VERSION,
        "kind": "swiftvr_stage_b1_sr_latent",
        "base_checkpoint": str(base),
        "source_checkpoint": str(delta),
        "source_weights_sha256": sha256_file(source_weights),
        "source_metadata_sha256": sha256_file(source_metadata),
        "reae_file": str(reae_path),
        "reae_sha256": sha256_file(reae_path),
        "transformer_config_sha256": sha256_file(transformer_config),
        "runtime_dtype": args.dtype,
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
        "unique_source_count": len(set(source_uids)),
        "full_dataset_length": len(full_dataset),
        "source_identity_method": SOURCE_IDENTITY_METHOD,
        "selection_mode": args.selection_mode,
        "selection_seed": int(args.selection_seed),
        "selected_indices": list(selected_indices),
        "selected_indices_sha256": selected_indices_sha256(selected_indices),
        "selected_record_count": len(selected_record_indices),
        "selected_unique_source_count": len(selected_source_uids),
        "sample_count": len(saved_samples),
        "samples": saved_samples,
    }
    _write_json(output / TINY_DECODER_CACHE_METADATA_FILENAME, metadata)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "cache_root": str(output),
                "samples": len(saved_samples),
                "records": len(selected_record_indices),
                "unique_sources": len(selected_source_uids),
                "selected_indices_sha256": metadata["selected_indices_sha256"],
                "storage_dtype": "float16",
                "elapsed_seconds": elapsed,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
