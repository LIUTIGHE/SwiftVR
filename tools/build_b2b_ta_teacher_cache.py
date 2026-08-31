#!/usr/bin/env python3
"""Cache B2-A D1536 endpoint velocities for D768 teaching-assistant distillation.

The cache is built on the same deterministic triplet views used by D3.  The
original folded checkpoint supplies only the frozen ReAE encoder; the teacher
Transformer is loaded from a full B2-A compact checkpoint.

No decoder or GT supervision participates in target construction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset

from tools import build_b2a_stage_a_teacher_cache as cache_base
from smoke_training_forward import move_video_batch, resolve_runtime_dtype, validate_folded_checkpoint
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training.b2a_width import B2ACompactVelocityDistillationForward, transformer_width_shape
from swiftvr.training.distillation import (
    TEACHER_CACHE_FORMAT_VERSION,
    TEACHER_CACHE_METADATA_FILENAME,
    DeterministicTripletViewDataset,
    distillation_sample_identity,
)
from swiftvr.training.reference import sha256_file


TA_CACHE_KIND = "swiftvr_b2b_d1536_ta_velocity"
EXPECTED_TA_SHAPE = {
    "hidden_dim": 1536,
    "num_heads": 12,
    "head_dim": 128,
    "ffn_dim": 8960,
    "num_layers": 30,
    "adapter_dim": 128,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--split", default="train")
    p.add_argument("--clip-length", type=int, default=13)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--views-per-record", type=int, default=8)
    p.add_argument("--view-seed", type=int, default=20260805)
    p.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    p.add_argument("--vertical-flip-probability", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--allow-dtype-mismatch", action="store_true")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--cache-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress-every", type=int, default=50)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    cache_base._validate_args(args)


def _checkpoint_hashes(root: Path, transformer_subfolder: str) -> dict[str, str]:
    transformer_root = root / transformer_subfolder
    config = transformer_root / "config.json"
    weights = transformer_root / "diffusion_pytorch_model.safetensors"
    missing = [str(path) for path in (config, weights) if not path.is_file()]
    if missing:
        raise FileNotFoundError("TA checkpoint is incomplete: " + ", ".join(missing))
    return {
        "teacher_config_sha256": sha256_file(config),
        "teacher_weights_sha256": sha256_file(weights),
    }


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    rank = 0
    distributed = False
    try:
        rank, _local_rank, world_size, device, distributed = cache_base._init_execution(args.device)

        base_root = args.base_checkpoint.expanduser().resolve()
        teacher_root = args.teacher_checkpoint.expanduser().resolve()
        folded_config = validate_folded_checkpoint(
            base_root,
            reae_filename=args.reae_filename,
            transformer_subfolder=args.transformer_subfolder,
        )
        runtime_dtype = resolve_runtime_dtype(
            args.dtype,
            folded_config,
            device,
            allow_mismatch=args.allow_dtype_mismatch,
        )
        if runtime_dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support BF16")
        cache_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.cache_dtype]
        teacher_hashes = _checkpoint_hashes(teacher_root, args.transformer_subfolder)

        output = args.output_dir.expanduser().resolve()
        cache_base._prepare_output(
            output,
            overwrite=args.overwrite,
            rank=rank,
            distributed=distributed,
        )

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
        dataset = DeterministicTripletViewDataset(
            base_dataset,
            views_per_record=args.views_per_record,
            view_seed=args.view_seed,
        )
        sample_limit = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))
        if sample_limit <= 0:
            raise RuntimeError("No deterministic views selected for caching")

        indices = cache_base._rank_indices(sample_limit, rank, world_size)
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        reae = ReAE(str(base_root / args.reae_filename))
        teacher = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
            str(teacher_root),
            subfolder=args.transformer_subfolder,
            torch_dtype=runtime_dtype,
            low_cpu_mem_usage=True,
        )
        teacher_shape = transformer_width_shape(teacher)
        if teacher_shape != EXPECTED_TA_SHAPE:
            raise ValueError(f"TA checkpoint shape mismatch: {teacher_shape} != {EXPECTED_TA_SHAPE}")
        for parameter in reae.parameters():
            parameter.requires_grad_(False)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        reae.to(device=device, dtype=runtime_dtype).eval()
        teacher.to(device=device, dtype=runtime_dtype).eval()
        closure = B2ACompactVelocityDistillationForward(
            reae,
            teacher,
            attention_backend=args.attention_backend,
            gradient_checkpointing=False,
        ).eval()
        closure.reae.eval()

        if rank == 0:
            mode = "distributed" if distributed else "single-process"
            print(
                f"D1536 TA cache: mode={mode} world_size={world_size} samples={sample_limit} "
                f"runtime_dtype={str(runtime_dtype).removeprefix('torch.')} cache_dtype={args.cache_dtype}",
                flush=True,
            )

        local_samples: list[dict[str, object]] = []
        local_processed = 0
        local_expected = len(indices)
        started = time.perf_counter()
        autocast_enabled = runtime_dtype in (torch.float16, torch.bfloat16)
        with torch.no_grad():
            for batch_cpu in loader:
                batch = move_video_batch(batch_cpu, device=device, dtype=runtime_dtype)
                with torch.autocast(
                    "cuda",
                    dtype=runtime_dtype,
                    enabled=device.type == "cuda" and autocast_enabled,
                ):
                    output_batch = closure(batch)
                velocities = output_batch["velocity"]
                for local_index in range(int(velocities.shape[0])):
                    identity = distillation_sample_identity(batch_cpu, local_index)
                    global_index = int(identity["distillation_index"])
                    if global_index not in indices:
                        raise RuntimeError(f"Rank {rank} received unassigned global index {global_index}")
                    velocity = velocities[local_index].detach().to(device="cpu", dtype=cache_dtype).contiguous()
                    relative_file = f"samples/{global_index:08d}_{identity['key']}.safetensors"
                    save_file({"velocity": velocity}, str(output / relative_file))
                    local_samples.append({**identity, "file": relative_file})
                    local_processed += 1
                    if local_processed == 1 or local_processed % args.progress_every == 0 or local_processed == local_expected:
                        elapsed = time.perf_counter() - started
                        print(
                            f"rank={rank} cached {local_processed}/{local_expected} "
                            f"({local_processed / max(elapsed, 1e-9):.3f} samples/s): "
                            f"global_index={global_index} {identity['record_uid']} view={identity['view_index']}",
                            flush=True,
                        )

        if local_processed != local_expected:
            raise RuntimeError(f"Rank {rank} cached {local_processed} samples, expected {local_expected}")

        elapsed = time.perf_counter() - started
        if distributed:
            elapsed_tensor = torch.tensor([elapsed], device=device, dtype=torch.float64)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            elapsed = float(elapsed_tensor.item())

        saved_samples = cache_base._gather_samples(
            local_samples,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        )
        if distributed:
            dist.barrier()

        if rank == 0:
            assert saved_samples is not None
            saved_samples = cache_base._validate_merged_samples(saved_samples, sample_limit)
            manifests = [path.expanduser().resolve() for path in args.manifest]
            metadata: dict[str, object] = {
                "format_version": TEACHER_CACHE_FORMAT_VERSION,
                "kind": TA_CACHE_KIND,
                "teacher_role": "b2a_d1536_teaching_assistant",
                "teacher_checkpoint": str(teacher_root),
                "teacher_shape": teacher_shape,
                **teacher_hashes,
                "base_checkpoint": str(base_root),
                "reae_file": str(base_root / args.reae_filename),
                "reae_sha256": sha256_file(base_root / args.reae_filename),
                "runtime_dtype": str(runtime_dtype).removeprefix("torch."),
                "dtype": args.cache_dtype,
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
                "full_dataset_length": len(dataset),
                "sample_count": len(saved_samples),
                "elapsed_seconds": elapsed,
                "distributed_world_size": world_size,
                "distributed_sharding": "strided_global_index_no_padding",
                "samples": saved_samples,
            }
            cache_base._write_json(output / TEACHER_CACHE_METADATA_FILENAME, metadata)
            print(json.dumps({key: value for key, value in metadata.items() if key != "samples"}, indent=2))

        if distributed:
            dist.barrier()
        return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
