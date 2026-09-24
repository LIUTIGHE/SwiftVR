#!/usr/bin/env python3
"""Cache D1536 TA endpoint velocities for materialized UltraVideo LR views.

This builder consumes the LR-only safetensors manifest produced by
materialize/merge_ultravideo_lr_materialization.py. Each manifest row is already
one fixed distillation view, so there is no second deterministic-view expansion.
No decoder, HQ, or HR tensor participates in target construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
for search_root in (ROOT, TOOLS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from tools import build_b2a_stage_a_teacher_cache as cache_base
from tools.build_b2b_ta_teacher_cache import (
    EXPECTED_TA_SHAPE,
    TA_CACHE_KIND,
    _checkpoint_hashes,
)
from tools.smoke_training_forward import (
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.data import UltraVideoMaterializedLRDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training.b2a_width import (
    B2ACompactVelocityDistillationForward,
    transformer_width_shape,
)
from swiftvr.training.distillation import (
    TEACHER_CACHE_FORMAT_VERSION,
    TEACHER_CACHE_METADATA_FILENAME,
    distillation_sample_identity,
)
from swiftvr.training.reference import sha256_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--materialized-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    p.add_argument("--allow-dtype-mismatch", action="store_true")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument(
        "--cache-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress-every", type=int, default=50)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "progress_every"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be positive")


def _dataset_geometry(dataset: UltraVideoMaterializedLRDataset) -> dict[str, int]:
    first = dataset.rows[0]
    clip_length = len(first["frame_indices"])
    crop_size = int(first["crop_size"])
    scale = int(first["scale"])
    target_height = int(first["target_height"])
    target_width = int(first["target_width"])
    expected_indices = list(range(len(dataset)))
    actual_indices = [int(row.get("materialized_index", index)) for index, row in enumerate(dataset.rows)]
    if actual_indices != expected_indices:
        raise ValueError(
            "UltraVideo teacher cache requires the merged materialized manifest "
            "with contiguous materialized_index 0..N-1"
        )
    for row in dataset.rows[1:]:
        current = (
            len(row["frame_indices"]),
            int(row["crop_size"]),
            int(row["scale"]),
            int(row["target_height"]),
            int(row["target_width"]),
        )
        expected = (clip_length, crop_size, scale, target_height, target_width)
        if current != expected:
            raise ValueError(
                f"Materialized views must share training geometry; got {current} vs {expected}"
            )
    return {
        "clip_length": clip_length,
        "crop_size": crop_size,
        "scale": scale,
        "target_height": target_height,
        "target_width": target_width,
    }


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    rank = 0
    distributed = False
    try:
        rank, _local_rank, world_size, device, distributed = cache_base._init_execution(
            args.device
        )

        base_root = args.base_checkpoint.expanduser().resolve()
        teacher_root = args.teacher_checkpoint.expanduser().resolve()
        manifest = args.materialized_manifest.expanduser().resolve()
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
        if (
            runtime_dtype == torch.bfloat16
            and device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("Selected GPU does not support BF16")
        cache_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.cache_dtype]
        teacher_hashes = _checkpoint_hashes(
            teacher_root, args.transformer_subfolder
        )

        output = args.output_dir.expanduser().resolve()
        cache_base._prepare_output(
            output,
            overwrite=args.overwrite,
            rank=rank,
            distributed=distributed,
        )

        dataset = UltraVideoMaterializedLRDataset(
            manifest,
            verify_paths=args.verify_paths,
        )
        geometry = _dataset_geometry(dataset)
        sample_limit = (
            len(dataset)
            if args.max_samples is None
            else min(len(dataset), int(args.max_samples))
        )
        if sample_limit <= 0:
            raise RuntimeError("No UltraVideo materialized views selected for caching")

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
            raise ValueError(
                f"TA checkpoint shape mismatch: {teacher_shape} != {EXPECTED_TA_SHAPE}"
            )
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
                f"UltraVideo D1536 TA cache: mode={mode} world_size={world_size} "
                f"samples={sample_limit} runtime_dtype="
                f"{str(runtime_dtype).removeprefix('torch.')} "
                f"cache_dtype={args.cache_dtype}",
                flush=True,
            )

        local_samples: list[dict[str, object]] = []
        local_processed = 0
        local_expected = len(indices)
        started = time.perf_counter()
        autocast_enabled = runtime_dtype in (torch.float16, torch.bfloat16)

        with torch.no_grad():
            for batch_cpu in loader:
                batch = move_video_batch(
                    batch_cpu,
                    device=device,
                    dtype=runtime_dtype,
                )
                with torch.autocast(
                    "cuda",
                    dtype=runtime_dtype,
                    enabled=device.type == "cuda" and autocast_enabled,
                ):
                    output_batch = closure(batch)
                velocities = output_batch["velocity"]
                for local_index in range(int(velocities.shape[0])):
                    identity = distillation_sample_identity(
                        batch_cpu,
                        local_index,
                    )
                    global_index = int(identity["distillation_index"])
                    if global_index not in indices:
                        raise RuntimeError(
                            f"Rank {rank} received unassigned global index {global_index}"
                        )
                    velocity = (
                        velocities[local_index]
                        .detach()
                        .to(device="cpu", dtype=cache_dtype)
                        .contiguous()
                    )
                    relative_file = (
                        f"samples/{global_index:08d}_{identity['key']}.safetensors"
                    )
                    save_file(
                        {"velocity": velocity},
                        str(output / relative_file),
                    )
                    local_samples.append(
                        {**identity, "file": relative_file}
                    )
                    local_processed += 1
                    if (
                        local_processed == 1
                        or local_processed % args.progress_every == 0
                        or local_processed == local_expected
                    ):
                        elapsed = time.perf_counter() - started
                        print(
                            f"rank={rank} cached {local_processed}/{local_expected} "
                            f"({local_processed / max(elapsed, 1e-9):.3f} samples/s): "
                            f"global_index={global_index} "
                            f"{identity['record_uid']} view={identity['view_index']}",
                            flush=True,
                        )

        if local_processed != local_expected:
            raise RuntimeError(
                f"Rank {rank} cached {local_processed} samples, expected {local_expected}"
            )

        elapsed = time.perf_counter() - started
        if distributed:
            elapsed_tensor = torch.tensor(
                [elapsed], device=device, dtype=torch.float64
            )
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
            saved_samples = cache_base._validate_merged_samples(
                saved_samples, sample_limit
            )
            variants = sorted(
                {str(row["variant"]) for row in dataset.rows[:sample_limit]}
            )
            metadata: dict[str, object] = {
                "format_version": TEACHER_CACHE_FORMAT_VERSION,
                "kind": TA_CACHE_KIND,
                "dataset_kind": "ultravideo_materialized_lr_v1",
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
                "materialized_manifest": str(manifest),
                "materialized_manifest_sha256": sha256_file(manifest),
                "variants": variants,
                **geometry,
                "views_per_record": 1,
                "base_record_count": len(dataset),
                "full_dataset_length": len(dataset),
                "sample_count": len(saved_samples),
                "elapsed_seconds": elapsed,
                "distributed_world_size": world_size,
                "distributed_sharding": "strided_global_index_no_padding",
                "samples": saved_samples,
            }
            cache_base._write_json(
                output / TEACHER_CACHE_METADATA_FILENAME,
                metadata,
            )
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in metadata.items()
                        if key != "samples"
                    },
                    indent=2,
                )
            )

        if distributed:
            dist.barrier()
        return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
