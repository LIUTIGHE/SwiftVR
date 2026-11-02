#!/usr/bin/env python3
"""Cache Stage-A D3072 velocities for materialized UltraVideo LR views.

Consumes the LR-only merged materialized manifest. Each row is one fixed
distillation sample. The Stage-A prompt-free/no-time teacher is reconstructed
from the folded base checkpoint plus its adapter delta checkpoint.
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
from tools.build_ultravideo_ta_teacher_cache import _dataset_geometry
from tools.smoke_training_forward import (
    configure_train_scope,
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.data import UltraVideoMaterializedLRDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training import (
    SwiftVRVelocityDistillationForward,
    cast_trainable_parameters,
    load_delta_checkpoint,
)
from swiftvr.training.distillation import (
    TEACHER_CACHE_FORMAT_VERSION,
    TEACHER_CACHE_METADATA_FILENAME,
    distillation_sample_identity,
)
from swiftvr.training.reference import sha256_file

STAGE_A_CACHE_KIND = "swiftvr_b2a_stage_a_teacher_velocity"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--teacher-delta-checkpoint", type=Path, required=True)
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
        teacher_delta = args.teacher_delta_checkpoint.expanduser().resolve()
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

        delta_meta = teacher_delta / "metadata.json"
        delta_weights = teacher_delta / "trainable.safetensors"
        if not delta_meta.is_file() or not delta_weights.is_file():
            raise FileNotFoundError(
                "Stage-A teacher delta must contain metadata.json and "
                f"trainable.safetensors: {teacher_delta}"
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
            raise RuntimeError("No UltraVideo views selected for caching")

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
            str(base_root),
            subfolder=args.transformer_subfolder,
            torch_dtype=runtime_dtype,
            low_cpu_mem_usage=True,
        )
        configure_train_scope(reae, teacher, "adapter")
        reae.to(device=device, dtype=runtime_dtype).eval()
        teacher.to(device=device, dtype=runtime_dtype)
        closure = SwiftVRVelocityDistillationForward(
            reae,
            teacher,
            attention_backend=args.attention_backend,
        )
        cast_trainable_parameters(closure, dtype=torch.float32)
        loaded = load_delta_checkpoint(teacher_delta, closure, strict=True)
        closure.eval()
        closure.reae.eval()

        if rank == 0:
            mode = "distributed" if distributed else "single-process"
            print(
                f"UltraVideo Stage-A cache: mode={mode} world_size={world_size} "
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

        with torch.inference_mode():
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
                    identity = distillation_sample_identity(batch_cpu, local_index)
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
                    save_file({"velocity": velocity}, str(output / relative_file))
                    local_samples.append({**identity, "file": relative_file})
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
                f"Rank {rank} cached {local_processed}, expected {local_expected}"
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
            metadata: dict[str, object] = {
                "format_version": TEACHER_CACHE_FORMAT_VERSION,
                "kind": STAGE_A_CACHE_KIND,
                "dataset_kind": "ultravideo_materialized_lr_v1",
                "teacher_role": "stage_a_200k_prompt_free_no_time",
                "base_checkpoint": str(base_root),
                "teacher_delta_checkpoint": str(teacher_delta),
                "teacher_delta_step": int(loaded.get("step", -1)),
                "teacher_delta_metadata_sha256": sha256_file(delta_meta),
                "teacher_delta_weights_sha256": sha256_file(delta_weights),
                "reae_file": str(base_root / args.reae_filename),
                "reae_sha256": sha256_file(base_root / args.reae_filename),
                "runtime_dtype": str(runtime_dtype).removeprefix("torch."),
                "dtype": args.cache_dtype,
                "attention_backend": args.attention_backend,
                "materialized_manifest": str(manifest),
                "materialized_manifest_sha256": sha256_file(manifest),
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
                ),
                flush=True,
            )

        if distributed:
            dist.barrier()
        return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
