#!/usr/bin/env python3
"""Cache the Stage-A 200k no-time/no-prompt teacher velocity for B2-A.

Unlike ``build_teacher_velocity_cache.py`` (which caches the original
conditional SwiftVR teacher), this tool materializes the exact Stage-A teacher as

    folded prompt-free/no-time base + Stage-A delta checkpoint

and caches only its endpoint velocity on deterministic triplet views.  No RGB
decoder participates in target construction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from smoke_training_forward import (
    configure_train_scope,
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training import cast_trainable_parameters, load_delta_checkpoint
from swiftvr.training.distillation import (
    TEACHER_CACHE_FORMAT_VERSION,
    TEACHER_CACHE_METADATA_FILENAME,
    DeterministicTripletViewDataset,
    SwiftVRVelocityDistillationForward,
    distillation_sample_identity,
)
from swiftvr.training.reference import sha256_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--teacher-delta-checkpoint", type=Path, required=True)
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


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("views_per_record", "batch_size", "progress_every"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    for name in ("horizontal_flip_probability", "vertical_flip_probability"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    base_root = args.base_checkpoint.expanduser().resolve()
    teacher_delta = args.teacher_delta_checkpoint.expanduser().resolve()
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

    delta_meta = teacher_delta / "metadata.json"
    delta_weights = teacher_delta / "trainable.safetensors"
    if not delta_meta.is_file() or not delta_weights.is_file():
        raise FileNotFoundError(
            "Stage-A teacher delta must contain metadata.json and trainable.safetensors: "
            f"{teacher_delta}"
        )

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Teacher cache is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    samples_dir = output / "samples"
    samples_dir.mkdir()

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
    loader = DataLoader(
        dataset,
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

    saved_samples: list[dict[str, object]] = []
    processed = 0
    started = time.perf_counter()
    autocast_enabled = runtime_dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode():
        for batch_cpu in loader:
            if processed >= sample_limit:
                break
            batch = move_video_batch(batch_cpu, device=device, dtype=runtime_dtype)
            with torch.autocast("cuda", dtype=runtime_dtype, enabled=device.type == "cuda" and autocast_enabled):
                output_batch = closure(batch)
            velocities = output_batch["velocity"]
            for local_index in range(int(velocities.shape[0])):
                if processed >= sample_limit:
                    break
                identity = distillation_sample_identity(batch_cpu, local_index)
                velocity = velocities[local_index].detach().to(
                    device="cpu", dtype=cache_dtype
                ).contiguous()
                relative_file = (
                    f"samples/{identity['distillation_index']:08d}_{identity['key']}.safetensors"
                )
                save_file({"velocity": velocity}, str(output / relative_file))
                saved_samples.append({**identity, "file": relative_file})
                processed += 1
                if processed == 1 or processed % args.progress_every == 0 or processed == sample_limit:
                    elapsed = time.perf_counter() - started
                    print(
                        f"cached {processed}/{sample_limit} "
                        f"({processed / max(elapsed, 1e-9):.3f} samples/s): "
                        f"{identity['record_uid']} view={identity['view_index']}",
                        flush=True,
                    )

    if processed != sample_limit:
        raise RuntimeError(f"Cached {processed} samples, expected {sample_limit}")

    manifests = [path.expanduser().resolve() for path in args.manifest]
    metadata: dict[str, object] = {
        "format_version": TEACHER_CACHE_FORMAT_VERSION,
        "kind": "swiftvr_b2a_stage_a_teacher_velocity",
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
        "sample_count": processed,
        "elapsed_seconds": time.perf_counter() - started,
        "samples": saved_samples,
    }
    _write_json(output / TEACHER_CACHE_METADATA_FILENAME, metadata)
    print(json.dumps({key: value for key, value in metadata.items() if key != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
