#!/usr/bin/env python3
"""Build the activation-pruned Wan-1.3B-shaped B2-A student initialization.

The source teacher is the exact Stage-A model:

    folded prompt-free/no-time base + Stage-A 200k delta

A small deterministic calibration prefix is run through the frozen teacher to
rank (1) shared residual channels, (2) whole self-attention heads per block, and
(3) FFN neurons per block.  The selected tensors are structurally sliced into a
30-layer 1536-dim / 12-head / 8960-FFN compact student.  No decoder or RGB loss is
used here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Mapping

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
from swiftvr.training.b2a_width import (
    ActivationImportanceCollector,
    B2AWidthSpec,
    build_compact_transformer_from_teacher,
    transfer_structured_width,
    transformer_width_shape,
    validate_b2a_teacher_shape,
)
from swiftvr.training.distillation import DeterministicTripletViewDataset, SwiftVRVelocityDistillationForward
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
    p.add_argument("--calibration-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--allow-dtype-mismatch", action="store_true")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress-every", type=int, default=8)
    p.add_argument("--student-hidden-dim", type=int, default=1536)
    p.add_argument("--student-num-heads", type=int, default=12)
    p.add_argument("--student-head-dim", type=int, default=128)
    p.add_argument("--student-ffn-dim", type=int, default=8960)
    p.add_argument("--student-num-layers", type=int, default=30)
    p.add_argument("--student-adapter-dim", type=int, default=128)
    return p


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _slice_batch(batch: Mapping[str, object], count: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value[:count]
        elif isinstance(value, list):
            result[key] = value[:count]
        elif isinstance(value, tuple):
            result[key] = value[:count]
        else:
            result[key] = value
    return result


def _score_stats(tensor: torch.Tensor) -> dict[str, float]:
    value = tensor.detach().float().cpu()
    return {
        "min": float(value.min().item()),
        "mean": float(value.mean().item()),
        "max": float(value.max().item()),
        "std": float(value.std(unbiased=False).item()),
    }


def _spec(args: argparse.Namespace) -> B2AWidthSpec:
    return B2AWidthSpec(
        hidden_dim=args.student_hidden_dim,
        num_heads=args.student_num_heads,
        head_dim=args.student_head_dim,
        ffn_dim=args.student_ffn_dim,
        num_layers=args.student_num_layers,
        adapter_dim=args.student_adapter_dim,
    )


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("calibration_samples", "batch_size", "progress_every", "views_per_record"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    for name in ("horizontal_flip_probability", "vertical_flip_probability"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    spec = _spec(args)
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
    dtype = resolve_runtime_dtype(
        args.dtype,
        folded_config,
        device,
        allow_mismatch=args.allow_dtype_mismatch,
    )
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    delta_meta = teacher_delta / "metadata.json"
    delta_weights = teacher_delta / "trainable.safetensors"
    if not delta_meta.is_file() or not delta_weights.is_file():
        raise FileNotFoundError(
            "Teacher delta must contain metadata.json and trainable.safetensors: "
            f"{teacher_delta}"
        )

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

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
    views = DeterministicTripletViewDataset(
        base_dataset,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
    )
    calibration_limit = min(len(views), int(args.calibration_samples))
    if calibration_limit <= 0:
        raise RuntimeError("Calibration dataset is empty")
    loader = DataLoader(
        views,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.num_workers > 0),
    )

    reae = ReAE(str(base_root / args.reae_filename))
    teacher = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(base_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    source_shape = validate_b2a_teacher_shape(teacher, spec)
    configure_train_scope(reae, teacher, "adapter")
    reae.to(device=device, dtype=dtype).eval()
    teacher.to(device=device, dtype=dtype)
    teacher_closure = SwiftVRVelocityDistillationForward(
        reae,
        teacher,
        attention_backend=args.attention_backend,
    )
    cast_trainable_parameters(teacher_closure, dtype=torch.float32)
    loaded = load_delta_checkpoint(teacher_delta, teacher_closure, strict=True)
    teacher_closure.eval()
    teacher_closure.reae.eval()

    collector = ActivationImportanceCollector(teacher)
    processed = 0
    started = time.perf_counter()
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    try:
        with torch.inference_mode():
            for batch_cpu in loader:
                if processed >= calibration_limit:
                    break
                batch_size = None
                frame_indices = batch_cpu.get("frame_indices")
                if isinstance(frame_indices, torch.Tensor):
                    batch_size = int(frame_indices.shape[0])
                if batch_size is None:
                    raise TypeError("Calibration batch is missing collated frame_indices")
                remaining = calibration_limit - processed
                if batch_size > remaining:
                    batch_cpu = _slice_batch(batch_cpu, remaining)
                    batch_size = remaining
                batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                with torch.autocast(
                    "cuda",
                    dtype=dtype,
                    enabled=device.type == "cuda" and autocast_enabled,
                ):
                    teacher_closure(batch)
                processed += batch_size
                if processed == calibration_limit or processed % args.progress_every == 0:
                    print(
                        f"calibration {processed}/{calibration_limit} "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
    finally:
        collector.close()

    if processed != calibration_limit:
        raise RuntimeError(f"Calibration processed {processed}, expected {calibration_limit}")
    scores = collector.scores()
    selected = collector.select(spec)

    print("building compact student on CPU...", flush=True)
    student = build_compact_transformer_from_teacher(teacher, spec)
    transfer = transfer_structured_width(
        teacher,
        student,
        hidden_indices=selected["hidden"],
        head_indices_by_block=selected["heads"],
        ffn_indices_by_block=selected["ffn"],
        spec=spec,
    )
    target_shape = transformer_width_shape(student)

    importance_path = output / "activation_importance.safetensors"
    save_file(
        {
            "hidden_global": scores["hidden_global"].contiguous(),
            "hidden_by_block": scores["hidden_by_block"].contiguous(),
            "head_by_block": scores["head_by_block"].contiguous(),
            "ffn_by_block": scores["ffn_by_block"].contiguous(),
        },
        str(importance_path),
    )

    student.to(device="cpu", dtype=dtype)
    transformer_dir = output / args.transformer_subfolder
    student.save_pretrained(str(transformer_dir), safe_serialization=True)

    report = {
        "kind": "swiftvr_b2a_wan13_structured_init",
        "method": "activation_rms_structured_width_pruning",
        "base_checkpoint": str(base_root),
        "teacher_delta_checkpoint": str(teacher_delta),
        "teacher_delta_step": int(loaded.get("step", -1)),
        "teacher_delta_metadata_sha256": sha256_file(delta_meta),
        "teacher_delta_weights_sha256": sha256_file(delta_weights),
        "teacher_shape": source_shape,
        "student_shape": target_shape,
        "student_parameter_count": sum(parameter.numel() for parameter in student.parameters()),
        "saved_dtype": str(dtype).removeprefix("torch."),
        "calibration": {
            "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
            "split": args.split,
            "clip_length": args.clip_length,
            "crop_size": args.crop_size,
            "scale": args.scale,
            "views_per_record": args.views_per_record,
            "view_seed": args.view_seed,
            "horizontal_flip_probability": args.horizontal_flip_probability,
            "vertical_flip_probability": args.vertical_flip_probability,
            "samples": processed,
            "batch_size": args.batch_size,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "importance_stats": {
            "hidden_global": _score_stats(scores["hidden_global"]),
            "head_by_block": _score_stats(scores["head_by_block"]),
            "ffn_by_block": _score_stats(scores["ffn_by_block"]),
        },
        "selection": {
            "hidden_indices": transfer["hidden_indices"],
            "head_indices_by_block": transfer["head_indices_by_block"],
            "ffn_indices_by_block": transfer["ffn_indices_by_block"],
        },
        "artifacts": {
            "transformer": str(transformer_dir),
            "activation_importance": str(importance_path),
        },
    }
    _write_json(output / "b2a_init_report.json", report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "teacher_shape": source_shape,
                "student_shape": target_shape,
                "student_parameter_count": report["student_parameter_count"],
                "calibration_samples": processed,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
