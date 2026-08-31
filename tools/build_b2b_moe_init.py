#!/usr/bin/env python3
"""Build the D1536-TA initialized D1024 1S12E2A sparse-MoE student.

The source is the locked Stage-B2A D1536 checkpoint.  A deterministic calibration
prefix measures activation-RMS importance.  Residual channels/attention heads are
selected structurally; dense FFN neurons are repartitioned into shared and routed
experts.  The output is a reloadable transformer-only checkpoint plus an audit
report.  No decoder or GT supervision is used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.smoke_training_forward import (
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training.b2a_width import (
    ActivationImportanceCollector,
    B2ACompactVelocityDistillationForward,
    B2AWidthSpec,
)
from swiftvr.training.b2b_moe import (
    B2BMoESpec,
    build_moe_transformer_from_teacher,
    parameter_accounting,
    transfer_d1536_to_moe,
    transformer_moe_shape,
    validate_d1536_ta,
)
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.reference import sha256_file


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
    p.add_argument("--calibration-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    p.add_argument("--allow-dtype-mismatch", action="store_true")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--router-seed", type=int, default=20260831)
    p.add_argument("--router-init-std", type=float, default=1e-3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress-every", type=int, default=8)
    return p


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _slice_batch(batch: dict[str, object], count: int) -> dict[str, object]:
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


def _score_stats(value: torch.Tensor) -> dict[str, float]:
    x = value.detach().float().cpu()
    return {
        "min": float(x.min().item()),
        "mean": float(x.mean().item()),
        "max": float(x.max().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "clip_length",
        "crop_size",
        "scale",
        "views_per_record",
        "calibration_samples",
        "batch_size",
        "progress_every",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.router_init_std < 0:
        raise ValueError("--router-init-std must be non-negative")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    spec = B2BMoESpec()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    base_root = args.base_checkpoint.expanduser().resolve()
    teacher_root = args.teacher_checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
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

    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    teacher_config = teacher_root / args.transformer_subfolder / "config.json"
    teacher_weights = (
        teacher_root
        / args.transformer_subfolder
        / "diffusion_pytorch_model.safetensors"
    )
    if not teacher_config.is_file() or not teacher_weights.is_file():
        raise FileNotFoundError(
            "Teacher checkpoint must contain transformer/config.json and "
            "transformer/diffusion_pytorch_model.safetensors"
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
    views = DeterministicTripletViewDataset(
        base_dataset,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
    )
    calibration_limit = min(len(views), int(args.calibration_samples))
    loader = DataLoader(
        views,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    reae = ReAE(str(base_root / args.reae_filename))
    teacher = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(teacher_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    source_shape = validate_d1536_ta(teacher)
    for parameter in reae.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    reae.to(device=device, dtype=dtype).eval()
    teacher.to(device=device, dtype=dtype).eval()
    closure = B2ACompactVelocityDistillationForward(
        reae,
        teacher,
        attention_backend=args.attention_backend,
        gradient_checkpointing=False,
    ).eval()
    closure.reae.eval()

    collector = ActivationImportanceCollector(teacher)
    processed = 0
    started = time.perf_counter()
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    try:
        with torch.no_grad():
            for batch_cpu in loader:
                if processed >= calibration_limit:
                    break
                frame_indices = batch_cpu.get("frame_indices")
                if not isinstance(frame_indices, torch.Tensor):
                    raise TypeError("Calibration batch is missing frame_indices")
                count = int(frame_indices.shape[0])
                remaining = calibration_limit - processed
                if count > remaining:
                    batch_cpu = _slice_batch(batch_cpu, remaining)
                    count = remaining
                batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                with torch.autocast(
                    "cuda",
                    dtype=dtype,
                    enabled=device.type == "cuda" and autocast_enabled,
                ):
                    closure(batch)
                processed += count
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

    selection_spec = B2AWidthSpec(
        hidden_dim=spec.hidden_dim,
        num_heads=spec.num_heads,
        head_dim=spec.head_dim,
        ffn_dim=spec.total_ffn_dim,
        num_layers=spec.num_layers,
        adapter_dim=spec.adapter_dim,
    )
    selected = collector.select(selection_spec)

    print("building D1024 sparse-MoE student on CPU...", flush=True)
    student = build_moe_transformer_from_teacher(teacher, spec)
    transfer = transfer_d1536_to_moe(
        teacher,
        student,
        hidden_indices=selected["hidden"],
        head_indices_by_block=selected["heads"],
        ffn_scores_by_block=scores["ffn_by_block"],
        spec=spec,
        router_seed=args.router_seed,
        router_init_std=args.router_init_std,
    )
    target_shape = transformer_moe_shape(student)
    params = parameter_accounting(student)

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
        "kind": "swiftvr_b2b_d1024_1s12e2a_moe_init",
        "method": "d1536_ta_activation_rms_dense_to_sparse_moe",
        "design": {
            "shared_expert": "top activation-RMS dense FFN neurons",
            "normal_experts": "next-ranked neurons distributed round-robin",
            "router": "small deterministic near-uniform random projection",
            "token_skipping": False,
            "block_skipping": False,
        },
        "base_checkpoint": str(base_root),
        "teacher_checkpoint": str(teacher_root),
        "teacher_config_sha256": sha256_file(teacher_config),
        "teacher_weights_sha256": sha256_file(teacher_weights),
        "teacher_shape": source_shape,
        "student_shape": target_shape,
        "parameter_accounting": params,
        "saved_dtype": str(dtype).removeprefix("torch."),
        "router_seed": args.router_seed,
        "router_init_std": args.router_init_std,
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
            "expert_allocations": transfer["expert_allocations"],
        },
        "artifacts": {
            "transformer": str(transformer_dir),
            "activation_importance": str(importance_path),
        },
    }
    _write_json(output / "moe_init_report.json", report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "teacher_shape": source_shape,
                "student_shape": target_shape,
                "parameter_accounting": params,
                "calibration_samples": processed,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
