#!/usr/bin/env python3
"""One-sample B2B-1A forward/backward gate for tiny DiT + extreme decoder.

No optimizer step is taken.  The script loads a D768/F4080 B2B DiT init, the
best B2B-0C extreme decoder, and one deterministic Stage-A teacher-cache sample.
It verifies the complete joint path, computes the minimal B2B loss, backpropagates
once, and reports finite non-zero gradients separately for DiT and decoder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_teacher_distillation_ddp as stage_a
from tools.diagnose_b2b_extreme_train_val_gap import _resolve_student_root
from tools.smoke_training_forward import move_video_batch
from swiftvr.models import ReAE
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training import (
    DistillationMetricAccumulator,
    TeacherVelocityCache,
    VideoMetricAccumulator,
    cast_trainable_parameters,
    decode_teacher_prediction,
)
from swiftvr.training.b2a_width import transformer_width_shape
from swiftvr.training.b2b_joint import (
    B2B_EXTREME_DECODER_CHANNELS,
    B2B_TINY_SPEC,
    B2BJointForward,
    b2b_compute_budget,
    b2b_joint_objective,
)


DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--student-init", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--teacher-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--split", default="train")
    p.add_argument("--clip-length", type=int, default=13)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--views-per-record", type=int, default=8)
    p.add_argument("--view-seed", type=int, default=20260805)
    p.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    p.add_argument("--vertical-flip-probability", type=float, default=0.0)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--representation-mse-weight", type=float, default=0.05)
    p.add_argument("--representation-cosine-weight", type=float, default=0.05)
    p.add_argument("--teacher-rgb-l1-weight", type=float, default=1.0)
    p.add_argument("--gt-rgb-l1-weight", type=float, default=0.5)
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--output-json", type=Path, default=Path("outputs/b2b/b2b_1a_joint_backward.json"))
    return p


def _gradient_report(module: torch.nn.Module) -> dict[str, float | int | bool]:
    sq = 0.0
    max_abs = 0.0
    tensors = 0
    elements = 0
    nonfinite = 0
    nonzero = 0
    for parameter in module.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        value = grad.detach().float()
        tensors += 1
        elements += value.numel()
        bad = ~torch.isfinite(value)
        nonfinite += int(bad.sum().item())
        finite = value.masked_fill(bad, 0.0)
        sq += float(finite.square().sum().item())
        if finite.numel():
            max_abs = max(max_abs, float(finite.abs().max().item()))
        nonzero += int((finite != 0).sum().item())
    return {
        "gradient_tensors": tensors,
        "gradient_elements": elements,
        "nonfinite_elements": nonfinite,
        "nonzero_elements": nonzero,
        "l2": math.sqrt(sq),
        "max_abs": max_abs,
        "finite": nonfinite == 0,
        "has_nonzero": nonzero > 0,
    }


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_index < 0:
        raise ValueError("sample-index must be non-negative")
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")

    base_root = args.base_checkpoint.expanduser().resolve()
    student_root = args.student_init.expanduser().resolve()
    decoder_root, decoder_resolution = _resolve_student_root(args.decoder)
    cache = TeacherVelocityCache(args.teacher_cache)
    if cache.metadata.get("kind") != "swiftvr_b2a_stage_a_teacher_velocity":
        raise ValueError("B2B requires the Stage-A 200k B2-A teacher-velocity cache")
    dataset = stage_a.build_cached_dataset(
        args.manifest,
        cache,
        split=args.split,
        path_root=args.path_root.expanduser().resolve(),
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
        hflip=args.horizontal_flip_probability,
        vflip=args.vertical_flip_probability,
        verify_paths=args.verify_paths,
    )
    if args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} >= dataset length {len(dataset)}")
    loader = DataLoader(Subset(dataset, [args.sample_index]), batch_size=1, shuffle=False, num_workers=0)
    batch_cpu = next(iter(loader))

    reae = ReAE(str(base_root / args.reae_filename)).to(device=device, dtype=dtype).eval()
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(student_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype)
    expected_shape = {
        "hidden_dim": B2B_TINY_SPEC.hidden_dim,
        "num_heads": B2B_TINY_SPEC.num_heads,
        "head_dim": B2B_TINY_SPEC.head_dim,
        "ffn_dim": B2B_TINY_SPEC.ffn_dim,
        "num_layers": B2B_TINY_SPEC.num_layers,
        "adapter_dim": B2B_TINY_SPEC.adapter_dim,
    }
    actual_shape = transformer_width_shape(transformer)
    if actual_shape != expected_shape:
        raise ValueError(f"B2B DiT shape mismatch: {actual_shape} != {expected_shape}")

    decoder = SlimReAEDecoder.from_pretrained(decoder_root, device=device, dtype=dtype)
    if tuple(decoder.channels) != B2B_EXTREME_DECODER_CHANNELS:
        raise ValueError(
            f"Extreme decoder channels {decoder.channels} != {B2B_EXTREME_DECODER_CHANNELS}"
        )

    closure = B2BJointForward(
        reae,
        transformer,
        decoder,
        attention_backend=args.attention_backend,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    ).to(device=device)
    closure.train()
    cast_report = cast_trainable_parameters(closure, dtype=torch.float32)

    teacher_velocity = cache.load_batch(batch_cpu, device=device, dtype=dtype)
    batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
    autocast_enabled = device.type == "cuda" and dtype == torch.bfloat16
    with torch.autocast(
        device_type=device.type,
        dtype=dtype if autocast_enabled else torch.float32,
        enabled=autocast_enabled,
    ):
        output = closure(batch)
        with torch.no_grad():
            teacher_prediction = decode_teacher_prediction(
                reae=closure.reae,
                z_lq=output["z_lq"],
                teacher_velocity=teacher_velocity,
                output_frames=int(output["target"].shape[1]),
            )

    objective = b2b_joint_objective(
        output["velocity"],
        teacher_velocity,
        output["prediction"],
        teacher_prediction,
        output["target"],
        representation_mse_weight=args.representation_mse_weight,
        representation_cosine_weight=args.representation_cosine_weight,
        teacher_rgb_l1_weight=args.teacher_rgb_l1_weight,
        gt_rgb_l1_weight=args.gt_rgb_l1_weight,
    )
    if not torch.isfinite(objective["loss"]):
        raise FloatingPointError(f"Non-finite B2B joint loss: {objective['loss']}")
    objective["loss"].backward()

    dit_grad = _gradient_report(closure.transformer)
    decoder_grad = _gradient_report(closure.decoder)
    if not dit_grad["finite"] or not decoder_grad["finite"]:
        raise FloatingPointError(f"Non-finite gradients: dit={dit_grad}, decoder={decoder_grad}")
    if not dit_grad["has_nonzero"] or not decoder_grad["has_nonzero"]:
        raise RuntimeError(f"Missing joint gradient branch: dit={dit_grad}, decoder={decoder_grad}")

    velocity_metrics = DistillationMetricAccumulator()
    velocity_metrics.update(output["velocity"].detach(), teacher_velocity)
    student_teacher = VideoMetricAccumulator()
    student_teacher.update(output["prediction"].detach(), teacher_prediction, clamp=True)
    student_gt = VideoMetricAccumulator()
    student_gt.update(output["prediction"].detach(), output["target"], clamp=True)
    teacher_gt = VideoMetricAccumulator()
    teacher_gt.update(teacher_prediction, output["target"], clamp=True)

    report = {
        "status": "PASS",
        "sample_index": args.sample_index,
        "dtype": args.dtype,
        "student_shape": actual_shape,
        "decoder_channels": list(decoder.channels),
        "decoder_resolution": decoder_resolution,
        "cast_trainable_parameters": cast_report,
        "compute": b2b_compute_budget(),
        "loss_weights": {
            "representation_mse": args.representation_mse_weight,
            "representation_cosine": args.representation_cosine_weight,
            "teacher_rgb_l1": args.teacher_rgb_l1_weight,
            "gt_rgb_l1": args.gt_rgb_l1_weight,
        },
        "loss": {key: float(value.detach().item()) for key, value in objective.items()},
        "velocity": velocity_metrics.compute(),
        "student_teacher": student_teacher.compute(),
        "student_gt": student_gt.compute(),
        "teacher_gt": teacher_gt.compute(),
        "dit_grad": dit_grad,
        "decoder_grad": decoder_grad,
    }
    _write_json(args.output_json, report)

    print("================ B2B-1A joint backward gate ================")
    print(f"Tiny DiT shape                : {actual_shape}")
    print(f"Extreme decoder               : {list(decoder.channels)}")
    print(f"DiT + decoder GMAC/frame      : {report['compute']['dit_plus_decoder_gmac_per_frame']:.6f}")
    print(f"Total loss                    : {report['loss']['loss']:.6f}")
    print(f"Velocity rel-L2               : {report['velocity']['velocity_relative_l2']:.6f}")
    print(f"Velocity cosine               : {report['velocity']['velocity_cosine']:.6f}")
    print(f"Student -> Teacher PSNR       : {report['student_teacher']['psnr']:.4f} dB")
    print(f"Student -> GT PSNR            : {report['student_gt']['psnr']:.4f} dB")
    print(f"Teacher -> GT PSNR            : {report['teacher_gt']['psnr']:.4f} dB")
    print(f"DiT grad L2 / finite          : {dit_grad['l2']:.6f} / {dit_grad['finite']}")
    print(f"Decoder grad L2 / finite      : {decoder_grad['l2']:.6f} / {decoder_grad['finite']}")
    print("Status                        : PASS")
    print(f"Saved                         : {args.output_json.expanduser().resolve()}")
    print("=============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
