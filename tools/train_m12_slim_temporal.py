#!/usr/bin/env python3
"""M12-2 single-GPU Slim100 temporal/detail adaptation on immutable M11-B D1 pairs.

Frozen M11-B 30k is NOT instantiated: only its verified cached restored latents
are used. Frozen Original ReAE is a detail reference; paired HR GT provides
the reference for temporal luminance. All prior T1/T1B trainers are untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import ConcatDataset, DataLoader

from swiftvr.models import ReAE
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder, SLIM100_CHANNELS
from swiftvr.training.forward import decode_reae_clip
from swiftvr.training.m11b_decoder_pairs import M11BPairedLatentDataset
from swiftvr.training.m12_slim_objective import m12_slim_objective, scene_pair_valid
from swiftvr.training.stage3 import VideoMetricAccumulator
from tools.m11b_s1_d1_cache import sha256
from tools.train_m11b_s1_t1b_decoder import (
    BalancedBatchSampler, append_jsonl, atomic_json, lr_for_update,
    rng_state, restore_rng,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints_prompt_free_no_time"))
    p.add_argument("--m11b-checkpoint", type=Path, required=True)
    p.add_argument("--slim-init", type=Path, required=True)
    p.add_argument("--legacy-train", type=Path, required=True)
    p.add_argument("--ultra-train", type=Path, required=True)
    p.add_argument("--legacy-val", type=Path, required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--lpips-weights", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--schedule-steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--teacher-rgb-weight", type=float, default=1.0)
    p.add_argument("--teacher-hf-weight", type=float, default=0.5)
    p.add_argument("--teacher-lpips-weight", type=float, default=0.1)
    p.add_argument("--temporal-weight", type=float, default=0.5)
    p.add_argument("--lowpass-kernel", type=int, default=7)
    p.add_argument("--cut-hist-threshold", type=float, default=0.65)
    p.add_argument("--lpips-frame-batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--save-every", type=int, default=125)
    p.add_argument("--validate-every", type=int, default=125)
    p.add_argument("--log-every", type=int, default=10)
    return p


def validate_args(a):
    for key in ("batch_size", "gradient_accumulation_steps", "max_steps",
                "schedule_steps", "save_every", "validate_every", "log_every",
                "lpips_frame_batch"):
        if getattr(a, key) <= 0:
            raise ValueError(f"Invalid positive {key}")
    if a.num_workers < 0 or a.smoke_steps < 0 or not 0 <= a.warmup_steps < a.schedule_steps:
        raise ValueError("Invalid worker/schedule/smoke")
    if a.max_steps > a.schedule_steps or not 0 < a.min_lr_ratio <= 1:
        raise ValueError("Max steps beyond schedule or invalid min-lr-ratio")
    for key in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("weight_decay", "teacher_rgb_weight", "teacher_hf_weight",
                "teacher_lpips_weight", "temporal_weight"):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) < 0:
            raise ValueError(f"Invalid {key}")
    if a.teacher_rgb_weight + a.teacher_hf_weight <= 0 or a.temporal_weight <= 0:
        raise ValueError("Teacher detail and GT temporal terms must both be enabled")
    if a.lowpass_kernel < 3 or a.lowpass_kernel % 2 != 1:
        raise ValueError("lowpass kernel must be odd >=3")
    if not 0 < a.cut_hist_threshold <= 2:
        raise ValueError("cut histogram threshold outside (0,2]")
    for key in ("base_checkpoint", "m11b_checkpoint", "slim_init", "legacy_train",
                "ultra_train", "legacy_val", "path_root", "lpips_weights",
                "output_dir", "resume"):
        val = getattr(a, key)
        if val is not None:
            setattr(a, key, val.expanduser().resolve())
    for source in (a.base_checkpoint, a.m11b_checkpoint, a.slim_init,
                   a.legacy_train, a.ultra_train, a.legacy_val):
        if a.output_dir == source or a.output_dir in source.parents or source in a.output_dir.parents:
            raise ValueError("Output overlaps an immutable source tree")
    if a.output_dir.exists() and any(a.output_dir.iterdir()) and not a.resume:
        raise FileExistsError(f"Training output already exists: {a.output_dir}")
    if a.resume and a.resume.parent.parent != a.output_dir:
        raise ValueError("Resume must use an own-run checkpoint")


def identity_config(a):
    paths = [
        a.base_checkpoint / "reae.safetensors",
        a.slim_init / "model.safetensors",
        a.slim_init / "config.json",
        a.m11b_checkpoint / "transformer/diffusion_pytorch_model.safetensors",
        a.m11b_checkpoint / "transformer/config.json",
        a.legacy_train / "metadata.json", a.legacy_train / "plan.json",
        a.ultra_train / "metadata.json", a.ultra_train / "plan.json",
        a.legacy_val / "metadata.json", a.legacy_val / "plan.json",
        a.lpips_weights,
        Path(__file__).resolve(),
        ROOT / "swiftvr/training/m12_slim_objective.py",
        ROOT / "swiftvr/training/m11b_decoder_pairs.py",
        ROOT / "swiftvr/training/forward.py",
        ROOT / "swiftvr/training/m8_task.py",
        ROOT / "swiftvr/models/reae_slim_decoder.py",
        ROOT / "swiftvr/models/reae.py",
        ROOT / "tools/m11b_s1_d1_cache.py",
        ROOT / "tools/train_m11b_s1_t1b_decoder.py",
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items() if k not in ("max_steps", "resume", "smoke_steps")
    }
    payload.update({
        "kind": "M12-Slim100-temporal-GT-v1",
        "world_size": 1,
        "global_batch": a.batch_size * a.gradient_accumulation_steps,
        "source_sha256": {str(p): sha256(p) for p in paths},
        "transformer_trainable": False,
        "encoder_trainable": False,
        "decoder_trainable": True,
    })
    return payload


def loss_kwargs(a):
    return dict(
        teacher_rgb_weight=a.teacher_rgb_weight,
        teacher_hf_weight=a.teacher_hf_weight,
        teacher_lpips_weight=a.teacher_lpips_weight,
        temporal_weight=a.temporal_weight,
        lowpass_kernel=a.lowpass_kernel,
    )


def paired_loss(decoder, reae, z, gt, lpips, a):
    # The original decoder has the SAME M11-B latent as student; it is frozen.
    with torch.no_grad(), torch.autocast("cuda", dtype=z.dtype, enabled=z.dtype == torch.bfloat16):
        original = decode_reae_clip(reae, z, output_frames=13, clamp=False)
    pair_valid = scene_pair_valid(gt, cut_hist_threshold=a.cut_hist_threshold)
    with torch.autocast("cuda", dtype=z.dtype, enabled=z.dtype == torch.bfloat16):
        prediction = decoder(z, output_frames=13, clamp=False)
    terms = m12_slim_objective(
        prediction, gt, original, teacher_lpips=lpips,
        pair_valid=pair_valid, **loss_kwargs(a),
    )
    return terms, prediction, original


@torch.no_grad()
def evaluate(student, reae, val, lpips, *, device, dtype, a, step):
    student.eval()
    totals = {}
    quality = VideoMetricAccumulator()
    original_quality = VideoMetricAccumulator()
    for batch in DataLoader(val, batch_size=1, shuffle=False, num_workers=0):
        z = batch["z_sr"].to(device=device, dtype=dtype)
        gt = batch["hr"].to(device=device, dtype=torch.float32)
        terms, pred, original = paired_loss(student, reae, z, gt, lpips, a)
        quality.update(pred.float(), gt.float(), clamp=True)
        original_quality.update(original.float(), gt.float(), clamp=True)
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    student.train()
    result = {
        "step": step, "samples": len(val),
        **{name: amount / len(val) for name, amount in totals.items()},
        "slim_gt_metrics": quality.compute(),
        "original_gt_metrics": original_quality.compute(),
        "full_origin_custom_video_pass": False,
    }
    atomic_json(a.output_dir / "validation" / f"step_{step:08d}.json", result)
    append_jsonl(a.output_dir / "validation_log.jsonl", result)
    print("[M12-Slim val] " + json.dumps(result, sort_keys=True), flush=True)


def main():
    a = parser().parse_args()
    validate_args(a)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("M12-2 pilot is single GPU; do not launch via torchrun")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    legacy = M11BPairedLatentDataset(a.legacy_train, model_root=a.m11b_checkpoint,
                                    reae_root=a.base_checkpoint, path_root=a.path_root)
    ultra = M11BPairedLatentDataset(a.ultra_train, model_root=a.m11b_checkpoint,
                                   reae_root=a.base_checkpoint, path_root=a.path_root)
    val = M11BPairedLatentDataset(a.legacy_val, model_root=a.m11b_checkpoint,
                                 reae_root=a.base_checkpoint, path_root=a.path_root)
    if len(val) != 13 or (legacy.meta["domain"], ultra.meta["domain"], val.meta["split"]) != ("legacy", "ultra", "val"):
        raise ValueError("Expected complete Legacy/Ultra and val13 D1 caches")
    lineage = identity_config(a)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config_file = a.output_dir / "run_config.json"
    if a.resume:
        if json.loads(config_file.read_text()) != lineage:
            raise ValueError("Resume code/data/objective/schedule fingerprint mismatch")
    else:
        atomic_json(config_file, lineage)

    student = SlimReAEDecoder.from_pretrained(a.slim_init, device="cpu", dtype=torch.float32)
    if tuple(student.channels) != SLIM100_CHANNELS:
        raise ValueError(f"Not a Slim100 checkpoint: {student.channels}")
    student.to(device).train()
    teacher = ReAE(str(a.base_checkpoint / "reae.safetensors"))
    teacher.encoder = torch.nn.Identity()
    teacher.requires_grad_(False)
    teacher.to(device=device, dtype=dtype).eval()

    lpips = None
    if a.teacher_lpips_weight > 0:
        from swiftvr.training.m8_task import LocalLPIPS
        lpips = LocalLPIPS(a.lpips_weights, frame_batch=a.lpips_frame_batch).to(device).eval()
        lpips.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=a.learning_rate,
                                  weight_decay=a.weight_decay, eps=1e-8)
    step, micro = 0, 0
    if a.resume:
        state = torch.load(a.resume / "training_state.pt", map_location="cpu", weights_only=False)
        if state["lineage"] != lineage:
            raise ValueError("Checkpoint source lineage mismatch")
        student.load_state_dict(load_file(str(a.resume / "tiny_decoder/model.safetensors"), device="cpu"), strict=True)
        optimizer.load_state_dict(state["optimizer"])
        step, micro = int(state["step"]), int(state["micro_cursor"])
        if micro != step * a.gradient_accumulation_steps:
            raise ValueError("Inconsistent resume microstep cursor")
        restore_rng(state["rng"], device)

    stop = step + a.smoke_steps if a.smoke_steps else a.max_steps
    if stop <= step or stop > a.schedule_steps:
        raise ValueError("Stop outside existing schedule")
    all_data = ConcatDataset((legacy, ultra))
    sampler = BalancedBatchSampler(
        len(legacy), len(ultra), start=micro,
        stop=stop * a.gradient_accumulation_steps, batch_size=a.batch_size,
        rank=0, world=1, seed=a.seed,
    )
    worker_args = {"num_workers": a.num_workers}
    if a.num_workers:
        worker_args.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = iter(DataLoader(all_data, batch_sampler=sampler, pin_memory=True, **worker_args))
    params = list(student.parameters())
    started = time.monotonic()

    def checkpoint():
        folder = a.output_dir / "checkpoints" / f"step_{step:08d}"
        if folder.exists():
            raise FileExistsError(f"Refuse checkpoint overwrite: {folder}")
        folder.mkdir(parents=True)
        if any(t.is_floating_point() and t.dtype != torch.float32
               for t in student.state_dict().values()):
            raise RuntimeError("Trainable Slim100 master weights must remain FP32")
        student.save_pretrained(folder / "tiny_decoder")
        state = {
            "step": step, "micro_cursor": micro, "optimizer": optimizer.state_dict(),
            "rng": rng_state(device), "lineage": lineage,
        }
        tmp = folder / "training_state.tmp"
        torch.save(state, tmp)
        os.replace(tmp, folder / "training_state.pt")
        atomic_json(folder / "metadata.json", {
            "step": step, "phase": "M12-2", "decoder": "Slim100",
            "m11b_checkpoint": str(a.m11b_checkpoint),
            "teacher": str(a.base_checkpoint / "reae.safetensors"),
            "quality_pass": False,
        })
        atomic_json(a.output_dir / "latest.json", {"step": step, "path": str(folder)})

    if not a.resume and not a.smoke_steps:
        evaluate(student, teacher, val, lpips, device=device, dtype=dtype, a=a, step=0)
        checkpoint()

    optimizer.zero_grad(set_to_none=True)
    while step < stop:
        lr = lr_for_update(step + 1, a)
        for group in optimizer.param_groups:
            group["lr"] = lr
        values = {}
        for _ in range(a.gradient_accumulation_steps):
            batch = next(loader)
            z = batch["z_sr"].to(device=device, dtype=dtype, non_blocking=True)
            gt = batch["hr"].to(device=device, dtype=torch.float32, non_blocking=True)
            if z.shape[1:] != (4, 48, 24, 24) or gt.shape[1:] != (13, 3, 384, 384):
                raise ValueError("M12 requires canonical D1 T13 384x384 paired view")
            terms, _, _ = paired_loss(student, teacher, z, gt, lpips, a)
            (terms["loss"] / a.gradient_accumulation_steps).backward()
            for name, value in terms.items():
                values[name] = values.get(name, 0.0) + float(value.detach()) / a.gradient_accumulation_steps
            micro += 1
        if any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError("Original ReAE unexpectedly received gradients")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm,
                                                   error_if_nonfinite=True)
        if float(grad_norm.detach()) <= 0:
            raise RuntimeError("Missing/nonzero Slim100 gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % a.log_every == 0 or step == stop:
            record = {
                "step": step, "lr": lr, "elapsed_s": time.monotonic() - started,
                "grad_norm": float(grad_norm.detach()), **values,
            }
            append_jsonl(a.output_dir / "train_log.jsonl", record)
            print("[M12-Slim train] " + json.dumps(record, sort_keys=True), flush=True)
        val_due = not a.smoke_steps and (step % a.validate_every == 0 or step == stop)
        if val_due:
            evaluate(student, teacher, val, lpips, device=device, dtype=dtype, a=a, step=step)
        if step % a.save_every == 0 or step == stop or val_due:
            checkpoint()

    atomic_json(a.output_dir / "summary.json", {
        "status": "smoke_pass" if a.smoke_steps else "budget_completed",
        "step": step, "legacy_samples": len(legacy), "ultra_samples": len(ultra),
        "val_samples": len(val), "m11b_updated": False,
        "encoder_updated": False, "decoder_updated": True,
        "full_origin_custom_video_pass": False,
    })


if __name__ == "__main__":
    main()
