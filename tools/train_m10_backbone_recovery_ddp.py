#!/usr/bin/env python3
"""M10 short gates: frozen ReAE/A1, trainable M8-A, direct Stage-A supervision.

Reuses canonical deterministic datasets, MoE forward/checkpointing, optimizer,
metrics, visual exporter and Transformer snapshot format. No old trainer is
modified. This gate entry point saves weight checkpoints (not optimizer resume).
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--student-init", type=Path, required=True,
                   help="M8-A30k or an M10 weight checkpoint root containing transformer/.")
    p.add_argument("--decoder-checkpoint", type=Path, required=True,
                   help="Frozen M9-A1 E99 tiny_decoder directory.")
    p.add_argument("--teacher-cache", type=Path, required=True,
                   help="Stage-A D3072 TRAIN velocity cache; NOT the D1536 TA cache.")
    p.add_argument("--val-teacher-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--lr-warmup-steps", type=int, default=50)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--velocity-nmse-weight", type=float, default=0.25)
    p.add_argument("--velocity-cosine-weight", type=float, default=0.25)
    p.add_argument("--rgb-weight", type=float, default=1.0)
    p.add_argument("--hf-weight", type=float, default=1.0)
    p.add_argument("--hf-temporal-weight", type=float, default=0.0)
    p.add_argument("--router-balance-weight", type=float, default=0.01)
    p.add_argument("--validate-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--visual-samples", type=int, default=13)
    p.add_argument("--visual-fps", type=float, default=30.0,
                   help="Playback rate only; not the source/training frame rate.")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--seed", type=int, default=0)
    return p


def _check_args(a) -> None:
    for key in ("batch_size", "gradient_accumulation_steps", "max_steps", "validate_every", "save_every", "log_every"):
        if getattr(a, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= a.lr_warmup_steps < a.max_steps:
        raise ValueError("lr-warmup-steps must be in [0,max-steps)")
    if a.learning_rate <= 0 or a.max_grad_norm <= 0 or not 0 < a.min_lr_ratio <= 1:
        raise ValueError("Invalid learning rate, clipping norm or minimum LR ratio")
    if a.visual_samples < 0 or a.visual_fps <= 0:
        raise ValueError("Invalid visualization settings")
    for key in ("base_checkpoint", "student_init", "decoder_checkpoint", "teacher_cache", "val_teacher_cache", "output_dir", "path_root"):
        setattr(a, key, getattr(a, key).expanduser().resolve())
    for path in (a.base_checkpoint / "reae.safetensors", a.student_init / "transformer/config.json",
                 a.decoder_checkpoint / "config.json", a.decoder_checkpoint / "model.safetensors",
                 a.teacher_cache / "metadata.json", a.val_teacher_cache / "metadata.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not list((a.student_init / "transformer").glob("*.safetensors")):
        raise FileNotFoundError(f"No local Transformer safetensors under {a.student_init}")
    for source in (a.base_checkpoint, a.student_init, a.decoder_checkpoint, a.teacher_cache, a.val_teacher_cache):
        if a.output_dir == source or source in a.output_dir.parents:
            raise ValueError(f"Output must not be inside immutable source {source}")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(f"Use a new M10 output directory: {a.output_dir}")


def _dataset(cache, *, path_root: Path):
    from tools import train_teacher_distillation_ddp as data_tools
    m = cache.metadata
    return data_tools.build_cached_dataset(
        [Path(value) for value in m["manifests"]], cache,
        split=m["split"], path_root=path_root,
        clip_length=int(m["clip_length"]), crop_size=int(m["crop_size"]), scale=int(m["scale"]),
        views_per_record=int(m["views_per_record"]), view_seed=int(m["view_seed"]),
        hflip=float(m["horizontal_flip_probability"]), vflip=float(m["vertical_flip_probability"]),
        verify_paths=False,
    )


def _clear_attention_caches() -> None:
    from swiftvr.models import transformer as ops
    ops._WindowIndexCache.clear()
    ops._WindowRuntimeMetaCache.clear()


@torch.no_grad()
def validate(closure, loader, cache, *, device, dtype, weights, high_pass, visual_samples):
    from tools.smoke_training_forward import move_video_batch
    from swiftvr.training.stage3 import VideoMetricAccumulator
    from swiftvr.training.m10_recovery import recovery_objective

    closure.eval()
    metrics = {name: VideoMetricAccumulator() for name in (
        "student_teacher", "student_gt", "teacher_gt", "backbone_original_teacher"
    )}
    sums, visuals, count = {}, [], 0
    _clear_attention_caches()
    try:
        for batch_cpu in loader:
            tv = cache.load_batch(batch_cpu, device=device, dtype=dtype)
            batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
                out = closure(batch, tv, include_original=True)
                terms = recovery_objective(out, tv, weights=weights, high_pass=high_pass)
            n = int(out["target"].shape[0])
            count += n
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + n * float(value.detach())
            s, t, gt = out["prediction"], out["teacher_prediction"], out["target"]
            metrics["student_teacher"].update(s, t, clamp=True)
            metrics["student_gt"].update(s, gt, clamp=True)
            metrics["teacher_gt"].update(t, gt, clamp=True)
            metrics["backbone_original_teacher"].update(out["student_original_prediction"], t, clamp=True)
            for i in range(n):
                if len(visuals) >= visual_samples:
                    break
                visuals.append({
                    "record_uid": str(batch_cpu["record_uid"][i]),
                    "lq_input": out["lq_input"][i].float().cpu(),
                    "target": gt[i].float().cpu(),
                    "teacher_prediction": t[i].float().cpu(),
                    "student_prediction": s[i].float().cpu(),
                })
    finally:
        _clear_attention_caches()
        closure.train()
    if not count:
        raise RuntimeError("Empty validation dataset")
    result = {key: value / count for key, value in sums.items()}
    result["samples"] = count
    for name, accumulator in metrics.items():
        result.update({f"{name}_{key}": value for key, value in accumulator.compute().items()})
    return result, visuals


def main() -> int:
    a = build_parser().parse_args()
    _check_args(a)
    # Heavy repository imports stay after --help/argument validation.
    from tools import train_b2a_compact_distill_ddp as base
    from tools import train_teacher_distillation_ddp as data_tools
    from tools.train_b2b_moe_ta_distill_ddp import _gradient_summary_allow_sparse_experts
    from tools.smoke_training_forward import move_video_batch
    from swiftvr.models import ReAE, WanTransformer3DModelPromptFreeNoTimeMoE
    from swiftvr.models.m9_factorized_decoder import M9A1FactorizedReAEDecoder, M9_A1_GMAC_1920X1088
    from swiftvr.training import TeacherVelocityCache, append_jsonl, build_fp32_adamw, cast_trainable_parameters, seed_everything, write_latest_checkpoint
    from swiftvr.training.b2b_moe import B2BMoESpec, expected_moe_shape, transformer_moe_shape
    from swiftvr.training.distillation_visuals import export_validation_visuals
    from swiftvr.training.m10_recovery import GaussianHighPass, M10BackboneRecoveryForward, RecoveryWeights, recovery_objective, validate_teacher_metadata
    from swiftvr.training.reference import sha256_file

    weights = RecoveryWeights(a.velocity_nmse_weight, a.velocity_cosine_weight, a.rgb_weight,
                              a.hf_weight, a.hf_temporal_weight, a.router_balance_weight)
    rank, local_rank, world, device = data_tools.init_distributed()
    try:
        dtype = getattr(torch, a.dtype)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This M10 BF16 gate requires a BF16-capable GPU")
        seed_everything(a.seed + rank)
        train_cache, val_cache = TeacherVelocityCache(a.teacher_cache), TeacherVelocityCache(a.val_teacher_cache)
        reae_hash = sha256_file(a.base_checkpoint / "reae.safetensors")
        validate_teacher_metadata(train_cache.metadata, val_cache.metadata, reae_sha256=reae_hash)
        if train_cache.metadata["split"] != "train" or val_cache.metadata["split"] != "val":
            raise ValueError("M10 requires train/val caches with the corresponding splits")
        train_set, val_set = _dataset(train_cache, path_root=a.path_root), _dataset(val_cache, path_root=a.path_root)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0) if rank == 0 else None

        reae = ReAE(str(a.base_checkpoint / "reae.safetensors")).to(device=device, dtype=dtype)
        decoder = M9A1FactorizedReAEDecoder.from_pretrained(a.decoder_checkpoint, device=device, dtype=dtype)
        transformer = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
            str(a.student_init), subfolder="transformer", torch_dtype=dtype,
            low_cpu_mem_usage=True, local_files_only=True,
        ).to(device=device, dtype=dtype)
        shape = transformer_moe_shape(transformer)
        if shape != expected_moe_shape(B2BMoESpec(num_layers=20)):
            raise ValueError(f"M10 expects the unchanged M8-A D1024/L20 MoE, got {shape}")
        closure = M10BackboneRecoveryForward(reae, transformer, decoder, attention_backend=a.attention_backend).train()
        cast_trainable_parameters(closure, dtype=torch.float32)
        trainable = [name for name, p in closure.named_parameters() if p.requires_grad]
        if not trainable or any(not name.startswith("transformer.") for name in trainable):
            raise RuntimeError("M10 trainable parameter scope is not Transformer-only")
        optimizer = build_fp32_adamw(closure, learning_rate=a.learning_rate, weight_decay=0.0, eps=1e-8)
        ddp = DistributedDataParallel(closure, device_ids=[local_rank], output_device=local_rank,
                                      broadcast_buffers=False, find_unused_parameters=True,
                                      gradient_as_bucket_view=True)
        high_pass = GaussianHighPass().to(device)
        config = {
            "trainer": "m10_transformer_hf_recovery_v1",
            "recipe": "spatial_temporal" if weights.hf_temporal_l1 else "spatial",
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            "loss_weights": asdict(weights), "student_shape": shape,
            "student_init_config_sha256": sha256_file(a.student_init / "transformer/config.json"),
            "student_init_weights_sha256": {p.name: sha256_file(p) for p in sorted((a.student_init / "transformer").glob("*.safetensors"))},
            "frozen_reae_sha256": reae_hash,
            "frozen_a1_weights_sha256": sha256_file(a.decoder_checkpoint / "model.safetensors"),
            "train_cache_metadata_sha256": sha256_file(a.teacher_cache / "metadata.json"),
            "val_cache_metadata_sha256": sha256_file(a.val_teacher_cache / "metadata.json"),
            "training_teacher": "Stage-A D3072 step200000 velocity + Original ReAE RGB",
            "teacher_delta_weights_sha256": train_cache.metadata["teacher_delta_weights_sha256"],
            "gt_role": "diagnostic_only", "train_scope": "transformer_only",
            "train_views": len(train_set), "val_views": len(val_set),
            "train_view_protocol": {k: train_cache.metadata[k] for k in (
                "manifests", "clip_length", "crop_size", "scale", "views_per_record", "view_seed",
                "horizontal_flip_probability", "vertical_flip_probability")},
            "global_batch_size": world * a.batch_size * a.gradient_accumulation_steps,
            "decoder_gmac_per_frame_1920x1088": M9_A1_GMAC_1920X1088,
            "deployment_architecture_changed": False,
            "high_pass": "5x5 Gaussian sigma1 RGB spatial-only reflect-padding, FP32",
            "selection": "rgb_l1 + hf_l1 + hf_temporal_l1; visual gate still required",
            "visual_fps_role": "playback_only_not_source_cadence",
            "checkpoint_type": "weights_only; --student-init warm start resets optimizer/LR",
        }
        if rank == 0:
            a.output_dir.mkdir(parents=True, exist_ok=True)
            base._write_json(a.output_dir / "run_config.json", config)
            print(json.dumps({k: config[k] for k in ("recipe", "training_teacher", "train_scope", "train_views", "val_views", "global_batch_size", "loss_weights")}, indent=2), flush=True)
        dist.barrier()
        best = None

        def save_snapshot(step: int, validation=None):
            path = a.output_dir / "checkpoints" / f"step_{step:08d}"
            if path.exists():
                raise FileExistsError(path)
            base._save_snapshot(transformer, path, runtime_dtype=dtype, transformer_subfolder="transformer",
                                metadata={"global_step": step, "m10_lineage": config, "validation": validation})
            write_latest_checkpoint(a.output_dir, path)
            return path

        def run_validation(step: int):
            nonlocal best
            dist.barrier()
            if rank == 0:
                result, visuals = validate(closure, val_loader, val_cache, device=device, dtype=dtype,
                                           weights=weights, high_pass=high_pass, visual_samples=a.visual_samples)
                append_jsonl(a.output_dir / "val_log.jsonl", {"global_step": step, **result})
                if visuals:
                    report = export_validation_visuals(visuals, output_root=a.output_dir, step=step,
                                                       frame_indices=(0, 6, 12), fps=a.visual_fps)
                    if report["video_errors"]:
                        print("Visual video warnings: " + json.dumps(report["video_errors"]), flush=True)
                path = save_snapshot(step, result) if step else a.student_init
                score = result["teacher_selection_score"]
                if best is None or score < best["teacher_selection_score"]:
                    best = {"global_step": step, "teacher_selection_score": score,
                            "checkpoint": str(path), "validation": result}
                    base._write_json(a.output_dir / "best.json", best)
                print(f"[val step={step}] deploy/StageA PSNR={result['student_teacher_psnr']:.4f} "
                      f"backbone+Orig/StageA PSNR={result['backbone_original_teacher_psnr']:.4f} "
                      f"HF={result['hf_l1']:.6g} HF-temporal={result['hf_temporal_l1']:.6g}", flush=True)
            dist.barrier()

        run_validation(0)
        # The canonical loader only needs these additional fixed attributes.
        a.num_workers, a.pin_memory = 0, True
        step, epoch = 0, 0
        while step < a.max_steps:
            loader = data_tools.make_train_loader(train_set, rank=rank, world_size=world, epoch=epoch, args=a)
            if len(loader) < a.gradient_accumulation_steps:
                raise RuntimeError("Training epoch is shorter than one effective batch")
            iterator = iter(loader)
            while step < a.max_steps:
                micro_batches = []
                for _ in range(a.gradient_accumulation_steps):
                    try:
                        micro_batches.append(next(iterator))
                    except StopIteration:
                        break
                if len(micro_batches) < a.gradient_accumulation_steps:
                    break
                lr = base._lr_for_step(a, step + 1)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                sums = {}
                for micro, batch_cpu in enumerate(micro_batches):
                    tv = train_cache.load_batch(batch_cpu, device=device, dtype=dtype)
                    batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                    sync = nullcontext() if micro + 1 == len(micro_batches) else ddp.no_sync()
                    with sync:
                        with torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
                            out = ddp(batch, tv)
                            if not out["prediction"].requires_grad:
                                raise RuntimeError("Frozen decoder severed the RGB gradient path")
                            terms = recovery_objective(out, tv, weights=weights, high_pass=high_pass)
                        bad = (~torch.isfinite(terms["loss"].detach())).to(torch.int32)
                        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                        if bool(bad):
                            raise FloatingPointError("Non-finite M10 loss on at least one rank")
                        (terms["loss"] / len(micro_batches)).backward()
                    for key, value in terms.items():
                        sums[key] = sums.get(key, 0.0) + float(value.detach()) / len(micro_batches)
                    del out, terms
                gradients = _gradient_summary_allow_sparse_experts(closure)
                if gradients["nonfinite_elements"] or gradients["forbidden_missing"] or not gradients["gradient_tensors"]:
                    raise RuntimeError("M10 gradient contract failed: " + json.dumps(gradients))
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), a.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step == 1 or step % a.log_every == 0:
                    keys = sorted(sums)
                    vector = torch.tensor([sums[k] for k in keys], device=device, dtype=torch.float64)
                    dist.all_reduce(vector)
                    if rank == 0:
                        record = {"global_step": step, "epoch": epoch, "learning_rate": lr,
                                  **dict(zip(keys, (vector / world).cpu().tolist())), "gradients": gradients}
                        append_jsonl(a.output_dir / "train_log.jsonl", record)
                        print(json.dumps(record), flush=True)
                if step % a.validate_every == 0 or step == a.max_steps:
                    run_validation(step)
                elif step % a.save_every == 0:
                    dist.barrier()
                    if rank == 0:
                        save_snapshot(step)
                    dist.barrier()
            epoch += 1
        if rank == 0:
            base._write_json(a.output_dir / "summary.json", {
                "status": "COMPLETED_VISUAL_REVIEW_REQUIRED", "global_step": step, "best": best,
                "frozen_decoder": str(a.decoder_checkpoint), "deployment_architecture_changed": False,
                "note": "Training completion is not an automatic quality PASS. Do not compare A/B scalar total losses.",
            })
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
