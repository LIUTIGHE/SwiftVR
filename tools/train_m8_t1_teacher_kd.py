#!/usr/bin/env python3
"""P1: GT RGB post-training through frozen deployment A1 into the unchanged M8.

Control: GT L1 + GT frame-difference MSE. Main: additionally GT LPIPS.
Reuses canonical deterministic datasets, balanced DDP sampler, MoE forward,
A1 forward and metrics. No teacher velocity/RGB targets, no GAN, no new topology.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--student-init", type=Path, required=True)
    p.add_argument("--decoder-checkpoint", type=Path, required=True)
    p.add_argument("--legacy-view-cache", type=Path, required=True, help="FULL cache metadata only; velocity tensors are not read")
    p.add_argument("--val-view-cache", type=Path, required=True)
    p.add_argument("--ultra-pairs", type=Path, action="append", required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path, help="This trainer's checkpoint, including training_state.pt")
    p.add_argument("--lpips-weights", type=Path, help="Complete local Alex-LPIPS state, not only the linear calibration weights")
    p.add_argument("--lpips-weight", type=float, default=0.1)
    p.add_argument("--pixel-weight", type=float, default=1.0)
    p.add_argument("--temporal-weight", type=float, default=1.0)
    p.add_argument("--router-weight", type=float, default=0.01)
    p.add_argument("--lpips-frame-batch", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--max-steps", type=int, default=5000, help="Stop update, independent of scheduler horizon")
    p.add_argument("--schedule-steps", type=int, default=10000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--no-decoder-checkpointing", action="store_true")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--custom-hook", type=Path, default=ROOT / "tools/visual_check_m8_task.sh")
    p.add_argument("--custom-ref-root", type=Path, help="Prepared Original/TinyCNN reference bank")
    p.add_argument("--smoke-steps", type=int, default=0, help="Partial HR allowed; GT-only backward; no custom validation. Not a quality result.")
    return p


def check_args(a):
    for key in ("batch_size", "gradient_accumulation_steps", "prefetch_factor", "lpips_frame_batch",
                "max_steps", "schedule_steps", "validate_every", "save_every", "log_every"):
        if getattr(a, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if a.num_workers < 0 or a.smoke_steps < 0 or not 0 <= a.warmup_steps < a.schedule_steps:
        raise ValueError("Invalid worker/smoke/warmup settings")
    if a.max_steps > a.schedule_steps or not math.isfinite(a.learning_rate) or a.learning_rate <= 0:
        raise ValueError("Invalid stop/schedule/learning rate")
    for key in ("pixel_weight", "temporal_weight", "lpips_weight", "router_weight"):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) < 0:
            raise ValueError("Loss weights must be finite/nonnegative")
    if a.pixel_weight + a.temporal_weight + a.lpips_weight <= 0:
        raise ValueError("Need an actual GT objective")
    if (a.lpips_weight or not a.smoke_steps) and a.lpips_weights is None:
        raise ValueError("Both formal runs require local LPIPS weights for comparable validation")
    if not a.smoke_steps and a.custom_ref_root is None:
        raise ValueError("Formal validation requires the two prepared custom references")
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.expanduser().resolve())
    a.ultra_pairs = [p.expanduser().resolve() for p in a.ultra_pairs]
    required = [a.base_checkpoint / "reae.safetensors", a.student_init / "transformer/config.json",
                a.decoder_checkpoint / "config.json", a.decoder_checkpoint / "model.safetensors",
                a.legacy_view_cache / "metadata.json", a.val_view_cache / "metadata.json", *a.ultra_pairs]
    if a.lpips_weights:
        required.append(a.lpips_weights)
    if a.resume:
        required.extend([a.resume / "training_state.pt", a.resume / "transformer/diffusion_pytorch_model.safetensors"])
    if not a.smoke_steps:
        required.extend([a.custom_hook, a.custom_ref_root / "reference.json"])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    immutable = [a.base_checkpoint, a.student_init, a.decoder_checkpoint, a.legacy_view_cache, a.val_view_cache]
    for path in immutable:
        if a.output_dir == path or path in a.output_dir.parents or a.output_dir in path.parents:
            raise ValueError("Output must be separate from all immutable source trees")
    if a.output_dir.exists() and any(a.output_dir.iterdir()) and a.resume is None:
        raise FileExistsError(f"Use a new run directory: {a.output_dir}")
    if a.resume and a.resume.parent.parent != a.output_dir:
        raise ValueError("Resume only inside the same recorded run; warm starts use --student-init")


def lr_scale(step, warmup, horizon):
    if warmup and step <= warmup:
        return step / warmup
    t = min(max((step-warmup) / max(1, horizon-warmup), 0.0), 1.0)
    return 0.1 + 0.9 * (1 + math.cos(math.pi*t)) / 2


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\n")


def clear_caches():
    from swiftvr.models import transformer as ops
    ops._WindowIndexCache.clear()
    ops._WindowRuntimeMetaCache.clear()


def main():
    a = build_parser().parse_args()
    check_args(a)
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, ConcatDataset, Sampler
    from safetensors.torch import save_file, load_file
    from swiftvr.data.task_pairs import build_legacy_pairs, build_ultra_pairs, sha256_file
    from swiftvr.models import ReAE, WanTransformer3DModelPromptFreeNoTimeMoE
    from swiftvr.models.m9_factorized_decoder import M9A1FactorizedReAEDecoder
    from swiftvr.training.b2b_moe import B2BMoESpec, expected_moe_shape, transformer_moe_shape
    from swiftvr.training.b2b_moe_training import B2BMoEVelocityDistillationForward, set_router_stats_collection
    from swiftvr.training.mixed_distillation import BalancedTwoDomainDistributedSampler
    from swiftvr.training.m8_task import M8TaskForward, LocalLPIPS, task_objective, phase_errors
    from swiftvr.training.stage3 import VideoMetricAccumulator

    rank, local_rank = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if not torch.cuda.is_available():
        raise RuntimeError("Real-model P1 trainer requires CUDA; unit tests are CPU-only")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = getattr(torch, a.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested BF16 is unsupported")
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=4))

    def barrier():
        if world > 1:
            dist.barrier()

    def rng_state():
        return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
                "python": random.getstate(), "numpy": np.random.get_state()}

    def restore_rng(value):
        torch.set_rng_state(value["torch"].cpu())
        torch.cuda.set_rng_state(value["cuda"].cpu(), device)
        random.setstate(value["python"])
        np.random.set_state(value["numpy"])

    try:
        random.seed(a.seed + rank)
        np.random.seed(a.seed + rank)
        torch.manual_seed(a.seed + rank)
        legacy, legacy_meta = build_legacy_pairs(a.legacy_view_cache, path_root=a.path_root, split="train")
        valset, val_meta = build_legacy_pairs(a.val_view_cache, path_root=a.path_root, split="val")
        ultra, ultra_meta = build_ultra_pairs(a.ultra_pairs, allow_partial=bool(a.smoke_steps))
        if len(valset) != 13:
            raise ValueError("P1 historical regression requires the full fixed val13")
        for meta in (legacy_meta, val_meta):
            if (meta["clip_length"], meta["crop_size"], meta["scale"]) != (13, 128, 3):
                raise ValueError("P1 first experiment keeps T13/LR128/3x geometry")
        dataset = ConcatDataset([legacy, ultra])
        sampler = BalancedTwoDomainDistributedSampler(dataset, num_replicas=world, rank=rank, seed=a.seed)
        # Keep a balanced prefix before batching exactly as the existing sampler;
        # 50:50 describes its global epoch, not every update.
        usable = (len(sampler) // a.batch_size) * a.batch_size
        if usable == 0:
            raise ValueError("Insufficient samples for one local batch")
        class CursorSampler(Sampler):
            def __init__(self): self.offset = 0
            def __iter__(self): return iter(list(sampler)[self.offset:usable])
            def __len__(self): return usable - self.offset
        cursor_sampler = CursorSampler()
        generator = torch.Generator().manual_seed(a.seed + 90000 + rank)
        workers = {"num_workers": a.num_workers}
        if a.num_workers:
            workers.update(prefetch_factor=a.prefetch_factor, persistent_workers=True)
        loader = DataLoader(dataset, batch_size=a.batch_size, sampler=cursor_sampler,
                            drop_last=True, pin_memory=True, generator=generator, **workers)
        val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0) if rank == 0 else None
        identity_files = [a.base_checkpoint / "reae.safetensors", a.decoder_checkpoint / "config.json",
                          a.decoder_checkpoint / "model.safetensors", a.student_init / "transformer/config.json",
                          *sorted((a.student_init / "transformer").glob("*.safetensors")),
                          a.legacy_view_cache / "metadata.json", a.val_view_cache / "metadata.json", *a.ultra_pairs]
        identity_files += [p.parent / "hr_metadata.json" for p in a.ultra_pairs]
        identity_files.extend([Path(__file__).resolve(), ROOT / "swiftvr/training/m8_task.py",
                               ROOT / "swiftvr/data/task_pairs.py", ROOT / "swiftvr/training/b2b_moe_training.py",
                               ROOT / "swiftvr/models/m9_factorized_decoder.py"])
        if a.lpips_weights: identity_files.append(a.lpips_weights)
        if a.custom_ref_root: identity_files.extend([a.custom_ref_root / "reference.json", a.custom_hook])
        config = {k: ([str(p) for p in v] if k == "ultra_pairs" else str(v) if isinstance(v, Path) else v)
                  for k,v in vars(a).items() if k not in {"resume", "max_steps"}}
        # Hash only once per run/rank0, not once per update or per GPU.
        hashes = {str(p): sha256_file(p) for p in identity_files} if rank == 0 else None
        if world > 1:
            obj = [hashes]; dist.broadcast_object_list(obj, src=0); hashes = obj[0]
        config.update(kind="m8_gt_task_posttrain_v1", hashes=hashes, world_size=world,
                      global_batch=world*a.batch_size*a.gradient_accumulation_steps,
                      legacy_views=len(legacy), ultra_views=len(ultra),
                      gt_role="primary_RGB_reconstruction_perceptual_temporal",
                      teacher_role="initialization_only; cache metadata used for historical view identity",
                      training_protocol="whole T13, same frozen A1 weights; not cross-chunk state-matched training",
                      deployment_changed=False, gan=False)
        if rank == 0:
            a.output_dir.mkdir(parents=True, exist_ok=True)
            if a.resume and json.loads((a.output_dir / "run_config.json").read_text()) != config:
                raise ValueError("Resume configuration/inputs differ; use a new explicit warm-start run")
            if not a.resume: write_json(a.output_dir / "run_config.json", config)
        barrier()
        reae = ReAE(str(a.base_checkpoint / "reae.safetensors")).to(device=device, dtype=dtype)
        decoder = M9A1FactorizedReAEDecoder.from_pretrained(a.decoder_checkpoint, device=device, dtype=dtype)
        transformer = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
            str(a.student_init), subfolder="transformer", torch_dtype=torch.float32,
            low_cpu_mem_usage=True, local_files_only=True).to(device)
        if transformer_moe_shape(transformer) != expected_moe_shape(B2BMoESpec(num_layers=20)):
            raise ValueError("P1 requires unchanged M8 D1024/H8/L20, 1 shared + 12 routed top2")
        backbone = B2BMoEVelocityDistillationForward(reae, transformer, attention_backend="sdpa",
                                                    gradient_checkpointing=not a.no_gradient_checkpointing)
        transformer.requires_grad_(True)
        closure = M8TaskForward(backbone, decoder, checkpoint_decoder=not a.no_decoder_checkpointing).train()
        set_router_stats_collection(transformer, False)
        perceptual = LocalLPIPS(a.lpips_weights, frame_batch=a.lpips_frame_batch).to(device) if a.lpips_weights else None
        params = list(transformer.parameters())
        optimizer = torch.optim.AdamW(params, lr=a.learning_rate, weight_decay=0.0, eps=1e-8, foreach=True)
        step, epoch, offset = 0, 0, 0
        if a.resume:
            # Only load training_state.pt produced by this trusted local trainer.
            state = torch.load(a.resume / "training_state.pt", map_location="cpu", weights_only=False)
            if state["config"] != config: raise ValueError("Checkpoint lineage mismatch")
            transformer.load_state_dict(load_file(str(a.resume / "transformer/diffusion_pytorch_model.safetensors")), strict=True)
            optimizer.load_state_dict(state["optimizer"])
            step, epoch, offset = state["step"], state["epoch"], state["offset"]
            restore_rng(state["rng_by_rank"][rank])
        ddp = DDP(closure, device_ids=[local_rank], broadcast_buffers=False,
                  find_unused_parameters=True, gradient_as_bucket_view=True) if world > 1 else closure
        sampler.set_epoch(epoch); cursor_sampler.offset = offset
        iterator = iter(loader)
        stop = step + a.smoke_steps if a.smoke_steps else a.max_steps
        if stop <= step: raise ValueError("Stop update must exceed resumed step")
        weights = dict(pixel_weight=a.pixel_weight, temporal_weight=a.temporal_weight,
                       lpips_weight=a.lpips_weight, router_weight=a.router_weight)

        def save(step):
            states = [None] * world
            if world > 1: dist.all_gather_object(states, rng_state())
            else: states[0] = rng_state()
            path = a.output_dir / "checkpoints" / f"step_{step:08d}"
            if rank == 0:
                if path.exists(): raise FileExistsError(path)
                model_dir = path / "transformer"; model_dir.mkdir(parents=True)
                transformer.save_config(model_dir)
                # Preserve FP32 training masters. Never cast live weights for snapshots.
                save_file({k:v.detach().cpu().contiguous() for k,v in transformer.state_dict().items()},
                          str(model_dir / "diffusion_pytorch_model.safetensors"), metadata={"format":"pt"})
                state = {"step":step, "epoch":epoch, "offset":offset, "config":config,
                         "optimizer":optimizer.state_dict(), "rng_by_rank":states}
                torch.save(state, path / "training_state.tmp")
                (path / "training_state.tmp").replace(path / "training_state.pt")
                write_json(path / "metadata.json", {"global_step":step, "decoder_checkpoint":str(a.decoder_checkpoint),
                                                     "train_scope":"M8_only", "gt_role":config["gt_role"]})
                write_json(a.output_dir / "latest.json", {"checkpoint":str(path), "global_step":step})
            barrier()
            return path

        def validate(step, checkpoint_path):
            barrier()
            rng = rng_state()
            clear_caches()
            try:
                if rank == 0:
                    closure.eval()
                    metric = VideoMetricAccumulator(); totals = {}; visuals=[]; count=0
                    with torch.no_grad():
                        for batch_cpu in val_loader:
                            batch = {k:v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k,v in batch_cpu.items()}
                            with torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
                                out = closure(batch)
                                terms = task_objective(out, perceptual=perceptual, **weights)
                            pred, gt = out["prediction"], out["target"]
                            metric.update(pred, gt, clamp=True)
                            terms.update(phase_errors(pred, gt))
                            if perceptual is not None and a.lpips_weight == 0:
                                terms["lpips"] = perceptual(pred, gt)
                            for key,val in terms.items(): totals[key] = totals.get(key, 0.0) + float(val)
                            count += 1
                            # Same panel format as old visual exporter; target is now real HR, not teacher.
                            lq = torch.nn.functional.interpolate(batch["lr"].flatten(0,1).float(), size=gt.shape[-2:], mode="bilinear", align_corners=False).reshape_as(gt)
                            visuals.append({"record_uid":batch_cpu["pair_id"][0], "lq_input":lq[0].cpu(),
                                            "target":gt[0].cpu(), "teacher_prediction":gt[0].cpu(), "student_prediction":pred[0].float().cpu()})
                    report = {"step":step, "gt":metric.compute(), **{k:v/count for k,v in totals.items()}}
                    # Write named GT comparison PNGs directly: no 'teacher' label for HR.
                    from swiftvr.training.perceptual_review import make_comparison_frame
                    folder = a.output_dir / "validation_visuals" / f"step_{step:08d}"; folder.mkdir(parents=True)
                    for i,v in enumerate(visuals):
                        for t in range(v["target"].shape[0]):
                            make_comparison_frame({"LQ":v["lq_input"][t], "Current A1":v["student_prediction"][t], "GT":v["target"][t]}).save(folder / f"sample_{i:02d}_frame_{t:03d}.png")
                    write_json(folder / "metrics.json", report)
                    append_json(a.output_dir / "val_log.jsonl", report)
                    print("[val] " + json.dumps(report), flush=True)
                    if not a.smoke_steps:
                        del visuals, batch, out, pred, gt, terms
                        torch.cuda.empty_cache()
                        env = os.environ.copy()
                        visible = env.get("CUDA_VISIBLE_DEVICES", "")
                        env["GPU"] = visible.split(",")[local_rank].strip() if visible else str(local_rank)
                        env["CUSTOM_REF_ROOT"] = str(a.custom_ref_root)
                        env["BASE"] = str(a.base_checkpoint); env["A1"] = str(a.decoder_checkpoint)
                        # Isolate ordinary inference from torchrun distributed environment.
                        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                            env.pop(key, None)
                        custom = a.output_dir / "custom_validation" / f"step_{step:08d}"
                        subprocess.run(["bash", str(a.custom_hook), "validate", str(checkpoint_path), str(custom)],
                                       cwd=ROOT, env=env, check=True)
            finally:
                clear_caches(); closure.train(); restore_rng(rng)
            barrier()

        if not a.resume and not a.smoke_steps:
            initial = save(0)
            validate(0, initial)
        started = time.perf_counter(); log_started=started; recent = {}; seen=0
        optimizer.zero_grad(set_to_none=True)
        while step < stop:
            data_seconds = 0.0
            for group in optimizer.param_groups:
                group["lr"] = a.learning_rate * lr_scale(step+1, a.warmup_steps, a.schedule_steps)
            for micro in range(a.gradient_accumulation_steps):
                before = time.perf_counter()
                if offset >= usable:
                    epoch += 1; offset=0; sampler.set_epoch(epoch); cursor_sampler.offset=0
                    iterator=iter(loader)
                batch_cpu=next(iterator); offset += a.batch_size
                data_seconds += time.perf_counter()-before
                batch = {k:v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k,v in batch_cpu.items()}
                sync = ddp.no_sync() if world > 1 and micro+1 < a.gradient_accumulation_steps else nullcontext()
                with sync:
                    with torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
                        out=ddp(batch); terms=task_objective(out, perceptual=perceptual, **weights)
                        # Smoke proves GT reaches M8, without router masking a broken decode graph.
                        loss=terms["gt_loss"] if a.smoke_steps else terms["loss"]
                    (loss / a.gradient_accumulation_steps).backward()
                for key,value in terms.items():
                    recent[key] = recent.get(key, torch.zeros((), device=device)) + value.detach() / a.gradient_accumulation_steps
            norm=torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            if a.smoke_steps:
                good=any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in params)
                frozen_ok=all(p.grad is None for m in (closure.backbone.reae, decoder) for p in m.parameters())
                if not good or not frozen_ok: raise RuntimeError("GT-only gradient path/frozen-scope smoke failed")
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step+=1; seen+=1
            del out, terms, loss, batch, batch_cpu
            if step % a.log_every == 0 or step == stop:
                values=torch.stack([v for v in recent.values()]) / seen
                if world > 1: dist.all_reduce(values); values/=world
                if rank==0:
                    now=time.perf_counter()
                    report={"step":step, "lr":optimizer.param_groups[0]["lr"], "grad_norm_rank0":float(norm),
                            "views_presented":step*config["global_batch"],
                            "updates_per_second":seen/(now-log_started), "last_update_loader_wait_s":data_seconds,
                            **dict(zip(recent, values.cpu().tolist()))}
                    append_json(a.output_dir / "train_log.jsonl", report); print(json.dumps(report), flush=True)
                recent={};seen=0;log_started=time.perf_counter()
            due_val=step % a.validate_every==0 or step==stop
            if due_val or step % a.save_every==0:
                path=save(step)
                if due_val: validate(step,path)
        if rank==0:
            write_json(a.output_dir / "summary.json", {"step":step,"status":"smoke_pass" if a.smoke_steps else "completed_budget",
                       "gt_only_backward_checked":bool(a.smoke_steps), "visual_quality_pass":False,
                       "seconds_this_invocation":time.perf_counter()-started, "config":config})
    finally:
        if world>1 and dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
