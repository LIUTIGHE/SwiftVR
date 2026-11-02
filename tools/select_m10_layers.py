#!/usr/bin/env python3
"""M10-LS: static D1024 L30->L20 selection with matched short recovery.

screen: bounded cross-region swaps, scored on full endpoint outputs.
recover: reset every candidate to the SAME parent, train only Transformer with
         Stage-A velocity NMSE + router regularization; rank by heldout LPIPS.
validate: only after selection is locked, render the untouched formal val13.

This is a discrete shortlist/recovery baseline, not a TinyFusion/TinySR replica.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.m10_layer_search_core import (
    checked_mask, compact_block_view, mask_id, propose_swaps,
    recovered_winner, shortlist, source_disjoint_split, training_order,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("--base-checkpoint", type=Path, required=True)
    screen.add_argument("--decoder-checkpoint", type=Path, required=True)
    screen.add_argument("--depth-init", type=Path, required=True,
                        help="Historical M8 init containing moe_depth_init_report.json")
    screen.add_argument("--source-checkpoint", type=Path,
                        help="Optional relocated M5 L30; must match historical SHA256")
    screen.add_argument("--baseline-checkpoint", type=Path, required=True,
                        help="Immutable mature M8-A30k; diagnostic only, never reset/trained")
    screen.add_argument("--teacher-cache", type=Path, required=True)
    screen.add_argument("--val-teacher-cache", type=Path, required=True)
    screen.add_argument("--path-root", type=Path, default=Path("."))
    screen.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    screen.add_argument("--probe-views", type=int, default=8)
    screen.add_argument("--rank-views", type=int, default=16)
    screen.add_argument("--adapt-views", type=int, default=128)
    screen.add_argument("--proposal-width", type=int, default=3)
    screen.add_argument("--swap-rounds", type=int, default=2)
    screen.add_argument("--candidates", type=int, default=4)
    screen.add_argument("--seed", type=int, default=0)
    recover = sub.add_parser("recover")
    recover.add_argument("--steps", type=int, default=250)
    recover.add_argument("--accumulation", type=int, default=4)
    recover.add_argument("--learning-rate", type=float, default=1e-5)
    recover.add_argument("--warmup-steps", type=int, default=25)
    validate = sub.add_parser("validate")
    for s in (screen, recover, validate):
        s.add_argument("--work-dir", type=Path, required=True)
        s.add_argument("--device", default="cuda")
    from tools.m10_layer_confirmation import add_arguments
    add_arguments(sub)
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def _fresh(path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Use a new output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _weights(root):
    paths = sorted((Path(root) / "transformer").glob("*.safetensors"))
    if not paths or not (Path(root) / "transformer/config.json").is_file():
        raise FileNotFoundError(f"Missing local Transformer config/weights: {root}")
    return paths


def _dataset(cache, path_root):
    from tools.train_teacher_distillation_ddp import build_cached_dataset
    m = cache.metadata
    return build_cached_dataset(
        [Path(v) for v in m["manifests"]], cache, split=m["split"],
        path_root=Path(path_root), clip_length=m["clip_length"], crop_size=m["crop_size"],
        scale=m["scale"], views_per_record=m["views_per_record"], view_seed=m["view_seed"],
        hflip=m["horizontal_flip_probability"], vflip=m["vertical_flip_probability"],
        verify_paths=False,
    )


def _load_model(root, layers, device, dtype):
    from swiftvr.models import WanTransformer3DModelPromptFreeNoTimeMoE
    from swiftvr.training.b2b_moe import B2BMoESpec, expected_moe_shape, transformer_moe_shape
    from swiftvr.training.forward import prepare_prompt_free_no_time_transformer_for_training
    _weights(root)
    model = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
        str(root), subfolder="transformer", torch_dtype=dtype,
        local_files_only=True, low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype)
    actual = transformer_moe_shape(model)
    if actual != expected_moe_shape(B2BMoESpec(num_layers=layers)):
        raise ValueError(f"Expected unchanged D1024 L{layers} MoE, got {actual}")
    prepare_prompt_free_no_time_transformer_for_training(model, attention_backend="sdpa")
    return model.requires_grad_(False).eval()


def _copy_candidate(parent, mask, device, dtype):
    from tools.build_b2b_moe_depth_init import _build_depth_student, _copy_depth_subset
    from swiftvr.training.forward import prepare_prompt_free_no_time_transformer_for_training
    # Canonical exact whole-block transfer; routers/experts are NOT reinitialized.
    model = _build_depth_student(parent, len(mask))
    _copy_depth_subset(parent, model, list(mask))
    model.to(device=device, dtype=dtype)
    prepare_prompt_free_no_time_transformer_for_training(model, attention_backend="sdpa")
    return model


def _local_perceptual(device):
    import torch
    from urllib.parse import urlparse
    from torchvision.models import AlexNet_Weights
    from swiftvr.training.tiny_decoder import LPIPSAlexLoss
    # A1 already used this exact loss. Preflight avoids its pretrained download.
    filename = Path(urlparse(AlexNet_Weights.IMAGENET1K_V1.url).path).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Existing A1 AlexNet cache required: {path}; no download attempted")
    return LPIPSAlexLoss().to(device).requires_grad_(False).eval(), path


def _frozen(config, device, dtype):
    from swiftvr.models import ReAE
    from swiftvr.models.m9_factorized_decoder import M9A1FactorizedReAEDecoder
    reae = ReAE(str(Path(config["base"]) / "reae.safetensors")).to(device=device, dtype=dtype)
    decoder = M9A1FactorizedReAEDecoder.from_pretrained(config["decoder"], device=device, dtype=dtype)
    return reae.requires_grad_(False).eval(), decoder.requires_grad_(False).eval()


def _fingerprint(paths):
    from swiftvr.training.reference import sha256_file
    return {str(Path(p).resolve()): sha256_file(p) for p in paths}


def _verify(config):
    current = _fingerprint(config["immutable_sha256"])
    if current != config["immutable_sha256"]:
        raise ValueError("Immutable checkpoint/cache metadata changed since screening")


def _prepare_packets(dataset, cache, indices, folder, reae, device, dtype, *, pixels):
    import torch
    from torch.utils.data import DataLoader, Subset
    from tools.smoke_training_forward import move_video_batch
    from swiftvr.training.distillation import distillation_sample_identity
    from swiftvr.training.forward import prepare_training_batch, encode_reae_clip, decode_reae_clip
    folder.mkdir(parents=True, exist_ok=False)
    paths = []
    with torch.no_grad():
        for n, batch in enumerate(DataLoader(Subset(dataset, indices), batch_size=1, num_workers=0)):
            target_v = cache.load_batch(batch, device=device, dtype=dtype)
            prepared = prepare_training_batch(move_video_batch(batch, device=device, dtype=dtype))
            with torch.autocast(device.type, dtype=dtype, enabled=dtype == torch.bfloat16):
                z = encode_reae_clip(reae, prepared["lq_input"]).permute(0, 2, 1, 3, 4).contiguous()
                if z.shape != target_v.shape:
                    raise ValueError("Encoded input and Stage-A velocity cache shape differ")
                packet = {"z": z.cpu(), "v": target_v.cpu(), "frames": int(prepared["target"].shape[1]),
                          "identity": distillation_sample_identity(batch, 0)}
                if pixels:
                    teacher = decode_reae_clip(reae, (z - target_v).permute(0, 2, 1, 3, 4).contiguous(),
                                               output_frames=packet["frames"], clamp=True)
                    packet.update(teacher=teacher.float().cpu(), gt=prepared["target"].float().cpu(),
                                  lq=prepared["lq_input"].float().cpu())
            path = folder / f"{n:05d}.pt"
            torch.save(packet, path)
            paths.append(path)
            if (n + 1) % 16 == 0 or n + 1 == len(indices):
                print(f"[encode {folder.name}] {n + 1}/{len(indices)}", flush=True)
    return paths


def _packet(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def _velocity(model, z, dtype, *, grad=False):
    import torch
    from swiftvr.training.b2b_moe_training import forward_moe_transformer_training
    with torch.autocast(z.device.type, dtype=dtype, enabled=dtype == torch.bfloat16):
        return forward_moe_transformer_training(model, z, gradient_checkpointing=grad)


def _nmse(v, target):
    import torch.nn.functional as F
    return F.mse_loss(v.float(), target.float()) / target.float().square().mean().clamp_min(1e-8)


def _probe(model, paths, device, dtype):
    import torch
    total = 0.0
    with torch.no_grad():
        for path in paths:
            p = _packet(path)
            z, tv = p["z"].to(device, dtype=dtype), p["v"].to(device, dtype=dtype)
            v, _ = _velocity(model, z, dtype)
            total += float(_nmse(v, tv))
    score = total / len(paths)
    if not math.isfinite(score):
        raise FloatingPointError("Non-finite endpoint screening score")
    return score


def _evaluate(model, paths, decoder, perceptual, device, dtype, output, *, visual_samples):
    import torch
    from swiftvr.training.stage3 import VideoMetricAccumulator
    from swiftvr.training.distillation_visuals import export_validation_visuals
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    metrics = {k: VideoMetricAccumulator() for k in ("student_teacher", "student_gt", "teacher_gt")}
    total_lpips = total_nmse = 0.0
    phase_sums, phase_counts, visuals = [0.0] * 4, [0] * 4, []
    with torch.no_grad():
        for path in paths:
            p = _packet(path)
            z, tv = p["z"].to(device, dtype=dtype), p["v"].to(device, dtype=dtype)
            v, _ = _velocity(model, z, dtype)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype == torch.bfloat16):
                s = decoder((z - v).permute(0, 2, 1, 3, 4).contiguous(), output_frames=p["frames"], clamp=True)
            teacher, gt = p["teacher"].to(device), p["gt"].to(device)
            total_nmse += float(_nmse(v, tv))
            # Evaluation only: no perceptual/RGB gradient or extra training loss.
            total_lpips += float(perceptual.forward_video(s.float(), teacher, microbatch_frames=4))
            metrics["student_teacher"].update(s, teacher, clamp=True)
            metrics["student_gt"].update(s, gt, clamp=True)
            metrics["teacher_gt"].update(teacher, gt, clamp=True)
            errors = (s.float() - teacher).square().mean((0, 2, 3, 4)).cpu().tolist()
            for t in range(1, len(errors)):
                phase = (t + 3) % 4
                phase_sums[phase] += errors[t]
                phase_counts[phase] += 1
            if len(visuals) < visual_samples:
                visuals.append({"record_uid": p["identity"]["record_uid"],
                                "lq_input": p["lq"][0], "target": p["gt"][0],
                                "teacher_prediction": p["teacher"][0], "student_prediction": s[0].float().cpu()})
    result = {"views": len(paths), "lpips": total_lpips / len(paths), "velocity_nmse": total_nmse / len(paths),
              "phase_teacher_mse_exclude_t0": {str(i): phase_sums[i] / phase_counts[i] if phase_counts[i] else None for i in range(4)},
              "phase_counts": phase_counts, "gt_role": "diagnostic_only"}
    for key, acc in metrics.items():
        result.update({f"{key}_{k}": v for k, v in acc.compute().items()})
    result["rgb_mse"] = result["student_teacher_mse"]
    write_json(output / "metrics.json", result)
    if visuals:
        frames = range(min(int(v["student_prediction"].shape[0]) for v in visuals))
        export_validation_visuals(visuals, output_root=output, step=0, frame_indices=frames, fps=30)
    print(f"[eval {output.name}] LPIPS={result['lpips']:.6f} PSNR={result['student_teacher_psnr']:.4f}", flush=True)
    return result


def _save(model, root, dtype, metadata):
    root.mkdir(parents=True, exist_ok=False)
    model.to(device="cpu", dtype=dtype)
    model.save_pretrained(str(root / "transformer"), safe_serialization=True)
    write_json(root / "selection_lineage.json", metadata)


def screen(a, device):
    import torch
    from swiftvr.training import TeacherVelocityCache
    from swiftvr.training.m10_recovery import validate_teacher_metadata
    from swiftvr.training.reference import sha256_file
    root = a.work_dir.resolve()
    if min(a.probe_views, a.rank_views, a.adapt_views, a.proposal_width, a.swap_rounds) <= 0 or a.candidates < 2:
        raise ValueError("Positive split/search sizes and >=2 candidates required")
    depth_root = a.depth_init.resolve()
    depth_report = depth_root / "moe_depth_init_report.json"
    historical = read_json(depth_report)
    source = (a.source_checkpoint or Path(historical["source_checkpoint"])).resolve()
    source_files = _weights(source)
    if len(source_files) != 1 or sha256_file(source_files[0]) != historical["source_weights_sha256"]:
        raise ValueError("M5 source does not match the historical M8 depth-init weight fingerprint")
    if sha256_file(source / "transformer/config.json") != historical["source_config_sha256"]:
        raise ValueError("M5 source config changed")
    mask = checked_mask(historical["selection"]["kept_source_blocks"], keep=20)
    cache, val_cache = TeacherVelocityCache(a.teacher_cache), TeacherVelocityCache(a.val_teacher_cache)
    base, dec, baseline = a.base_checkpoint.resolve(), a.decoder_checkpoint.resolve(), a.baseline_checkpoint.resolve()
    reae_hash = sha256_file(base / "reae.safetensors")
    validate_teacher_metadata(cache.metadata, val_cache.metadata, reae_sha256=reae_hash)
    if cache.metadata["split"] != "train" or val_cache.metadata["split"] != "val":
        raise ValueError("Need Stage-A TRAIN/VAL caches, never the M5 TA or M8 z_SR cache")
    splits = source_disjoint_split(cache.metadata["samples"], val_cache.metadata["samples"],
                                  probe=a.probe_views, rank=a.rank_views, adapt=a.adapt_views, seed=a.seed)
    for immutable in (base, dec, baseline, source, depth_root, cache.root, val_cache.root):
        if root == immutable or immutable in root.parents:
            raise ValueError(f"Output must not be inside immutable input: {immutable}")
    files = [base / "reae.safetensors", dec / "config.json", dec / "model.safetensors", depth_report,
             source / "transformer/config.json", *source_files, baseline / "transformer/config.json", *_weights(baseline),
             cache.root / "metadata.json", val_cache.root / "metadata.json"]
    config = {"kind": "m10_static_L30_L20_recovery_selection_v1", "base": str(base), "decoder": str(dec),
              "source": str(source), "baseline": str(baseline), "depth_init": str(depth_root),
              "teacher_cache": str(cache.root), "val_teacher_cache": str(val_cache.root),
              "path_root": str(a.path_root.resolve()), "dtype": a.dtype, "seed": a.seed,
              "baseline_mask": list(mask), "splits": splits, "immutable_sha256": _fingerprint(files),
              "search": {"proposal_width": a.proposal_width, "swap_rounds": a.swap_rounds, "candidates": a.candidates},
              "target_layers": 20, "shift_policy": "compact_index_mod2_in_screen_recovery_export",
              "reference": "Stage-A200k velocity and OriginalD RGB; candidate/parent use frozen A1",
              "train_objective": "velocity_NMSE + 0.01 router_balance; no RGB/HF/LPIPS training",
              "ranking": "source-disjoint TRAIN rank-set LPIPS, then RGB MSE; GT/val13 never select",
              "scope": "single GPU sequential; bounded discrete search, not global optimum/TinyFusion reproduction"}
    perceptual, lpips_path = _local_perceptual(device)
    config["lpips_trunk_sha256"] = {str(lpips_path): sha256_file(lpips_path)}
    config["immutable_sha256"].update(config["lpips_trunk_sha256"])
    _fresh(root)
    write_json(root / "run_config.json", config)
    dtype = getattr(torch, a.dtype)
    reae, decoder = _frozen(config, device, dtype)
    dataset = _dataset(cache, config["path_root"])
    for group in ("probe", "adapt", "rank"):
        _prepare_packets(dataset, cache, splits[group], root / "packets" / group, reae, device, dtype, pixels=group == "rank")
    reae.to("cpu")
    del reae, dataset
    probe_paths = sorted((root / "packets/probe").glob("*.pt"))
    rank_paths = sorted((root / "packets/rank").glob("*.pt"))
    parent = _load_model(source, 30, device, dtype)
    scores, trace = {}, []

    def score(kept):
        kept = tuple(kept)
        if kept not in scores:
            with compact_block_view(parent, kept):
                value = _probe(parent, probe_paths, device, dtype)
            scores[kept] = value
            entry = {"kept_source_blocks": list(kept), "layers": len(kept), "probe_velocity_nmse": value}
            trace.append(entry)
            with (root / "screen_trace.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"[screen {len(trace)}] L{len(kept)} {value:.6f} kept={list(kept)}", flush=True)
        return scores[kept]

    score(tuple(range(30)))
    score(mask)
    current = mask
    candidate_scores = {mask: scores[mask]}
    for _ in range(a.swap_rounds):
        drops = {i: score(tuple(k for k in current if k != i)) for i in current if i not in (0, 29)}
        adds = {i: score(tuple(sorted((*current, i)))) for i in range(1, 29) if i not in current}
        for candidate in propose_swaps(current, drops, adds, width=a.proposal_width):
            candidate_scores[candidate] = score(candidate)
        best = min(candidate_scores, key=lambda m: (candidate_scores[m], m))
        if best == current or candidate_scores[best] >= candidate_scores[current]:
            break
        current = best
    masks = shortlist(candidate_scores, mask, count=a.candidates)
    candidates = [{"id": "heuristic" if m == mask else "swap_" + mask_id(m),
                   "kept_source_blocks": list(m), "probe_velocity_nmse": candidate_scores[m]} for m in masks]
    # Shortlist is frozen before rendering search-heldout images.
    write_json(root / "candidates.json", {"candidates": candidates, "evaluated_masks": len(trace),
                                         "screen_only": True, "not_final_selection": True})
    _evaluate(parent, rank_paths, decoder, perceptual, device, dtype, root / "screen_reference/parent_L30", visual_samples=3)
    with compact_block_view(parent, mask):
        _evaluate(parent, rank_paths, decoder, perceptual, device, dtype, root / "screen_reference/heuristic_init_L20", visual_samples=3)
    del parent
    gc.collect(); torch.cuda.empty_cache()
    mature = _load_model(baseline, 20, device, dtype)
    _evaluate(mature, rank_paths, decoder, perceptual, device, dtype, root / "screen_reference/mature_M8A30k", visual_samples=3)
    print(f"Screening complete: {len(candidates)} candidates. No weights trained; run recover next. {root}", flush=True)


def recover(a, device):
    import torch
    from swiftvr.training import build_fp32_adamw, cast_trainable_parameters
    root = a.work_dir.resolve()
    config = read_json(root / "run_config.json")
    _verify(config)
    if min(a.steps, a.accumulation) <= 0 or not 0 <= a.warmup_steps < a.steps or not math.isfinite(a.learning_rate) or a.learning_rate <= 0:
        raise ValueError("Invalid recovery steps/accumulation/warmup/LR")
    out = root / "recovery"
    _fresh(out)
    dtype = getattr(torch, config["dtype"])
    perceptual, _ = _local_perceptual(device)
    reae, decoder = _frozen(config, device, dtype)
    reae.to("cpu"); del reae
    parent = _load_model(config["source"], 30, "cpu", dtype)
    adapt = sorted((root / "packets/adapt").glob("*.pt"))
    rank = sorted((root / "packets/rank").glob("*.pt"))
    if len(adapt) != len(config["splits"]["adapt"]) or len(rank) != len(config["splits"]["rank"]):
        raise ValueError("Prepared packet count mismatch; screening must complete first")
    schedule = training_order(len(adapt), a.steps, a.accumulation, config["seed"])
    recovery_config = {"steps": a.steps, "accumulation": a.accumulation, "learning_rate": a.learning_rate,
                       "warmup_steps": a.warmup_steps, "seed": config["seed"], "schedule": schedule,
                       "train_scope": "all Transformer parameters only; no decoder gradients", "router_weight": 0.01}
    write_json(out / "recovery_config.json", recovery_config)
    records = []
    for candidate in read_json(root / "candidates.json")["candidates"]:
        cid = candidate["id"]
        folder = out / cid
        folder.mkdir()
        mask = checked_mask(candidate["kept_source_blocks"], keep=20)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model = _copy_candidate(parent, mask, "cpu", dtype)
        init_path = folder / "init"
        _save(model, init_path, dtype, {**candidate, "source": config["source"], "work_config": str(root / "run_config.json")})
        model.to(device).requires_grad_(False).eval()
        initial = _evaluate(model, rank, decoder, perceptual, device, dtype, folder / "before", visual_samples=0)
        model.requires_grad_(True).train()
        cast_trainable_parameters(model, dtype=torch.float32)
        optimizer = build_fp32_adamw(model, learning_rate=a.learning_rate, weight_decay=0.0, eps=1e-8)
        for step in range(1, a.steps + 1):
            lr = a.learning_rate * min(1.0, step / max(1, a.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total_loss = total_nmse = total_router = 0.0
            for micro in range(a.accumulation):
                p = _packet(adapt[schedule[(step - 1) * a.accumulation + micro]])
                z, tv = p["z"].to(device, dtype=dtype), p["v"].to(device, dtype=dtype)
                v, balance = _velocity(model, z, dtype, grad=True)
                nmse = _nmse(v, tv)
                loss = nmse + 0.01 * balance.float()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{cid}: non-finite training loss at {step}")
                (loss / a.accumulation).backward()
                total_loss += float(loss.detach()) / a.accumulation
                total_nmse += float(nmse.detach()) / a.accumulation
                total_router += float(balance.detach()) / a.accumulation
                del v, loss, nmse, balance
            if any(p.grad is not None for p in decoder.parameters()):
                raise RuntimeError("Frozen decoder unexpectedly acquired gradients")
            grads = [p for p in model.parameters() if p.grad is not None]
            if not grads:
                raise RuntimeError("No Transformer gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(grads, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 25 == 0 or step == a.steps:
                row = {"step": step, "lr": lr, "loss": total_loss, "velocity_nmse": total_nmse,
                       "router_balance": total_router, "preclip_grad_norm": float(grad_norm)}
                with (folder / "train_log.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[recover {cid}] {json.dumps(row)}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, grads, grad_norm
        # Rank the saved deployment-precision weights, not FP32 master weights.
        trained = folder / f"step_{a.steps:08d}"
        _save(model, trained, dtype, {**candidate, "recovery_config": str(out / "recovery_config.json"),
                                     "source": config["source"], "weights_only": True})
        model.to(device).requires_grad_(False).eval()
        final = _evaluate(model, rank, decoder, perceptual, device, dtype, folder / "after", visual_samples=3)
        records.append({"id": cid, "steps": a.steps, "kept_source_blocks": list(mask),
                        "init_checkpoint": str(init_path), "checkpoint": str(trained),
                        "lpips": final["lpips"], "rgb_mse": final["rgb_mse"],
                        "before": initial, "after": final})
        del model
        gc.collect(); torch.cuda.empty_cache()
    chosen = recovered_winner(records)
    selection = {"kind": "m10_layer_selection_locked_before_val13", "selected": chosen,
                 "candidates": records, "ranking": "TRAIN-heldout LPIPS then RGB MSE; equal recovery steps",
                 "quality_status": "VISUAL_REVIEW_REQUIRED_NOT_A_NEW_MILESTONE",
                 "selected_nonheuristic": chosen["id"] != "heuristic"}
    write_json(root / "selection.json", selection)
    _verify(config)
    print(f"Selected {chosen['id']}: {chosen['kept_source_blocks']}. Official val13 not yet used.", flush=True)


def validate(a, device):
    import torch
    from swiftvr.training import TeacherVelocityCache
    root = a.work_dir.resolve()
    config, selection = read_json(root / "run_config.json"), read_json(root / "selection.json")
    _verify(config)
    out = root / "formal_val13"
    _fresh(out)
    dtype = getattr(torch, config["dtype"])
    perceptual, _ = _local_perceptual(device)
    reae, decoder = _frozen(config, device, dtype)
    cache = TeacherVelocityCache(config["val_teacher_cache"])
    dataset = _dataset(cache, config["path_root"])
    if len(dataset) != 13:
        raise ValueError(f"Expected existing formal val13, got {len(dataset)}")
    paths = _prepare_packets(dataset, cache, list(range(len(dataset))), out / "packets", reae, device, dtype, pixels=True)
    reae.to("cpu"); del reae
    old = next(r for r in selection["candidates"] if r["id"] == "heuristic")
    models = [("parent_L30", config["source"], 30), ("mature_M8A30k", config["baseline"], 20),
              ("heuristic_matched_recovery", old["checkpoint"], 20)]
    if selection["selected"]["id"] != "heuristic":
        models.append(("selected_matched_recovery", selection["selected"]["checkpoint"], 20))
    results = {}
    for name, checkpoint, layers in models:
        model = _load_model(checkpoint, layers, device, dtype)
        results[name] = _evaluate(model, paths, decoder, perceptual, device, dtype, out / name, visual_samples=13)
        del model
        gc.collect(); torch.cuda.empty_cache()
    write_json(out / "report.json", {"selection_locked": str(root / "selection.json"), "metrics": results,
                                    "all_students_use_same_frozen_a1": True, "no_selection_on_val13": True,
                                    "mature_baseline_has_different_training_budget": True})


def main():
    a = build_parser().parse_args()
    import torch
    device = torch.device(a.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Real selection/recovery requires a CUDA GPU; CPU tests use synthetic blocks")
    dtype_name = a.dtype if a.stage == "screen" else read_json(a.work_dir / "run_config.json")["dtype"]
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 mode requires a BF16-capable GPU")
    if a.stage == "confirm":
        from tools.m10_layer_confirmation import run
        run(a, device)
    else:
        {"screen": screen, "recover": recover, "validate": validate}[a.stage](a, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
