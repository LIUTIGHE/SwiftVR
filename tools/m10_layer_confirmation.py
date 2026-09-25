"""Confirm two locked L20 masks on all non-heldout cached training views.

Invoked by select_m10_layers.py confirm. Selection is never re-run. The old
selector supplies loading, forward, cache validation, metrics and visual export.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch

from tools import select_m10_layers as ls
from tools.m10_layer_search_core import checked_mask, training_order


def add_arguments(subparsers):
    p = subparsers.add_parser("confirm", help="Longer equal-budget recovery of heuristic and locked selected mask")
    p.add_argument("--work-dir", type=Path, required=True, help="Existing completed layer-selection work directory")
    p.add_argument("--output-dir", type=Path, required=True, help="New confirmation directory; never overwrite search results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=2000, help="Total updates PER candidate, including resumed updates")
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--visual-samples", type=int, default=13)
    p.add_argument("--resume", action="store_true", help="Restore FP32 weights, optimizer, RNG and data position; may increase --steps")


def locked_pair(selection):
    """Use the two init checkpoints, never the 250-step search checkpoints."""
    selected = selection["selected"]
    candidates = selection["candidates"]
    ids = [c["id"] for c in candidates]
    if len(set(ids)) != len(ids) or ids.count("heuristic") != 1 or selected["id"] not in ids:
        raise ValueError("Ambiguous locked candidate records")
    old = next(c for c in candidates if c["id"] == "heuristic")
    new = next(c for c in candidates if c["id"] == selected["id"])
    if old["id"] == new["id"] or new["kept_source_blocks"] != selected["kept_source_blocks"]:
        raise ValueError("Expected a distinct locked selected mask and its matching record")
    pair = []
    for role, record in (("heuristic", old), ("selected", new)):
        mask = checked_mask(record["kept_source_blocks"], keep=20)
        init = Path(record["init_checkpoint"]).expanduser().resolve()
        if init.name != "init":
            raise ValueError(f"Expected the parent-derived init checkpoint, got {init}")
        pair.append({"role": role, "id": record["id"], "kept_source_blocks": list(mask), "init_checkpoint": str(init)})
    if pair[0]["kept_source_blocks"] == pair[1]["kept_source_blocks"]:
        raise ValueError("The confirmation masks must differ")
    return pair


def training_indices(train_samples, val_samples, splits):
    """Expand beyond adapt128; exclude whole probe/rank/val sources, not just views."""
    groups = splits["source_groups"]
    excluded = set(groups["probe"]) | set(groups["rank"]) | set(splits["excluded_validation_sources"])
    all_samples = [*train_samples, *val_samples]
    if any(not isinstance(s.get("sample_id"), str) or not s["sample_id"] for s in all_samples):
        raise ValueError("Every cache record needs sample_id for source-level exclusion")
    excluded.update(s["sample_id"] for s in val_samples)
    ids = [int(s["distillation_index"]) for s in train_samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate training cache indices")
    eligible = [s for s in train_samples if s["sample_id"] not in excluded]
    indices = sorted(int(s["distillation_index"]) for s in eligible)
    if not indices or not set(splits["adapt"]).issubset(indices):
        raise ValueError("Expanded training set is empty or contradicts the earlier disjoint split")
    if set(splits["probe"]) & set(indices) or set(splits["rank"]) & set(indices):
        raise ValueError("Heldout indices leaked into training")
    return {"indices": indices, "views": len(indices),
            "sources": len({s["sample_id"] for s in eligible}),
            "excluded_sources": sorted(excluded),
            "grouping": "sample_id across all variants/views; external aliases remain undetected"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _rng(device):
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def save_resume(path, model, optimizer, *, step, plan_hash, device):
    """Rolling state: FP32 master weights, not the BF16 deployment snapshot."""
    temp = path.with_name(path.name + ".tmp")
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "step": step,
                "plan_hash": plan_hash, "rng": _rng(device)}, temp)
    temp.replace(path)


def load_resume(path, model, optimizer, *, plan_hash, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["plan_hash"] != plan_hash:
        raise ValueError("Resume state belongs to a different candidate/confirmation plan")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    _restore_rng(state["rng"], device)
    return int(state["step"])


class EncodedTrainingViews:
    """Encode each actually visited view once; share derived packets between masks."""

    def __init__(self, dataset, cache, reae, root, indices, device, dtype):
        self.dataset, self.cache, self.reae = dataset, cache, reae
        self.root, self.allowed = root, set(indices)
        self.device, self.dtype = device, dtype
        root.mkdir(parents=True, exist_ok=True)

    def get(self, index):
        if index not in self.allowed:
            raise ValueError("Requested a heldout/non-training view")
        path = self.root / f"{index:08d}.pt"
        if not path.is_file():
            from torch.utils.data import default_collate
            from tools.smoke_training_forward import move_video_batch
            from swiftvr.training.distillation import distillation_sample_identity
            from swiftvr.training.forward import prepare_training_batch, encode_reae_clip
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
            # Dataset access calls torch.manual_seed: isolate BOTH CPU/CUDA RNG so
            # cache hits/misses cannot change candidate training randomness.
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                batch = default_collate([self.dataset[index]])
                tv = self.cache.load_batch(batch, device=self.device, dtype=self.dtype)
                prepared = prepare_training_batch(move_video_batch(batch, device=self.device, dtype=self.dtype))
                with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype == torch.bfloat16):
                    z = encode_reae_clip(self.reae, prepared["lq_input"]).permute(0, 2, 1, 3, 4).contiguous()
                if z.shape != tv.shape:
                    raise ValueError("Encoded input/teacher velocity shape mismatch")
                packet = {"z": z.cpu(), "v": tv.cpu(), "identity": distillation_sample_identity(batch, 0)}
                temp = path.with_name(path.name + ".tmp")
                torch.save(packet, temp)
                temp.replace(path)
        result = ls._packet(path)
        if result["identity"]["distillation_index"] != index:
            raise ValueError("Derived packet identity does not match requested view")
        return result


def train_step(model, optimizer, packets, batch_indices, *, lr, device, dtype):
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    sums = {"loss": 0.0, "velocity_nmse": 0.0, "router_balance": 0.0}
    model.train()
    for index in batch_indices:
        p = packets.get(index)
        z, tv = p["z"].to(device, dtype=dtype), p["v"].to(device, dtype=dtype)
        v, balance = ls._velocity(model, z, dtype, grad=True)
        nmse = ls._nmse(v, tv)
        loss = nmse + 0.01 * balance.float()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite confirmation loss")
        (loss / len(batch_indices)).backward()
        for key, value in (("loss", loss), ("velocity_nmse", nmse), ("router_balance", balance)):
            sums[key] += float(value.detach()) / len(batch_indices)
    gradients = [p for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise RuntimeError("No Transformer gradients")
    norm = torch.nn.utils.clip_grad_norm_(gradients, 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {**sums, "lr": lr, "preclip_grad_norm": float(norm)}


def _snapshot(model, path, dtype, metadata):
    from tools.train_b2a_compact_distill_ddp import _save_snapshot
    if path.exists():
        if ls.read_json(path / "metadata.json") != metadata:
            raise ValueError("Existing snapshot does not match this confirmation step/lineage")
        return  # Validated resume at a previously saved step.
    # Canonical exporter copies state to CPU/runtime dtype; it does not cast the
    # live FP32 model or corrupt optimizer state when saving intermediate steps.
    _save_snapshot(model, path, runtime_dtype=dtype, transformer_subfolder="transformer", metadata=metadata)


def _packet_paths(root, group, expected_indices):
    paths = sorted(root.glob("*.pt"))
    actual = [int(ls._packet(p)["identity"]["distillation_index"]) for p in paths]
    if actual != list(expected_indices):
        raise ValueError(f"{group}: missing/mismatched evaluation packets; complete prior stages first")
    return paths


def _evaluate_step(candidate_dir, step, checkpoint, config, decoder, perceptual,
                   rank_paths, val_paths, device, dtype, visual_samples):
    destination = candidate_dir / "evaluations" / f"step_{step:08d}"
    completed = destination / "metrics.json"
    if completed.is_file():
        return ls.read_json(completed)
    if destination.exists():
        shutil.rmtree(destination)  # Only this run's incomplete derived evaluation.
    destination.mkdir(parents=True)
    model = ls._load_model(checkpoint, 20, device, dtype)
    result = {"step": step, "checkpoint": str(checkpoint),
              "rank16": ls._evaluate(model, rank_paths, decoder, perceptual, device, dtype, destination / "rank16", visual_samples=0),
              "val13": ls._evaluate(model, val_paths, decoder, perceptual, device, dtype, destination / "val13", visual_samples=visual_samples)}
    ls.write_json(completed, result)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _comparison(out, pair, plan):
    rows = {}
    for c in pair:
        rows[c["role"]] = {int(p.parent.name.removeprefix("step_")): ls.read_json(p)
                            for p in (out / c["role"] / "evaluations").glob("step_*/metrics.json")}
    common = sorted(set(rows["heuristic"]) & set(rows["selected"]))
    result = {"status": "MATCHED_CHECKPOINTS_VISUAL_REVIEW_REQUIRED", "matched_steps": common,
              "train_views": plan["training"]["views"], "global_batch_size": plan["recipe"]["accumulation"],
              "metrics_are_from_deployment_precision_snapshots": True, "fixed_masks_no_new_search": True,
              "baseline_reference_report": plan["baseline_reference_report"], "comparisons": []}
    for step in common:
        old, new = rows["heuristic"][step], rows["selected"][step]
        item = {"step": step, "sample_presentations_per_candidate": step * plan["recipe"]["accumulation"],
                "data_passes_equivalent": step * plan["recipe"]["accumulation"] / plan["training"]["views"],
                "heuristic": old, "selected": new}
        for split in ("rank16", "val13"):
            a, b = old[split], new[split]
            item[split + "_selected_minus_heuristic"] = {
                "lpips": b["lpips"] - a["lpips"],
                "lpips_relative_percent": 100 * (b["lpips"] / a["lpips"] - 1) if a["lpips"] else None,
                "rgb_mse_relative_percent": 100 * (b["rgb_mse"] / a["rgb_mse"] - 1) if a["rgb_mse"] else None,
                "psnr_db": b["student_teacher_psnr"] - a["student_teacher_psnr"]}
        result["comparisons"].append(item)
    _atomic_json(out / "comparison.json", result)


def run(a, device):
    from swiftvr.training import TeacherVelocityCache, build_fp32_adamw, cast_trainable_parameters
    from swiftvr.training.reference import sha256_file
    if min(a.steps, a.accumulation, a.eval_every) <= 0 or a.warmup_steps < 0 or a.visual_samples < 0:
        raise ValueError("Invalid confirmation budget/visual settings")
    if not math.isfinite(a.learning_rate) or a.learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    work, out = a.work_dir.expanduser().resolve(), a.output_dir.expanduser().resolve()
    config = ls.read_json(work / "run_config.json")
    selection = ls.read_json(work / "selection.json")
    ls._verify(config)
    pair = locked_pair(selection)
    # Keep every previous stage, checkpoint and source directory outside output.
    protected = [work, *(Path(config[k]) for k in ("base", "decoder", "source", "baseline", "teacher_cache", "val_teacher_cache")),
                 *map(Path, config["immutable_sha256"]), *(Path(c["init_checkpoint"]) for c in pair)]
    for source in protected:
        if out == source or out in source.parents or source in out.parents:
            raise ValueError(f"Confirmation output overlaps an immutable input: {source}")
    cache, val_cache = TeacherVelocityCache(config["teacher_cache"]), TeacherVelocityCache(config["val_teacher_cache"])
    if cache.metadata["split"] != "train" or val_cache.metadata["split"] != "val":
        raise ValueError("Expected the original Stage-A train/val cache roles")
    training = training_indices(cache.metadata["samples"], val_cache.metadata["samples"], config["splits"])
    files = [work / "run_config.json", work / "selection.json", work / "formal_val13/report.json"]
    for c in pair:
        root = Path(c["init_checkpoint"])
        lineage = ls.read_json(root / "selection_lineage.json")
        if lineage["kept_source_blocks"] != c["kept_source_blocks"] or lineage["id"] != c["id"]:
            raise ValueError("Init checkpoint lineage differs from locked selection")
        files += [root / "selection_lineage.json", root / "transformer/config.json", *ls._weights(root)]
    plan = {"kind": "m10_two_mask_full_training_confirmation_v1", "work_dir": str(work), "pair": pair,
            "training": training, "input_sha256": ls._fingerprint(files), "dtype": config["dtype"], "seed": config["seed"],
            "recipe": {"accumulation": a.accumulation, "learning_rate": a.learning_rate, "warmup_steps": a.warmup_steps,
                       "eval_every": a.eval_every, "visual_samples": a.visual_samples,
                       "loss": "Stage-A velocity NMSE + 0.01 router balance", "lr_schedule": "warmup_then_constant"},
            "baseline_reference_report": str(work / "formal_val13/report.json"),
            "checkpoint_selection": "compare SAME fixed step; no independent-best or mask re-selection",
            "val_role": "existing monitored validation, not a new untouched test set"}
    plan_hash = _digest(plan)
    if a.resume:
        if ls.read_json(out / "confirmation_plan.json") != plan:
            raise ValueError("Confirmation inputs/recipe changed; --resume may only change total --steps")
    else:
        ls._fresh(out)
        ls.write_json(out / "confirmation_plan.json", plan)
    rank_paths = _packet_paths(work / "packets/rank", "rank16", config["splits"]["rank"])
    val_paths = _packet_paths(work / "formal_val13/packets", "val13", range(len(val_cache.samples_by_index)))
    if len(val_paths) != 13:
        raise ValueError("This confirmation expects the existing formal val13 packets")
    dtype = getattr(torch, config["dtype"])
    perceptual, _ = ls._local_perceptual(device)
    reae, decoder = ls._frozen(config, device, dtype)
    reae.decoder.to("cpu")  # Teachers are already rendered; only frozen E is needed online.
    dataset = ls._dataset(cache, config["path_root"])
    packets = EncodedTrainingViews(dataset, cache, reae, out / "latent_packets", training["indices"], device, dtype)
    schedule = training_order(training["views"], a.steps, a.accumulation, config["seed"])
    print(json.dumps({"train_views": training["views"], "train_sources": training["sources"],
                      "steps_per_candidate": a.steps, "effective_batch": a.accumulation,
                      "sample_presentations": a.steps * a.accumulation,
                      "data_passes_equivalent": a.steps * a.accumulation / training["views"],
                      "candidates": pair}, indent=2), flush=True)
    for candidate in pair:
        folder = out / candidate["role"]
        folder.mkdir(exist_ok=True)
        candidate_hash = _digest({"plan": plan_hash, "candidate": candidate["id"]})
        torch.manual_seed(config["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
        model = ls._load_model(candidate["init_checkpoint"], 20, device, dtype).requires_grad_(True).train()
        cast_trainable_parameters(model, dtype=torch.float32)
        optimizer = build_fp32_adamw(model, learning_rate=a.learning_rate, weight_decay=0.0, eps=1e-8)
        state_path = folder / "resume.pt"
        step = load_resume(state_path, model, optimizer, plan_hash=candidate_hash, device=device) if a.resume and state_path.exists() else 0
        if step > a.steps:
            raise ValueError("Cannot resume backwards to a smaller total step count")

        def checkpoint_and_evaluate(current):
            state = _rng(device)
            path = Path(candidate["init_checkpoint"]) if current == 0 else folder / "checkpoints" / f"step_{current:08d}"
            if current:
                _snapshot(model, path, dtype, {"global_step": current, "confirmation_plan_sha256": plan_hash,
                                              "candidate": candidate, "frozen_decoder": config["decoder"]})
            save_resume(state_path, model, optimizer, step=current, plan_hash=candidate_hash, device=device)
            _evaluate_step(folder, current, path, config, decoder, perceptual, rank_paths, val_paths,
                           device, dtype, a.visual_samples)
            _restore_rng(state, device)
            _comparison(out, pair, plan)

        checkpoint_and_evaluate(step)
        for step in range(step + 1, a.steps + 1):
            offset = (step - 1) * a.accumulation
            indices = [training["indices"][j] for j in schedule[offset:offset + a.accumulation]]
            lr = a.learning_rate * min(1.0, step / max(1, a.warmup_steps))
            row = train_step(model, optimizer, packets, indices, lr=lr, device=device, dtype=dtype)
            if step == 1 or step % 25 == 0 or step == a.steps:
                row.update(step=step, sample_presentations=step * a.accumulation,
                           data_passes_equivalent=step * a.accumulation / training["views"])
                with (folder / "train_log.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[confirm {candidate['role']}] {json.dumps(row)}", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                checkpoint_and_evaluate(step)
        del optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _comparison(out, pair, plan)
    ls._verify(config)
    if ls._fingerprint(plan["input_sha256"]) != plan["input_sha256"]:
        raise ValueError("Locked selection or init checkpoint changed during confirmation")
    print(f"Confirmation complete: {out / 'comparison.json'}. Compare equal steps, then inspect custom video.", flush=True)
