#!/usr/bin/env python3
"""Run heuristic and selected L20 masks with the original M8-A DDP recipe.

This is the long-budget confirmation after mask search. It intentionally reuses
the historical M8-A training teacher/protocol (D1536 TA -> Stage-A validation),
not the short Stage-A-direct search-recovery recipe.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def completed(run: Path, steps: int) -> bool:
    summary = run / "summary.json"
    if not summary.is_file():
        return False
    data = read_json(summary)
    return int(data.get("global_step", -1)) >= int(steps)


def add_repeat(command: list[str], flag: str, values) -> None:
    for value in values:
        command.extend([flag, str(value)])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection-work", type=Path,
                   default=Path("outputs/b2b/m10_layer_selection_v1"))
    p.add_argument("--reference-run", type=Path,
                   default=Path("outputs/b2b/m8a_d1024_l20_gate200k"))
    p.add_argument("--output-root", type=Path,
                   default=Path("outputs/b2b/m10_layer_fullbudget_v1"))
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument(
        "--local-batch-size",
        type=int,
        default=None,
        help="Runtime per-GPU microbatch. Defaults to the reference M8-A value when GPU count matches.",
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Runtime accumulation. Defaults to the reference M8-A value when GPU count matches.",
    )
    p.add_argument("--steps", type=int, default=60000,
                   help="Optimizer updates PER mask. 60k = 2x the mature M8-A30k checkpoint budget.")
    p.add_argument("--lr-schedule-total-steps", type=int, default=200000,
                   help="Preserve the original long-run cosine horizon while stopping at --steps.")
    p.add_argument("--validate-every", type=int, default=5000)
    p.add_argument("--save-every", type=int, default=15000)
    p.add_argument("--visualize-every", type=int, default=15000)
    p.add_argument("--num-workers", type=int, default=4,
                   help="Training DataLoader workers PER rank.")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--no-persistent-workers", action="store_true")
    p.add_argument(
        "--only",
        choices=("heuristic", "selected"),
        default=None,
        help="Run only one locked mask. Default runs heuristic then selected.",
    )
    args = p.parse_args()

    if min(args.steps, args.lr_schedule_total_steps, args.validate_every,
           args.save_every, args.visualize_every) <= 0:
        raise ValueError("Step/schedule intervals must be positive")
    if args.steps > args.lr_schedule_total_steps:
        raise ValueError("--steps cannot exceed --lr-schedule-total-steps")

    selection_path = args.selection_work.expanduser().resolve() / "selection.json"
    reference_path = args.reference_run.expanduser().resolve() / "run_config.json"
    selection = read_json(selection_path)
    reference = read_json(reference_path)

    selected = selection["selected"]
    candidates = selection["candidates"]
    heuristic = next(c for c in candidates if c["id"] == "heuristic")
    selected_record = next(c for c in candidates if c["id"] == selected["id"])
    all_pair = [("heuristic", heuristic), ("selected", selected_record)]
    pair = all_pair if args.only is None else [
        item for item in all_pair if item[0] == args.only
    ]
    if heuristic["kept_source_blocks"] == selected_record["kept_source_blocks"]:
        raise ValueError("Locked masks are identical")

    # This run is meant to reproduce the actual M8-A architecture-training protocol.
    expected = {
        "world_size": 4,
        "local_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "global_effective_batch_size": 64,
        "learning_rate": 2e-5,
        "lr_warmup_steps": 100,
        "training_teacher": "b2a_d1536_teaching_assistant",
        "training_teacher_cache_kind": "swiftvr_b2b_d1536_ta_velocity",
        "validation_teacher": "stage_a_d3072_reference",
    }
    mismatch = {k: (reference.get(k), v) for k, v in expected.items()
                if reference.get(k) != v}
    if mismatch:
        raise ValueError(
            "Reference M8-A run_config does not match the known long-run protocol: "
            + json.dumps(mismatch, indent=2)
        )
    if int(reference["global_effective_batch_size"]) != (
        int(reference["world_size"])
        * int(reference["local_batch_size"])
        * int(reference["gradient_accumulation_steps"])
    ):
        raise ValueError("Reference global batch accounting is inconsistent")
    if not reference.get("manifests") or not reference.get("val_manifests"):
        raise ValueError("Reference run_config is missing train/validation manifests")

    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus must contain one or more distinct GPU IDs")
    world_size = len(gpu_ids)
    if world_size == int(reference["world_size"]):
        local_batch_size = (
            int(reference["local_batch_size"])
            if args.local_batch_size is None
            else int(args.local_batch_size)
        )
        accumulation_steps = (
            int(reference["gradient_accumulation_steps"])
            if args.gradient_accumulation_steps is None
            else int(args.gradient_accumulation_steps)
        )
    else:
        if args.local_batch_size is None or args.gradient_accumulation_steps is None:
            raise ValueError(
                "Changing GPU count requires explicit --local-batch-size and "
                "--gradient-accumulation-steps. For GPUs 4,5,6, first try "
                "--local-batch-size 21 --gradient-accumulation-steps 1 "
                "(global batch 63; best throughput if memory fits). If that OOMs, "
                "fall back to --local-batch-size 11 --gradient-accumulation-steps 2 "
                "(global batch 66)."
            )
        local_batch_size = int(args.local_batch_size)
        accumulation_steps = int(args.gradient_accumulation_steps)
    if local_batch_size <= 0 or accumulation_steps <= 0:
        raise ValueError("Runtime batch/accumulation must be positive")
    if args.num_workers < 0 or args.prefetch_factor <= 0:
        raise ValueError("DataLoader worker count must be non-negative and prefetch positive")
    runtime_global_batch = world_size * local_batch_size * accumulation_steps
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    plan = {
        "kind": "m10_layer_masks_fullbudget_m8a_protocol_v1",
        "selection": str(selection_path),
        "reference_run_config": str(reference_path),
        "physical_gpus": gpu_ids,
        "runtime_world_size": world_size,
        "runtime_local_batch_size": local_batch_size,
        "runtime_gradient_accumulation_steps": accumulation_steps,
        "runtime_global_effective_batch_size": runtime_global_batch,
        "num_workers_per_rank": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": bool(args.num_workers > 0 and not args.no_persistent_workers),
        "reference_global_effective_batch_size": int(reference["global_effective_batch_size"]),
        "global_batch_ratio_to_reference": runtime_global_batch / int(reference["global_effective_batch_size"]),
        "per_mask_steps": args.steps,
        "mature_m8a_reference_step": 30000,
        "lr_schedule_total_steps": args.lr_schedule_total_steps,
        "reference_protocol": expected,
        "teacher_cache": reference["teacher_cache"],
        "val_teacher_cache": reference["val_teacher_cache"],
        "masks": {
            role: {
                "id": record["id"],
                "init_checkpoint": record["init_checkpoint"],
                "kept_source_blocks": record["kept_source_blocks"],
            }
            for role, record in all_pair
        },
        "note": (
            "Heuristic and selected use the same locked M8-A teacher/loss/LR protocol. "
            "When runtime_global_effective_batch_size differs from the reference 64, "
            "comparison to historical M8-A is approximate; strict mask comparison "
            "requires both masks to use the same runtime batch. LR follows the "
            "original 200k horizon."
        ),
    }
    plan_path = output_root / "longrun_plan.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError(f"Existing plan differs: {plan_path}")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env.setdefault("OMP_NUM_THREADS", "1")

    print(
        f"[runtime] GPUs={gpu_ids} world={world_size} "
        f"local_batch={local_batch_size} accum={accumulation_steps} "
        f"global_batch={runtime_global_batch} "
        f"(reference={reference['global_effective_batch_size']})",
        flush=True,
    )

    for role, record in pair:
        run = output_root / role
        if completed(run, args.steps):
            print(f"[skip] {role} already completed >= {args.steps} steps: {run}", flush=True)
            continue
        if run.exists() and any(run.iterdir()):
            raise FileExistsError(
                f"Incomplete/nonempty run exists: {run}. Do not mix or overwrite long-run results."
            )

        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc_per_node={world_size}",
            str(ROOT / "tools/train_b2b_moe_ta_distill_ddp_cache_safe.py"),
            "--architecture", "m8-d1024-l20",
            "--lr-schedule-total-steps", str(args.lr_schedule_total_steps),
            "--base-checkpoint", reference["base_checkpoint"],
            "--student-init", str(Path(record["init_checkpoint"]).expanduser().resolve()),
            "--teacher-cache", reference["teacher_cache"],
            "--val-teacher-cache", reference["val_teacher_cache"],
            "--path-root", ".",
            "--clip-length", str(reference["clip_length"]),
            "--crop-size", str(reference["crop_size"]),
            "--scale", str(reference["scale"]),
            "--views-per-record", str(reference["views_per_record"]),
            "--view-seed", str(reference["view_seed"]),
            "--batch-size", str(local_batch_size),
            "--gradient-accumulation-steps", str(accumulation_steps),
            "--expected-global-batch-size", str(runtime_global_batch),
            "--learning-rate", str(reference["learning_rate"]),
            "--lr-warmup-steps", str(reference["lr_warmup_steps"]),
            "--min-lr-ratio", str(reference["min_lr_ratio"]),
            "--velocity-mse-weight", str(reference["velocity_mse_weight"]),
            "--velocity-cosine-weight", str(reference["velocity_cosine_weight"]),
            "--router-balance-weight", str(reference["router_balance_weight"]),
            "--dtype", reference["runtime_dtype"],
            "--attention-backend", "sdpa",
            "--max-steps", str(args.steps),
            "--validate-every", str(args.validate_every),
            "--save-every", str(args.save_every),
            "--visualize-every", str(args.visualize_every),
            "--visual-validation-samples", "13",
            "--visual-frame-indices", "0,1,2,3,4,5,6,7,8,9,10,11,12",
            "--visual-video-fps", "30",
            "--validate-at-start",
            "--pin-memory",
            "--num-workers", str(args.num_workers),
            "--prefetch-factor", str(args.prefetch_factor),
            "--seed", "0",
            "--no-tensorboard",
            "--output-dir", str(run),
        ]
        if args.num_workers > 0 and not args.no_persistent_workers:
            cmd.append("--persistent-workers")
        add_repeat(cmd, "--manifest", reference["manifests"])
        add_repeat(cmd, "--val-manifest", reference["val_manifests"])

        print("\n[launch]", role, flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    print(f"Both long runs completed. Results: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
