#!/usr/bin/env python3
"""Select an M8-C checkpoint without sacrificing the Stage-A velocity anchor.

The core joint trainer reports a plain joint-loss best.  For M8-C we instead use
a constrained rule:

1. take step-0 Stage-A velocity relative-L2 as the behavior baseline;
2. keep only saved checkpoints whose relative-L2 is no worse than
   ``baseline + max_drift``;
3. among those safe checkpoints, choose the lowest joint teacher validation loss.

GT metrics remain diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-drift", type=float, default=0.005)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--output-name", default="safe_best.json")
    return parser


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            values.append(value)
    if not values:
        raise ValueError(f"empty validation log: {path}")
    return values


def _metric(record: Mapping[str, object], name: str) -> float:
    if name not in record:
        raise KeyError(f"validation record lacks {name!r}: step={record.get('global_step')}")
    return float(record[name])


def select_safe_checkpoint(
    run_dir: Path,
    *,
    max_drift: float,
    checkpoint_every: int,
) -> dict[str, object]:
    if max_drift < 0:
        raise ValueError("max_drift must be non-negative")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")

    run_dir = run_dir.expanduser().resolve()
    records = _read_jsonl(run_dir / "val_log.jsonl")
    step0 = next((item for item in records if int(item.get("global_step", -1)) == 0), None)
    if step0 is None:
        raise ValueError("M8-C validation log has no step-0 baseline")
    baseline = _metric(step0, "velocity_relative_l2")
    maximum = baseline + float(max_drift)

    candidates: list[dict[str, object]] = []
    for item in records:
        step = int(item.get("global_step", -1))
        if step <= 0 or step % checkpoint_every != 0:
            continue
        checkpoint = run_dir / "checkpoints" / f"step_{step:08d}"
        if not checkpoint.is_dir():
            continue
        relative_l2 = _metric(item, "velocity_relative_l2")
        if relative_l2 > maximum:
            continue
        candidates.append(
            {
                "global_step": step,
                "checkpoint": checkpoint.relative_to(run_dir).as_posix(),
                "joint_teacher_validation_loss": _metric(item, "loss"),
                "velocity_relative_l2": relative_l2,
                "velocity_relative_l2_drift": relative_l2 - baseline,
                "velocity_cosine": _metric(item, "velocity_cosine"),
                "student_teacher_psnr": _metric(item, "student_teacher_psnr"),
                "student_teacher_ssim": _metric(item, "student_teacher_ssim"),
                "student_gt_psnr": _metric(item, "student_gt_psnr"),
                "student_gt_ssim": _metric(item, "student_gt_ssim"),
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item["joint_teacher_validation_loss"]),
            -float(item["student_teacher_psnr"]),
        )
    )
    return {
        "status": "PASS" if candidates else "NO_SAFE_CHECKPOINT",
        "selection_rule": "min_joint_teacher_val_loss_subject_to_stage_a_velocity_rel_l2_guard",
        "baseline_step": 0,
        "baseline_velocity_relative_l2": baseline,
        "max_velocity_relative_l2_drift": float(max_drift),
        "max_allowed_velocity_relative_l2": maximum,
        "checkpoint_every": int(checkpoint_every),
        "selected": candidates[0] if candidates else None,
        "safe_candidates": candidates,
        "gt_role": "diagnostic_only",
    }


def main() -> int:
    args = build_parser().parse_args()
    result = select_safe_checkpoint(
        args.run_dir,
        max_drift=args.max_drift,
        checkpoint_every=args.checkpoint_every,
    )
    output = args.run_dir.expanduser().resolve() / args.output_name
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
