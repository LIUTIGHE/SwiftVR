#!/usr/bin/env python3
"""D768 SwiftVR DiT distillation with LIFT coarse-to-fine teacher guidance.

This experiment is paired with ``train_b2b_dit_full_decoder_distill_ddp.py``.
Architecture, data, optimizer, validation, and the original frozen ReAE decoder
are identical.  The only training change is an additional LIFT coarse-to-fine
loss on the cached Stage-A endpoint velocity.

No GT, compressed decoder, GAN, LPIPS, or feature KD is used in this first LIFT
ablation.  PLACE is intentionally deferred until global LIFT is validated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
for search_root in (ROOT, TOOLS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from tools import train_b2a_compact_distill_ddp as base
from swiftvr.training.lift_distillation import lift_velocity_distillation_objective


D768_SHAPE = {
    "student_hidden_dim": 768,
    "student_num_heads": 6,
    "student_head_dim": 128,
    "student_ffn_dim": 4080,
    "student_num_layers": 30,
    "student_adapter_dim": 128,
}

_original_build_parser = base.build_parser
_original_validate_args = base._validate_args
_original_write_json = base._write_json
_lift_weight = 1.0


def build_parser():
    parser = _original_build_parser()
    parser.description = __doc__
    parser.set_defaults(**D768_SHAPE)
    parser.add_argument(
        "--lift-weight",
        type=float,
        default=1.0,
        help="Weight of LIFT coarse-to-fine endpoint-velocity distillation.",
    )
    return parser


def _validate_args(args):
    global _lift_weight
    result = _original_validate_args(args)
    if float(args.lift_weight) < 0:
        raise ValueError("--lift-weight must be non-negative")
    _lift_weight = float(args.lift_weight)
    return result


def _lift_objective(
    student_velocity,
    teacher_velocity,
    *,
    velocity_mse_weight=1.0,
    velocity_cosine_weight=1.0,
    output_l1_weight=0.0,
    output_temporal_weight=0.0,
    gt_loss_mode="none",
    gt_pixel_weight=0.0,
    gt_temporal_weight=0.0,
    epsilon=1e-8,
    **_unused,
):
    if any(
        float(value) != 0.0
        for value in (output_l1_weight, output_temporal_weight, gt_pixel_weight, gt_temporal_weight)
    ) or str(gt_loss_mode).lower() != "none":
        raise ValueError("B2B D768 LIFT experiment is strictly teacher-velocity-only")
    return lift_velocity_distillation_objective(
        student_velocity,
        teacher_velocity,
        velocity_mse_weight=velocity_mse_weight,
        velocity_cosine_weight=velocity_cosine_weight,
        lift_weight=_lift_weight,
        epsilon=epsilon,
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    payload = dict(value)
    if path.name == "run_config.json":
        payload["experiment"] = "b2b_d768_full_decoder_lift_v1"
        payload["deployment_priority"] = "teacher_behavior"
        payload["training_decoder"] = "none"
        payload["validation_decoder"] = "original_frozen_reae"
        payload["gt_role"] = "diagnostic_only"
        payload["distillation_method"] = "direct_velocity_plus_lift"
        payload["lift_weight"] = _lift_weight
        payload["place_enabled"] = False
        payload["checkpoint_selection_metric"] = "velocity_relative_l2"
        payload["locked_student_shape"] = {
            "hidden_dim": 768,
            "num_heads": 6,
            "head_dim": 128,
            "ffn_dim": 4080,
            "num_layers": 30,
            "adapter_dim": 128,
        }
    _original_write_json(path, payload)


def main() -> int:
    base.build_parser = build_parser
    base._validate_args = _validate_args
    base._write_json = _write_json
    base.velocity_distillation_objective = _lift_objective
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
