#!/usr/bin/env python3
"""Teacher-only D768 DiT distillation with the original frozen ReAE decoder.

This is the clean B2B DiT-compression baseline.  Decoder compression is removed
from the optimization problem entirely:

    LQ -> frozen original ReAE encoder -> trainable D768/F4080/L30 DiT
       -> (validation only) frozen original ReAE decoder -> RGB

Training is decoder-free and GT-free.  The student matches the cached Stage-A
teacher endpoint velocity using normalized MSE + cosine loss.  The original ReAE
decoder is used only for validation/visualization of Student vs Teacher behavior;
GT metrics are diagnostic only and never contribute gradients.

The implementation deliberately reuses the validated B2-A DDP trainer.  This
wrapper only locks the D768 student shape and records the teacher-behavior
protocol so future decoder experiments cannot silently contaminate this baseline.
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


D768_SHAPE = {
    "student_hidden_dim": 768,
    "student_num_heads": 6,
    "student_head_dim": 128,
    "student_ffn_dim": 4080,
    "student_num_layers": 30,
    "student_adapter_dim": 128,
}

_original_build_parser = base.build_parser
_original_write_json = base._write_json


def build_parser():
    parser = _original_build_parser()
    parser.description = __doc__
    parser.set_defaults(**D768_SHAPE)
    return parser


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    payload = dict(value)
    if path.name == "run_config.json":
        payload["experiment"] = "b2b_d768_full_decoder_teacher_velocity_v1"
        payload["deployment_priority"] = "teacher_behavior"
        payload["training_decoder"] = "none"
        payload["validation_decoder"] = "original_frozen_reae"
        payload["gt_role"] = "diagnostic_only"
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
    # Patch only in this standalone wrapper process. Importing the module leaves
    # the locked B2-A trainer unchanged.
    base.build_parser = build_parser
    base._write_json = _write_json
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
