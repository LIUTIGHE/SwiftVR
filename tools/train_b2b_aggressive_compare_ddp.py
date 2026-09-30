#!/usr/bin/env python3
"""Paired B2B comparison with the trained aggressive SlimReAE decoder.

This wrapper intentionally reuses the validated B2B-1B DDP trainer so the staged
and joint experiments differ only in whether decoder parameters are updated.
Both branches use the same D768/F4080/L30 DiT, the same aggressive decoder
(256,128,64,32), the same losses/data/validation path, and the same DDP code.

Deployment priority is teacher-behavior preservation rather than GT-regression.
Student->Teacher RGB fidelity remains the checkpoint-selection criterion; GT
metrics are diagnostics, and any non-zero GT loss weight should be treated as an
explicit diagnostic/training choice rather than the default long-run objective.

``--freeze-decoder`` implements the staged branch by forcing the decoder optimizer
learning rate to exactly zero after argument validation. Decoder gradients are
still computed so the validated DDP/autograd path is unchanged, but decoder
parameters cannot update. This is deliberate: it isolates update policy without
introducing a second training implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

# Direct execution (``python tools/...`` or ``torchrun tools/...``) places the
# tools directory rather than the repository root on sys.path. Add the root
# before importing the top-level ``tools`` package, matching the other standalone
# SwiftVR training scripts.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_b2b_joint_recovery_ddp as base
from swiftvr.models.reae_slim_decoder import AGGRESSIVE_CHANNELS
from swiftvr.training.b2b_joint import estimate_b2b_dit_gmac_per_output_frame


AGGRESSIVE_DECODER_GMAC_PER_FRAME = 86.79211008
FROZEN_REAE_ENCODER_GMAC_PER_FRAME = 42.278
_original_build_parser = base.build_parser
_original_validate_args = base._validate_args
_original_write_json = base._write_json
_freeze_decoder_updates = False


def b2b_aggressive_compute_budget() -> dict[str, float]:
    dit = float(estimate_b2b_dit_gmac_per_output_frame())
    decoder = float(AGGRESSIVE_DECODER_GMAC_PER_FRAME)
    encoder = float(FROZEN_REAE_ENCODER_GMAC_PER_FRAME)
    combined = dit + decoder
    whole = encoder + combined
    return {
        "dit_gmac_per_frame": dit,
        "decoder_gmac_per_frame": decoder,
        "dit_plus_decoder_gmac_per_frame": combined,
        "encoder_gmac_per_frame": encoder,
        "whole_model_gmac_per_frame": whole,
        "whole_model_gflops_per_frame": 2.0 * whole,
    }


def build_parser():
    parser = _original_build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--freeze-decoder",
        action="store_true",
        help=(
            "Staged comparison: keep the pretrained aggressive decoder fixed by "
            "forcing its optimizer learning rate to zero."
        ),
    )
    return parser


def _validate_args(args) -> None:
    global _freeze_decoder_updates
    _original_validate_args(args)
    _freeze_decoder_updates = bool(args.freeze_decoder)
    if _freeze_decoder_updates:
        args.decoder_learning_rate = 0.0


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    payload = dict(value)
    if path.name == "run_config.json":
        payload["experiment"] = "b2b_aggressive_decoder_staged_vs_joint_v1"
        payload["decoder_variant"] = "aggressive"
        payload["decoder_channels"] = list(AGGRESSIVE_CHANNELS)
        payload["decoder_update_mode"] = (
            "frozen_zero_lr" if _freeze_decoder_updates else "joint_trainable"
        )
        payload["freeze_decoder"] = _freeze_decoder_updates
        payload["deployment_priority"] = "teacher_behavior"
        payload["checkpoint_selection_metric"] = "student_teacher_psnr"
        payload["gt_role"] = "diagnostic_or_explicit_optional_loss"
    _original_write_json(path, payload)


def main() -> int:
    # Patch only while this wrapper is executed. Merely importing this module for
    # tests or tooling leaves the validated B2B-1B module untouched.
    base.build_parser = build_parser
    base._validate_args = _validate_args
    base._write_json = _write_json
    base.B2B_EXTREME_DECODER_CHANNELS = tuple(AGGRESSIVE_CHANNELS)
    base.b2b_compute_budget = b2b_aggressive_compute_budget
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
