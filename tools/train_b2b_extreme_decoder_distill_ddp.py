#!/usr/bin/env python3
"""B2B-0C multi-sample teacher-latent generalization gate for the extreme decoder.

This is intentionally a thin wrapper around the already validated
``train_reae_slim_teacher_distill_ddp.py`` training path. It does not modify the
legacy B1 variant registry on disk. At process startup it registers exactly one
B2B variant:

    extreme = (96, 48, 24, 16)  # 13.35785472 GMAC / 1080p output frame

The wrapper also supplies a compatibility parser. The current shared formal
parser already defines ``--teacher-l2-weight`` while the historical B1 slim
trainer attempts to add it again; calling the historical ``build_parser``
directly therefore raises an argparse option conflict. We avoid changing the
locked B1 trainer by constructing the same effective parser here and adding
B1-specific arguments only when they are not already present.
"""

from __future__ import annotations

import argparse

from tools import train_reae_slim_teacher_distill_ddp as base


EXTREME_CHANNELS = (96, 48, 24, 16)
EXTREME_GMAC_PER_1080P_FRAME = 13.35785472

# Register only inside this process. ``base.VARIANT_CHANNELS`` is the imported
# dictionary object used by the initialization path, fingerprint and trainer.
base.VARIANT_CHANNELS["extreme"] = EXTREME_CHANNELS
base.VARIANT_GMAC["extreme"] = EXTREME_GMAC_PER_1080P_FRAME
base.TRAINER_ID = "swiftvr_b2b0c_extreme_decoder_teacher_distill_ddp_v1"


def _option_exists(parser: argparse.ArgumentParser, option: str) -> bool:
    return option in parser._option_string_actions


def build_parser() -> argparse.ArgumentParser:
    """Build the B2B-0C CLI without the historical duplicate-option conflict."""

    parser = base.formal.build_parser()
    parser.description = __doc__

    # B1 slim training never uses --init-decoder because the structured student
    # is materialized from the frozen ReAE teacher (or restored from --resume).
    for action in parser._actions:
        if action.dest == "init_decoder":
            action.required = False
            action.default = None
            action.help = argparse.SUPPRESS

    if not _option_exists(parser, "--variant"):
        parser.add_argument("--variant", choices=tuple(base.VARIANT_CHANNELS), required=True)
    if not _option_exists(parser, "--prune-calibration-samples"):
        parser.add_argument("--prune-calibration-samples", type=int, default=64)

    # ``formal.build_parser`` already owns --teacher-l2-weight in the current
    # codebase. Preserve that action and only adjust its default to the B1 value.
    teacher_l2_action = parser._option_string_actions.get("--teacher-l2-weight")
    if teacher_l2_action is None:
        parser.add_argument("--teacher-l2-weight", type=float, default=10.0)
    else:
        teacher_l2_action.default = 10.0
        parser.set_defaults(teacher_l2_weight=10.0)

    if not _option_exists(parser, "--teacher-lpips-weight"):
        parser.add_argument("--teacher-lpips-weight", type=float, default=0.1)
    if not _option_exists(parser, "--teacher-temporal-weight"):
        parser.add_argument("--teacher-temporal-weight", type=float, default=1.0)

    parser.set_defaults(learning_rate=3e-5)
    return parser


# ``base.main`` resolves ``build_parser`` from its own module globals at runtime.
# Override only in this wrapper process; the source B1 module remains unchanged.
base.build_parser = build_parser


if __name__ == "__main__":
    raise SystemExit(base.main())
