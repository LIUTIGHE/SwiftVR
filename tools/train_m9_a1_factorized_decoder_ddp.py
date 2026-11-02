#!/usr/bin/env python3
"""M9-A1 matched-FLOPs factorized decoder training on cached M8-A latents.

This is intentionally a thin wrapper around the validated M8-B ReAE teacher
trainer. Data, cached z_SR, frozen original ReAE teacher, loss weights, optimizer,
DDP loop and checkpointing stay unchanged; only the student decoder architecture
is swapped to M9A1FactorizedReAEDecoder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_reae_slim_teacher_distill_ddp as trainer
from swiftvr.models.m9_factorized_decoder import (
    M9_A1_CHANNELS,
    M9_A1_GMAC_1920X1088,
    M9A1FactorizedReAEDecoder,
)


M9_VARIANT = "m9a1_factorized"


def _build_parser() -> argparse.ArgumentParser:
    parser = trainer.formal.build_parser()
    parser.description = __doc__

    found_teacher_l2 = False
    for action in parser._actions:
        if action.dest == "init_decoder":
            action.required = False
            action.default = None
            action.help = argparse.SUPPRESS
        elif action.dest == "teacher_l2_weight":
            action.default = 10.0
            found_teacher_l2 = True
    if not found_teacher_l2:
        raise RuntimeError("Parent decoder parser is missing --teacher-l2-weight")

    parser.add_argument(
        "--variant",
        choices=(M9_VARIANT,),
        default=M9_VARIANT,
    )
    parser.add_argument("--prune-calibration-samples", type=int, default=64)
    parser.add_argument("--teacher-lpips-weight", type=float, default=0.1)
    parser.add_argument("--teacher-temporal-weight", type=float, default=1.0)
    parser.add_argument(
        "--visual-validation-samples",
        type=int,
        default=0,
        help=(
            "Accepted for launch-recipe compatibility. Formal training records "
            "quantitative validation; run an isolated full-resolution visual gate "
            "after a short A1 checkpoint."
        ),
    )
    parser.set_defaults(learning_rate=3e-5)
    return parser


def main() -> int:
    # Register the new architecture into the already validated trainer without
    # changing the training objective or data path.
    trainer.VARIANT_CHANNELS[M9_VARIANT] = M9_A1_CHANNELS
    trainer.VARIANT_GMAC[M9_VARIANT] = M9_A1_GMAC_1920X1088
    trainer.SlimReAEDecoder = M9A1FactorizedReAEDecoder
    trainer.TRAINER_ID = "swiftvr_m9a1_factorized_decoder_m8latent_teacher_distill_ddp_v1"
    trainer.build_parser = _build_parser

    if "--variant" not in sys.argv and not any(
        token.startswith("--variant=") for token in sys.argv
    ):
        sys.argv.extend(["--variant", M9_VARIANT])

    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
