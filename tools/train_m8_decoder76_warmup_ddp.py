#!/usr/bin/env python3
"""M8-B Decoder76 teacher-only warm-up on cached M8-A restored latents.

This is intentionally a thin wrapper around the validated structured ReAE
teacher-distillation trainer. It registers the M8 hardware-oriented
(128,96,64,64) decoder point while preserving the existing DDP loop, activation-
RMS structured initialization, frozen full-ReAE teacher, LPIPS/temporal losses,
and checkpoint format.

Important: --train-cache/--val-cache must contain z_SR produced by the selected
M8-A D1024/L20 checkpoint, not Stage-A z_SR. The frozen original ReAE and the
Decoder76 student therefore receive exactly the same M8 latent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_reae_slim_teacher_distill_ddp as trainer


M8_VARIANT = "m8decoder76"
M8_DECODER_GMAC_PER_FRAME = 76.45175808


def _build_m8_parser() -> argparse.ArgumentParser:
    """Compose the M8 parser without re-registering inherited options.

    ``train_tiny_decoder_formal_ddp.build_parser`` already owns
    ``--teacher-l2-weight``. The generic ReAE-slim trainer historically adds the
    same option again, which raises an argparse conflict before training starts.
    M8-B therefore builds the parser directly from the formal parent, changes the
    inherited teacher-L2 default in place, and only adds options that are truly
    new to the ReAE-slim path.
    """

    parser = trainer.formal.build_parser()
    parser.description = __doc__

    found_teacher_l2 = False
    for action in parser._actions:
        if action.dest == "init_decoder":
            # Structured initialization comes from the frozen original ReAE.
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
        choices=tuple(trainer.VARIANT_CHANNELS),
        required=True,
    )
    parser.add_argument("--prune-calibration-samples", type=int, default=64)
    parser.add_argument("--teacher-lpips-weight", type=float, default=0.1)
    parser.add_argument("--teacher-temporal-weight", type=float, default=1.0)
    parser.add_argument(
        "--visual-validation-samples",
        type=int,
        default=0,
        help=(
            "Compatibility knob for the M8-B launch recipe. The validated "
            "ReAE-slim trainer records quantitative validation only; use the "
            "post-warm-up isolated M8 Original-vs-Decoder76 visualization for "
            "full-resolution qualitative review."
        ),
    )
    parser.set_defaults(learning_rate=3e-5)
    return parser


def main() -> int:
    # Keep the validated training implementation untouched. Only register the new
    # architecture/accounting identity and replace its broken parser composition
    # for this M8-specific entry point.
    trainer.VARIANT_GMAC[M8_VARIANT] = M8_DECODER_GMAC_PER_FRAME
    trainer.TRAINER_ID = "swiftvr_m8b_decoder76_m8latent_teacher_distill_ddp_v1"
    trainer.build_parser = _build_m8_parser

    if "--variant" not in sys.argv and not any(
        token.startswith("--variant=") for token in sys.argv
    ):
        sys.argv.extend(["--variant", M8_VARIANT])

    visual_count = 0
    for index, token in enumerate(sys.argv):
        if token == "--visual-validation-samples" and index + 1 < len(sys.argv):
            try:
                visual_count = int(sys.argv[index + 1])
            except ValueError:
                pass
        elif token.startswith("--visual-validation-samples="):
            try:
                visual_count = int(token.split("=", 1)[1])
            except ValueError:
                pass
    if visual_count > 0:
        print(
            "[M8-B] --visual-validation-samples is accepted for launch "
            "compatibility, but this validated warm-up path exports metrics only. "
            "Run the isolated Original-Decoder vs Decoder76 visual gate after "
            "warm-up.",
            flush=True,
        )

    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
