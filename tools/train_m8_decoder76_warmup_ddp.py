#!/usr/bin/env python3
"""M8-B Decoder76 teacher-only warm-up on cached M8-A restored latents.

This is intentionally a thin wrapper around the validated structured ReAE
teacher-distillation trainer.  It registers the M8 hardware-oriented
(128,96,64,64) decoder point while preserving the existing DDP loop, activation-
RMS structured initialization, frozen full-ReAE teacher, LPIPS/temporal losses,
and checkpoint format.

Important: --train-cache/--val-cache must contain z_SR produced by the selected
M8-A D1024/L20 checkpoint, not Stage-A z_SR.  The frozen original ReAE and the
Decoder76 student therefore receive exactly the same M8 latent.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_reae_slim_teacher_distill_ddp as trainer


M8_VARIANT = "m8decoder76"
M8_DECODER_GMAC_PER_FRAME = 76.45175808


def main() -> int:
    # Keep the validated trainer implementation untouched; register only the new
    # architecture/accounting identity before its parser/fingerprint are built.
    trainer.VARIANT_GMAC[M8_VARIANT] = M8_DECODER_GMAC_PER_FRAME
    trainer.TRAINER_ID = "swiftvr_m8b_decoder76_m8latent_teacher_distill_ddp_v1"

    if "--variant" not in sys.argv and not any(
        token.startswith("--variant=") for token in sys.argv
    ):
        sys.argv.extend(["--variant", M8_VARIANT])
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
