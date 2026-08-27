#!/usr/bin/env python3
"""B2B-0C multi-sample teacher-latent generalization gate for the extreme decoder.

This is intentionally a thin wrapper around the already validated
``train_reae_slim_teacher_distill_ddp.py`` path. It does not modify the legacy
B1 variant registry on disk. At process startup it registers exactly one B2B
variant:

    extreme = (96, 48, 24, 16)  # 13.35785472 GMAC / 1080p output frame

The underlying trainer still performs activation-RMS structured channel
selection from the frozen full ReAE decoder, trains on cached Stage-A z_SR
latents, evaluates on the fixed validation cache, and reports both
student->teacher and student->GT metrics. For the first B2B-0C gate we
recommend passing teacher-L2=1, teacher-LPIPS=0 and teacher-temporal=0 so the
experiment answers the same question as B2B-0B at dataset scale: can the
13.36-GMAC decoder generalize teacher RGB rendering from correct teacher
latents?
"""

from __future__ import annotations

from tools import train_reae_slim_teacher_distill_ddp as base


EXTREME_CHANNELS = (96, 48, 24, 16)
EXTREME_GMAC_PER_1080P_FRAME = 13.35785472

# Register only inside this process. ``base.VARIANT_CHANNELS`` is the imported
# dictionary object used by the base parser, initialization path and fingerprint.
base.VARIANT_CHANNELS["extreme"] = EXTREME_CHANNELS
base.VARIANT_GMAC["extreme"] = EXTREME_GMAC_PER_1080P_FRAME
base.TRAINER_ID = "swiftvr_b2b0c_extreme_decoder_teacher_distill_ddp_v1"


if __name__ == "__main__":
    raise SystemExit(base.main())
