#!/usr/bin/env python3
"""Existing phase-bias diagnostic with F1b condition-bypass checkpoint support."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import diagnose_tiny_decoder_variant_phase_bias as base
from swiftvr.models.tiny_conditional_decoder_condition_bypass_resize_conv import (
    ConditionBypassResizeConvTinyConditionalDecoder,
)


base.SUPPORTED_DECODER_CLASSES = {
    **base.SUPPORTED_DECODER_CLASSES,
    ConditionBypassResizeConvTinyConditionalDecoder.__name__: (
        ConditionBypassResizeConvTinyConditionalDecoder
    ),
}


if __name__ == "__main__":
    raise SystemExit(base.main())
