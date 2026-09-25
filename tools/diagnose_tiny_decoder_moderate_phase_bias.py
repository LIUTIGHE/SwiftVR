#!/usr/bin/env python3
"""Existing Tiny-decoder phase diagnostic with Moderate v1 checkpoint support."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import diagnose_tiny_decoder_variant_phase_bias as base
from swiftvr.models.tiny_conditional_decoder_moderate_resize_conv import (
    ModerateResizeConvTinyConditionalDecoder,
)

base.SUPPORTED_DECODER_CLASSES = {
    **base.SUPPORTED_DECODER_CLASSES,
    ModerateResizeConvTinyConditionalDecoder.__name__: ModerateResizeConvTinyConditionalDecoder,
}

if __name__ == "__main__":
    raise SystemExit(base.main())
