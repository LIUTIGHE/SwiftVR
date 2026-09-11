#!/usr/bin/env python3
"""Existing Tiny-decoder visualizer with Moderate v1 checkpoint support."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import visualize_tiny_decoder_variants as base
from swiftvr.models.tiny_conditional_decoder_moderate_resize_conv import (
    ModerateResizeConvTinyConditionalDecoder,
)

base.SUPPORTED_DECODER_CLASSES = {
    **base.SUPPORTED_DECODER_CLASSES,
    ModerateResizeConvTinyConditionalDecoder.__name__: ModerateResizeConvTinyConditionalDecoder,
}

if __name__ == "__main__":
    raise SystemExit(base.main())
