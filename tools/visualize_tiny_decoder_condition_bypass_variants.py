#!/usr/bin/env python3
"""Existing Tiny-decoder visualizer with F1b condition-bypass checkpoint support."""

from __future__ import annotations

from tools import visualize_tiny_decoder_variants as base
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
