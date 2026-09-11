#!/usr/bin/env python3
"""Visualize Stage-B1 variants including the F1 direct-condition decoder.

This wrapper leaves visualize_tiny_decoder_variants.py unchanged and only extends
its checkpoint-class registry for DirectConditionResizeConvTinyConditionalDecoder.
Sampling, metrics, comparison frames, and videos are otherwise identical.
"""

from __future__ import annotations

from tools import visualize_tiny_decoder_variants as base
from swiftvr.models.tiny_conditional_decoder_direct_condition_resize_conv import (
    DirectConditionResizeConvTinyConditionalDecoder,
)


base.SUPPORTED_DECODER_CLASSES = {
    **base.SUPPORTED_DECODER_CLASSES,
    DirectConditionResizeConvTinyConditionalDecoder.__name__: (
        DirectConditionResizeConvTinyConditionalDecoder
    ),
}


if __name__ == "__main__":
    raise SystemExit(base.main())
