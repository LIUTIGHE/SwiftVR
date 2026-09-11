#!/usr/bin/env python3
"""Phase-bias diagnostic with support for the F1 direct-condition decoder.

This wrapper leaves the existing variant diagnostic unchanged and only extends
its checkpoint-class registry for DirectConditionResizeConvTinyConditionalDecoder.
All metrics, validation sampling, phase definitions, and output formats are
therefore identical to diagnose_tiny_decoder_variant_phase_bias.py.
"""

from __future__ import annotations

from tools import diagnose_tiny_decoder_variant_phase_bias as base
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
