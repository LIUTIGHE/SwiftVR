#!/usr/bin/env python3
"""Report whether Moderate v1 actually uses its newly widened capacity.

The function-preserving initializer gates new features with exact-zero downstream
weights.  During recovery training those gates should move away from zero if the
optimizer uses the extra condition / CompactMemBlock capacity.  This diagnostic
reports RMS/max norms for those formerly-zero slices and compares them with the
corresponding inherited slices.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.models.tiny_conditional_decoder_moderate_resize_conv import (
    ModerateResizeConvTinyConditionalDecoder,
)
from swiftvr.models.tiny_conditional_decoder_resize_conv import ResizeConvTinyConditionalDecoder
from swiftvr.models.tiny_decoder_sparsity import CompactMemBlock


def _stats(value: torch.Tensor) -> dict[str, float | int]:
    x = value.detach().float().reshape(-1)
    if x.numel() == 0:
        return {"elements": 0, "rms": 0.0, "mean_abs": 0.0, "max_abs": 0.0}
    return {
        "elements": int(x.numel()),
        "rms": float(math.sqrt(float(x.square().mean().item()))),
        "mean_abs": float(x.abs().mean().item()),
        "max_abs": float(x.abs().max().item()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-decoder", type=Path, required=True)
    parser.add_argument("--moderate-decoder", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = ResizeConvTinyConditionalDecoder.from_pretrained(args.source_decoder)
    moderate = ModerateResizeConvTinyConditionalDecoder.from_pretrained(args.moderate_decoder)

    if source.channels != moderate.channels or source.blocks_per_stage != moderate.blocks_per_stage:
        raise ValueError("Source/moderate outer topology mismatch")
    if moderate.condition_channels < source.condition_channels:
        raise ValueError("Moderate condition width is not wider than source")

    latent = int(source.latent_channels)
    old_cond = int(source.condition_channels)
    report: dict[str, object] = {
        "source_decoder": str(args.source_decoder.expanduser().resolve()),
        "moderate_decoder": str(args.moderate_decoder.expanduser().resolve()),
        "condition": {
            "source_channels": old_cond,
            "target_channels": int(moderate.condition_channels),
            "inherited_input_columns": _stats(
                moderate.decoder[1].weight[:, latent : latent + old_cond]
            ),
            "new_input_columns": _stats(
                moderate.decoder[1].weight[:, latent + old_cond :]
            ),
            "new_projection_rows": _stats(
                moderate.condition_projection.weight[old_cond:]
            ),
        },
        "compact_blocks": [],
    }

    source_blocks = [
        (name, module)
        for name, module in source.named_modules()
        if isinstance(module, CompactMemBlock)
    ]
    moderate_modules = dict(moderate.named_modules())
    for name, src in source_blocks:
        dst = moderate_modules.get(name)
        if not isinstance(dst, CompactMemBlock):
            raise TypeError(f"Moderate block {name!r} is missing")
        old_k = int(src.internal_channels)
        new_k = int(dst.internal_channels)
        payload = {
            "module": name,
            "interface_channels": int(src.interface_channels),
            "source_internal_channels": old_k,
            "target_internal_channels": new_k,
            "inherited_output_columns": _stats(dst.conv[4].weight[:, :old_k]),
            "new_output_columns": _stats(dst.conv[4].weight[:, old_k:]),
            "new_first_conv_rows": _stats(dst.conv[0].weight[old_k:]),
        }
        report["compact_blocks"].append(payload)  # type: ignore[union-attr]

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
