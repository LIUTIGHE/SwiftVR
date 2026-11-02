#!/usr/bin/env python3
"""Analytic MAC profiler for ResizeConv Tiny / Moderate TC decoders.

The counting convention matches the existing SwiftVR decoder accounting:

* one multiply-accumulate is one MAC;
* temporal compression is 4x at S0/S1, 2x at S2, and full-rate at S3/output;
* spatial stages are /16, /8, /4, /2, then full-resolution RGB;
* CompactMemBlock is 2C->K->K->C with three 3x3 convolutions;
* SwiftVR TGrow(stride=1) is one C->C 1x1 conv;
* SwiftVR TGrow(stride=2) is one dense C->C temporal 3x1x1 conv at the
  post-upsample temporal rate;
* nearest-neighbor resize itself contributes zero MACs.

At 1920x1088 this reproduces the locked keep040 decoder value
47.94476544 GMAC/output-frame and predicts Moderate v1 at 66.54799872 GMAC/frame.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence


def _positive_tuple(values: Sequence[int], *, name: str, length: int = 4) -> tuple[int, ...]:
    result = tuple(int(v) for v in values)
    if len(result) != length or any(v <= 0 for v in result):
        raise ValueError(f"{name} must contain {length} positive integers, got {result}")
    return result


def estimate_resizeconv_decoder_macs(
    *,
    output_height: int,
    output_width: int,
    latent_channels: int,
    condition_channels: int,
    channels: Sequence[int],
    internal_channels: Sequence[int],
    blocks_per_stage: Sequence[int],
    packed_condition_channels: int = 3072,
) -> dict[str, object]:
    height = int(output_height)
    width = int(output_width)
    latent_channels = int(latent_channels)
    condition_channels = int(condition_channels)
    packed_condition_channels = int(packed_condition_channels)
    channels = _positive_tuple(channels, name="channels")
    internal_channels = _positive_tuple(internal_channels, name="internal_channels")
    blocks_per_stage = _positive_tuple(blocks_per_stage, name="blocks_per_stage")
    if min(height, width, latent_channels, condition_channels, packed_condition_channels) <= 0:
        raise ValueError("geometry/channel values must be positive")
    if height % 16 or width % 16:
        raise ValueError("output geometry must be divisible by 16")
    if any(k > c for k, c in zip(internal_channels, channels)):
        raise ValueError("internal width cannot exceed its stage interface width")

    stage_hw = (
        (height // 16, width // 16),
        (height // 8, width // 8),
        (height // 4, width // 4),
        (height // 2, width // 2),
    )
    temporal_rates = (0.25, 0.25, 0.5, 1.0)
    c0, c1, c2, c3 = channels

    parts: dict[str, float] = {}
    area0 = stage_hw[0][0] * stage_hw[0][1]
    parts["condition_projection"] = (
        area0 * temporal_rates[0] * packed_condition_channels * condition_channels
    )
    parts["input_conv"] = (
        area0
        * temporal_rates[0]
        * (latent_channels + condition_channels)
        * c0
        * 9
    )

    for stage, (c, k, blocks, (h, w), rate) in enumerate(
        zip(channels, internal_channels, blocks_per_stage, stage_hw, temporal_rates)
    ):
        area = h * w
        per_block = area * rate * 9 * (2 * c * k + k * k + k * c)
        parts[f"stage{stage}_blocks"] = per_block * blocks

    area1 = stage_hw[1][0] * stage_hw[1][1]
    area2 = stage_hw[2][0] * stage_hw[2][1]
    area3 = stage_hw[3][0] * stage_hw[3][1]

    parts["tgrow01"] = area1 * temporal_rates[1] * c0 * c0
    parts["transition01_conv"] = area1 * temporal_rates[1] * c0 * c1 * 9

    parts["tgrow12"] = area2 * temporal_rates[2] * c1 * c1 * 3
    parts["transition12_conv"] = area2 * temporal_rates[2] * c1 * c2 * 9

    parts["tgrow23"] = area3 * temporal_rates[3] * c2 * c2 * 3
    parts["transition23_conv"] = area3 * temporal_rates[3] * c2 * c3 * 9

    full_area = height * width
    parts["resizeconv_rgb_head"] = full_area * c3 * 3 * 9

    total = float(sum(parts.values()))
    return {
        "output_height": height,
        "output_width": width,
        "latent_channels": latent_channels,
        "condition_channels": condition_channels,
        "packed_condition_channels": packed_condition_channels,
        "channels": list(channels),
        "internal_channels": list(internal_channels),
        "blocks_per_stage": list(blocks_per_stage),
        "parts_gmac": {name: value / 1e9 for name, value in parts.items()},
        "total_mac": total,
        "total_gmac": total / 1e9,
        "total_gflops_if_mac_is_2_flops": 2.0 * total / 1e9,
    }


def _csv4(value: str) -> tuple[int, int, int, int]:
    result = tuple(int(part.strip()) for part in value.split(","))
    if len(result) != 4:
        raise argparse.ArgumentTypeError("expected four comma-separated integers")
    return result  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--latent-channels", type=int, default=48)
    parser.add_argument("--condition-channels", type=int, default=128)
    parser.add_argument("--channels", type=_csv4, default=(192, 128, 64, 32))
    parser.add_argument("--internal-channels", type=_csv4, default=(128, 96, 48, 24))
    parser.add_argument("--blocks-per-stage", type=_csv4, default=(2, 2, 2, 1))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = estimate_resizeconv_decoder_macs(
        output_height=args.height,
        output_width=args.width,
        latent_channels=args.latent_channels,
        condition_channels=args.condition_channels,
        channels=args.channels,
        internal_channels=args.internal_channels,
        blocks_per_stage=args.blocks_per_stage,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
