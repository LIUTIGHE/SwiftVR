#!/usr/bin/env python3
"""Analytical MAC profile for full and structurally slimmed ReAE decoders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.models.reae_slim_decoder import AGGRESSIVE_CHANNELS, SLIM100_CHANNELS, TEACHER_CHANNELS


M8_DECODER76_CHANNELS = (128, 96, 64, 64)


def _parse_channels(value: str) -> tuple[int, int, int, int]:
    try:
        channels = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("channels must be four comma-separated integers") from exc
    if len(channels) != 4 or any(item <= 0 for item in channels):
        raise argparse.ArgumentTypeError("channels must contain four positive integers")
    return channels  # type: ignore[return-value]


def estimate_reae_decoder_macs(
    channels,
    *,
    output_height: int = 1088,
    output_width: int = 1920,
    latent_channels: int = 48,
    patch_size: int = 2,
):
    c0, c1, c2, c3 = (int(value) for value in channels)
    if output_height % 16 or output_width % 16:
        raise ValueError("output geometry must be divisible by 16")
    h0, w0 = output_height // 16, output_width // 16
    h1, w1 = output_height // 8, output_width // 8
    h2, w2 = output_height // 4, output_width // 4
    h3, w3 = output_height // 2, output_width // 2

    values = {
        "input_conv": h0 * w0 * latent_channels * c0 * 9 * 0.25,
        "stage0_memblocks": h0 * w0 * 108 * c0 * c0 * 0.25,
        "tgrow01": h1 * w1 * c0 * c0 * 0.25,
        "conv01": h1 * w1 * 9 * c0 * c1 * 0.25,
        "stage1_memblocks": h1 * w1 * 108 * c1 * c1 * 0.25,
        "tgrow12": h2 * w2 * 3 * c1 * c1 * 0.5,
        "conv12": h2 * w2 * 9 * c1 * c2 * 0.5,
        "stage2_memblocks": h2 * w2 * 108 * c2 * c2 * 0.5,
        "tgrow23": h3 * w3 * 3 * c2 * c2,
        "conv23": h3 * w3 * 9 * c2 * c3,
        "output_head": h3 * w3 * 9 * c3 * (3 * patch_size**2),
    }
    return {
        "channels": [c0, c1, c2, c3],
        "components_gmac": {key: value / 1e9 for key, value in values.items()},
        "total_gmac": sum(values.values()) / 1e9,
        "total_gflops_2flop_per_mac": 2.0 * sum(values.values()) / 1e9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument(
        "--channels",
        type=_parse_channels,
        action="append",
        default=[],
        help="Optional custom C0,C1,C2,C3 decoder widths; repeat to audit several candidates.",
    )
    args = parser.parse_args()
    result = {
        "teacher": estimate_reae_decoder_macs(
            TEACHER_CHANNELS, output_height=args.height, output_width=args.width
        ),
        "slim100": estimate_reae_decoder_macs(
            SLIM100_CHANNELS, output_height=args.height, output_width=args.width
        ),
        "aggressive": estimate_reae_decoder_macs(
            AGGRESSIVE_CHANNELS, output_height=args.height, output_width=args.width
        ),
        "m8_decoder76": estimate_reae_decoder_macs(
            M8_DECODER76_CHANNELS, output_height=args.height, output_width=args.width
        ),
    }
    if args.channels:
        result["custom"] = [
            estimate_reae_decoder_macs(
                channels, output_height=args.height, output_width=args.width
            )
            for channels in args.channels
        ]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
