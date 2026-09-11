#!/usr/bin/env python3
"""B2B-0A architecture/MAC gate for the proposed extreme ReAE decoder.

This is deliberately a read-only diagnostic.  It does not load training data,
does not modify an existing checkpoint, and does not train anything.  The goal is
to establish one unambiguous architecture point before B2B decoder fitting:

    channels = [96, 48, 24, 16]

The script reuses ``SlimReAEDecoder`` (same causal MemBlock/TGrow/PixelShuffle
contract as B1), verifies a minimal forward and checkpoint round-trip, and reuses
the validated analytical B1 MAC estimator at the requested output geometry.

Primary convention: 1 MAC = one multiply-accumulate; the optional FLOP view uses
2 FLOPs/MAC.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.models.reae_slim_decoder import SlimReAEDecoder
from tools.profile_reae_slim_macs import estimate_reae_decoder_macs


EXTREME_CHANNELS = (96, 48, 24, 16)
CANONICAL_1080P_HEIGHT = 1088
CANONICAL_1080P_WIDTH = 1920
CANONICAL_EXTREME_GMAC = 13.35785472
DEFAULT_DIT_DECODER_BUDGET_GMAC = 210.0
DEFAULT_ENCODER_GMAC = 42.278


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=CANONICAL_1080P_HEIGHT)
    parser.add_argument("--width", type=int, default=CANONICAL_1080P_WIDTH)
    parser.add_argument(
        "--dit-decoder-budget-gmac",
        type=float,
        default=DEFAULT_DIT_DECODER_BUDGET_GMAC,
        help="Target combined DiT+decoder GMAC/frame budget.",
    )
    parser.add_argument(
        "--encoder-gmac",
        type=float,
        default=DEFAULT_ENCODER_GMAC,
        help="Encoder GMAC/frame used only for the whole-model budget view.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _architecture_gate() -> dict[str, object]:
    torch.manual_seed(0)
    decoder = SlimReAEDecoder(channels=EXTREME_CHANNELS).float().eval()
    latents = torch.randn(1, 1, 48, 1, 1)
    with torch.no_grad():
        output = decoder(latents, output_frames=1, clamp=False)
    expected_shape = (1, 1, 3, 16, 16)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Extreme decoder forward shape {tuple(output.shape)} != {expected_shape}"
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "extreme_decoder"
        decoder.save_pretrained(root)
        loaded = SlimReAEDecoder.from_pretrained(root, dtype=torch.float32).eval()
        if tuple(loaded.channels) != EXTREME_CHANNELS:
            raise RuntimeError(
                f"Round-trip channels {loaded.channels} != {EXTREME_CHANNELS}"
            )
        for name, value in decoder.state_dict().items():
            torch.testing.assert_close(
                value,
                loaded.state_dict()[name],
                rtol=0,
                atol=0,
            )
        with torch.no_grad():
            loaded_output = loaded(latents, output_frames=1, clamp=False)
        torch.testing.assert_close(output, loaded_output, rtol=0, atol=0)

    return {
        "channels": list(EXTREME_CHANNELS),
        "parameters": _parameter_count(decoder),
        "forward_shape": list(output.shape),
        "save_load_roundtrip": "PASS",
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height/width must be positive")
    if args.dit_decoder_budget_gmac <= 0 or args.encoder_gmac < 0:
        raise ValueError("compute budgets must be positive/non-negative")

    architecture = _architecture_gate()
    macs = estimate_reae_decoder_macs(
        EXTREME_CHANNELS,
        output_height=args.height,
        output_width=args.width,
    )
    decoder_gmac = float(macs["total_gmac"])

    canonical_geometry = (
        args.height == CANONICAL_1080P_HEIGHT
        and args.width == CANONICAL_1080P_WIDTH
    )
    if canonical_geometry and abs(decoder_gmac - CANONICAL_EXTREME_GMAC) > 1e-8:
        raise RuntimeError(
            "Canonical extreme-decoder MAC regression: "
            f"got {decoder_gmac:.8f}, expected {CANONICAL_EXTREME_GMAC:.8f}"
        )

    remaining_dit_gmac = float(args.dit_decoder_budget_gmac) - decoder_gmac
    if remaining_dit_gmac <= 0:
        raise RuntimeError(
            f"Decoder {decoder_gmac:.6f} GMAC exhausts DiT+decoder budget "
            f"{args.dit_decoder_budget_gmac:.6f} GMAC"
        )
    whole_budget_gmac = float(args.encoder_gmac) + float(args.dit_decoder_budget_gmac)

    report = {
        "status": "PASS",
        "kind": "swiftvr_b2b_extreme_decoder_architecture_gate",
        "architecture": architecture,
        "geometry": {
            "output_height": int(args.height),
            "output_width": int(args.width),
            "canonical_1080p_internal_geometry": canonical_geometry,
        },
        "decoder_macs": macs,
        "budget": {
            "dit_decoder_target_gmac_per_frame": float(args.dit_decoder_budget_gmac),
            "decoder_gmac_per_frame": decoder_gmac,
            "remaining_dit_gmac_per_frame": remaining_dit_gmac,
            "encoder_gmac_per_frame": float(args.encoder_gmac),
            "whole_model_target_gmac_per_frame": whole_budget_gmac,
            "whole_model_target_gflops_per_frame_if_1mac_2flops": 2.0 * whole_budget_gmac,
        },
        "note": (
            "Architecture gate only. Passing this test does not establish trained decoder "
            "quality; B2B-0B must test teacher-latent fitting next."
        ),
    }

    print("\n================ B2B-0A extreme decoder gate ================")
    print(f"Channels                    : {list(EXTREME_CHANNELS)}")
    print(f"Parameters                  : {architecture['parameters']:,}")
    print(f"Forward shape               : {architecture['forward_shape']}")
    print(f"Save/load round-trip        : {architecture['save_load_roundtrip']}")
    print(f"Decoder GMAC/frame          : {decoder_gmac:.6f}")
    print(f"Decoder GFLOPs/frame (2/MAC): {2.0 * decoder_gmac:.6f}")
    print(f"DiT+decoder target          : {args.dit_decoder_budget_gmac:.6f} GMAC/frame")
    print(f"Remaining DiT budget        : {remaining_dit_gmac:.6f} GMAC/frame")
    print(f"Whole target incl. encoder  : {whole_budget_gmac:.6f} GMAC/frame")
    print(f"Whole target GFLOPs (2/MAC) : {2.0 * whole_budget_gmac:.6f}")
    print("Status                      : PASS")
    print("==============================================================")

    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
