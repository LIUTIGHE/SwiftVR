#!/usr/bin/env python3
"""Audit the D1024 1S12E2A student against the locked D768 compute point.

The MAC estimate uses the exact canonical SwiftVR steady-state geometry:
1920x1088 internal RGB, 24-frame MIDDLE chunk -> 6 latent frames, 2x2 DiT
spatial patching -> 12,240 tokens/chunk = 510 tokens/output RGB frame.

QKV/out projections, activated FFN linears, and the router are computed directly
from the MoE shape.  Remaining DiT work (window-attention QK/AV, adapters,
patch/output projections) is linear in hidden width at this fixed geometry; its
coefficient is anchored to the three previously validated dense profiles
(D768/D1536/D3072), which agree to <2e-7 GMAC/frame per hidden channel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swiftvr.models.transformer_prompt_free_no_time_moe import (
    WanTransformer3DModelPromptFreeNoTimeMoE,
)
from swiftvr.training.b2b_moe import (
    B2BMoESpec,
    expected_moe_shape,
    parameter_accounting,
    transformer_moe_shape,
)


D768_DIT_GMAC_PER_FRAME = 196.292
CANONICAL_TOKENS_PER_RGB_FRAME = 510
DENSE_OTHER_GMAC_PER_HIDDEN = 0.0837388


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--expected-max-delta-percent", type=float, default=1.5)
    return p


def canonical_compute(shape: dict[str, int | float]) -> dict[str, float]:
    d = int(shape["hidden_dim"])
    layers = int(shape["num_layers"])
    active_ffn = int(shape["active_ffn_dim"])
    experts = int(shape["num_experts"])
    tokens = CANONICAL_TOKENS_PER_RGB_FRAME

    attention_projection = tokens * layers * 4 * d * d / 1e9
    active_ffn_linear = tokens * layers * 2 * d * active_ffn / 1e9
    router_linear = tokens * layers * d * experts / 1e9
    dense_other = DENSE_OTHER_GMAC_PER_HIDDEN * d
    total = attention_projection + active_ffn_linear + router_linear + dense_other
    return {
        "attention_qkv_out_projection_gmac_per_frame": attention_projection,
        "active_ffn_linear_gmac_per_frame": active_ffn_linear,
        "router_linear_gmac_per_frame": router_linear,
        "other_hidden_linear_gmac_per_frame": dense_other,
        "estimated_dit_gmac_per_frame": total,
        "estimated_dit_gflops_per_frame_2flop_per_mac": 2.0 * total,
        "d768_reference_gmac_per_frame": D768_DIT_GMAC_PER_FRAME,
        "delta_vs_d768_gmac_per_frame": total - D768_DIT_GMAC_PER_FRAME,
        "delta_vs_d768_percent": 100.0 * (total / D768_DIT_GMAC_PER_FRAME - 1.0),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.expected_max_delta_percent < 0:
        raise ValueError("--expected-max-delta-percent must be non-negative")

    if args.checkpoint is None:
        shape = expected_moe_shape(B2BMoESpec())
        params = None
        checkpoint = None
    else:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        transformer = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
            str(checkpoint_path),
            subfolder=args.transformer_subfolder,
            low_cpu_mem_usage=True,
        )
        shape = transformer_moe_shape(transformer)
        expected = expected_moe_shape(B2BMoESpec())
        if shape != expected:
            raise ValueError(f"Checkpoint shape differs from locked D1024-MoE: {shape} != {expected}")
        params = parameter_accounting(transformer)
        checkpoint = str(checkpoint_path)

    compute = canonical_compute(shape)
    passed = abs(float(compute["delta_vs_d768_percent"])) <= args.expected_max_delta_percent
    report = {
        "kind": "swiftvr_b2b_d1024_moe_architecture_audit",
        "checkpoint": checkpoint,
        "canonical_geometry": {
            "internal_rgb_resolution": [1920, 1088],
            "middle_rgb_frames": 24,
            "latent_frames": 6,
            "latent_spatial": [120, 68],
            "dit_patch_spatial": [2, 2],
            "tokens_per_middle_chunk": 12240,
            "tokens_per_output_rgb_frame": CANONICAL_TOKENS_PER_RGB_FRAME,
        },
        "shape": shape,
        "parameter_accounting": params,
        "compute": compute,
        "mac_scope_note": (
            "Linear/GEMM MAC estimate. Softmax, top-k, gather/scatter/index_add and "
            "kernel-launch overhead are not represented; real latency must be benchmarked."
        ),
        "pass_tolerance_percent": args.expected_max_delta_percent,
        "status": "PASS" if passed else "FAIL",
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not passed:
        raise SystemExit(
            f"Estimated D1024-MoE compute differs from D768 by "
            f"{compute['delta_vs_d768_percent']:.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
