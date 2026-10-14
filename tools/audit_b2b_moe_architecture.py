#!/usr/bin/env python3
"""Audit supported sparse-MoE students under canonical SwiftVR geometry.

The MAC estimate uses the canonical SwiftVR steady-state geometry: 1920x1088
internal RGB, 24-frame MIDDLE chunk -> 6 latent frames, 2x2 DiT spatial patching
-> 12,240 tokens/chunk = 510 tokens/output RGB frame.

QKV/out projections, activated FFN linears, and router linears are computed from
the MoE shape. Remaining per-block hidden-width work is anchored to the validated
30-layer dense profiles and scaled linearly by depth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import inspect

import torch

from swiftvr.models.transformer_prompt_free_no_time_moe import (
    WanTransformer3DModelPromptFreeNoTimeMoE,
)
from swiftvr.training.b2b_moe import (
    B2BMoESpec,
    MOE_ARCHITECTURES,
    M5_MOE_ARCHITECTURE,
    expected_moe_shape,
    moe_spec_from_name,
    parameter_accounting,
    transformer_moe_shape,
)


M8_MOE_ARCHITECTURE = "m8-d1024-l20"
M8_MOE_SPEC = B2BMoESpec(num_layers=20)
# M11 budget candidates. Keep these out of the legacy training registry until
# exact MAC audit + initialization are accepted as a new experiment lineage.
M11A_MOE_ARCHITECTURE = "m11a-d1024-l22"
M11B_MOE_ARCHITECTURE = "m11b-d1024-l20-e576"
M11A_MOE_SPEC = B2BMoESpec(num_layers=22)
M11B_MOE_SPEC = B2BMoESpec(num_layers=20, normal_expert_dim=576)
CANDIDATE_SPECS = {
    M8_MOE_ARCHITECTURE: M8_MOE_SPEC,
    M11A_MOE_ARCHITECTURE: M11A_MOE_SPEC,
    M11B_MOE_ARCHITECTURE: M11B_MOE_SPEC,
}
SUPPORTED_ARCHITECTURES = tuple(MOE_ARCHITECTURES) + tuple(CANDIDATE_SPECS)
BASELINE_M8_TRANSFORMER_GMAC = 132.159
FIXED_REAE_ENCODER_GMAC = 42.278
FIXED_A1_DECODER_GMAC = 75.9504
CANONICAL_MAX_TOTAL_GFLOPS = 530.0
D768_DIT_GMAC_PER_FRAME = 196.292
CANONICAL_TOKENS_PER_RGB_FRAME = 510
DENSE_OTHER_GMAC_PER_HIDDEN_AT_L30 = 0.0837388
REFERENCE_LAYERS = 30


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        default=M5_MOE_ARCHITECTURE,
    )
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--max-total-gflops", type=float, default=CANONICAL_MAX_TOTAL_GFLOPS,
                   help="Hard GFLOPs/frame gate for the entire Encoder + DiT + frozen A1, default 530")
    p.add_argument("--source-config", type=Path, default=None,
                   help="Existing M8 transformer/config.json for verified candidate config export")
    p.add_argument("--export-config", type=Path, default=None,
                   help="Write candidate transformer config JSON only; this is NOT a checkpoint")
    p.add_argument("--meta-parameters", action="store_true",
                   help="Build candidate on torch.meta to audit total/active parameter count, without weights")
    p.add_argument(
        "--expected-max-delta-percent",
        type=float,
        default=None,
        help="Optional absolute tolerance versus D768. Omit for report-only auditing.",
    )
    return p


def _spec(architecture: str) -> B2BMoESpec:
    if architecture in CANDIDATE_SPECS:
        return CANDIDATE_SPECS[architecture]
    return moe_spec_from_name(architecture)


def calibrated_budget_compute(shape: dict[str, int | float]) -> dict[str, float | bool]:
    """M11 delta: exact active Linear MAC changes + legacy residual estimate.

    One emitted MIDDLE RGB frame contains 510 transformer tokens; every
    token executes the shared expert and exactly top_k routed experts.
    The absolute base is the previously measured M8 132.159 GMAC/frame.
    Residual hidden-work changes from layer additions are the existing auditor's
    validated L30 linear extrapolation. This is NOT an independent GPU profile.
    """
    baseline = canonical_compute(expected_moe_shape(M8_MOE_SPEC))
    candidate = canonical_compute(shape)
    delta = candidate["estimated_dit_gmac_per_frame"] - baseline["estimated_dit_gmac_per_frame"]
    dit = BASELINE_M8_TRANSFORMER_GMAC + delta
    total = FIXED_REAE_ENCODER_GMAC + dit + FIXED_A1_DECODER_GMAC
    return {
        "source_measured_m8_transformer_gmac_per_frame": BASELINE_M8_TRANSFORMER_GMAC,
        "relative_dit_delta_gmac_per_frame": delta,
        "calibrated_transformer_gmac_per_frame": dit,
        "fixed_encoder_gmac_per_frame": FIXED_REAE_ENCODER_GMAC,
        "fixed_a1_decoder_gmac_per_frame": FIXED_A1_DECODER_GMAC,
        "calibrated_total_gmac_per_frame": total,
        "calibrated_total_gflops_per_frame_2flop_per_mac": 2.0*total,
        "budget_max_gflops": CANONICAL_MAX_TOTAL_GFLOPS,
        "note": "M8-calibrated MAC estimate; profile candidate on real MIDDLE chunk before training",
    }


def make_config(source: dict, shape: dict) -> dict:
    if not isinstance(source, dict):
        raise ValueError("M8 source config must be a JSON object")
    required = {
        "num_layers":20, "num_attention_heads":8, "attention_head_dim":128,
        "ffn_dim":1536, "shared_expert_dim":1024, "normal_expert_dim":256,
        "num_experts":12, "top_k":2,
    }
    for key, expected in required.items():
        if source.get(key) != expected:
            raise ValueError(f"Source is not the canonical M8 config: {key}: {source.get(key)} != {expected}")
    if "MoE" not in str(source.get("_class_name", "")):
        raise ValueError("Source transformer config is not a MoE class")
    out = dict(source)
    out["num_layers"] = int(shape["num_layers"])
    out["ffn_dim"] = int(shape["active_ffn_dim"])
    out["normal_expert_dim"] = int(shape["normal_expert_dim"])
    return out


def count_candidate_meta_parameters(config: dict) -> dict:
    signature = inspect.signature(WanTransformer3DModelPromptFreeNoTimeMoE.__init__)
    allowed = set(signature.parameters) - {"self"}
    kwargs = {k:v for k,v in config.items() if k in allowed}
    with torch.device("meta"):
        model = WanTransformer3DModelPromptFreeNoTimeMoE(**kwargs)
    return parameter_accounting(model)


def canonical_compute(shape: dict[str, int | float]) -> dict[str, float]:
    d = int(shape["hidden_dim"])
    layers = int(shape["num_layers"])
    active_ffn = int(shape["active_ffn_dim"])
    experts = int(shape["num_experts"])
    tokens = CANONICAL_TOKENS_PER_RGB_FRAME

    attention_projection = tokens * layers * 4 * d * d / 1e9
    active_ffn_linear = tokens * layers * 2 * d * active_ffn / 1e9
    router_linear = tokens * layers * d * experts / 1e9
    other_hidden = (
        DENSE_OTHER_GMAC_PER_HIDDEN_AT_L30
        * d
        * layers
        / REFERENCE_LAYERS
    )
    total = attention_projection + active_ffn_linear + router_linear + other_hidden
    return {
        "attention_qkv_out_projection_gmac_per_frame": attention_projection,
        "active_ffn_linear_gmac_per_frame": active_ffn_linear,
        "router_linear_gmac_per_frame": router_linear,
        "other_hidden_linear_gmac_per_frame": other_hidden,
        "estimated_dit_gmac_per_frame": total,
        "estimated_dit_gflops_per_frame_2flop_per_mac": 2.0 * total,
        "d768_reference_gmac_per_frame": D768_DIT_GMAC_PER_FRAME,
        "delta_vs_d768_gmac_per_frame": total - D768_DIT_GMAC_PER_FRAME,
        "delta_vs_d768_percent": 100.0 * (total / D768_DIT_GMAC_PER_FRAME - 1.0),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.expected_max_delta_percent is not None and args.expected_max_delta_percent < 0:
        raise ValueError("--expected-max-delta-percent must be non-negative")
    if args.max_total_gflops <= 0:
        raise ValueError("--max-total-gflops must be positive")
    if args.export_config is not None and args.source_config is None:
        raise ValueError("--export-config requires --source-config")
    if args.meta_parameters and args.source_config is None:
        raise ValueError("--meta-parameters requires --source-config")
    spec = _spec(args.architecture)
    expected = expected_moe_shape(spec)
    if args.checkpoint is None:
        shape = expected
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
        if shape != expected:
            raise ValueError(
                f"Checkpoint shape differs from locked {args.architecture}: {shape} != {expected}"
            )
        params = parameter_accounting(transformer)
        checkpoint = str(checkpoint_path)

    compute = canonical_compute(shape)
    calibrated = calibrated_budget_compute(shape)
    passed_budget = calibrated["calibrated_total_gflops_per_frame_2flop_per_mac"] <= args.max_total_gflops
    source_config = None
    if args.source_config:
        source_config = json.loads(args.source_config.expanduser().resolve().read_text())
        source_config = make_config(source_config, shape)
    if args.meta_parameters:
        meta_params = count_candidate_meta_parameters(source_config)
        if params is not None and params != meta_params:
            raise ValueError("Checkpoint parameter audit disagrees with meta model")
        params = meta_params
    if args.export_config:
        target = args.export_config.expanduser().resolve()
        if target.exists():
            raise FileExistsError("Refuse to overwrite exported candidate config")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(source_config, indent=2, sort_keys=True) + "\n")
    passed = (
        None
        if args.expected_max_delta_percent is None
        else abs(float(compute["delta_vs_d768_percent"])) <= args.expected_max_delta_percent
    )
    report = {
        "kind": "swiftvr_b2b_moe_architecture_audit",
        "architecture": args.architecture,
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
        "budget": calibrated,
        "budget_gate": {
            "requested_max_total_gflops": args.max_total_gflops,
            "pass": passed_budget,
            "status": "ESTIMATE_ONLY_REQUIRES_REAL_MODEL_RUNTIME_COUNTER",
        },
        "source_config": None if args.source_config is None else str(args.source_config.expanduser().resolve()),
        "exported_config": None if args.export_config is None else str(args.export_config.expanduser().resolve()),
        "checkpoint_weights_available": checkpoint is not None,
        "mac_scope_note": (
            "Linear/GEMM MAC estimate. Softmax, top-k, gather/scatter/index_add and "
            "kernel-launch overhead are not represented; real latency must be benchmarked. "
            "The residual 'other' term assumes the validated L30 hidden-linear coefficient "
            "scales linearly with layer count."
        ),
        "pass_tolerance_percent": args.expected_max_delta_percent,
        "status": "REPORT" if passed is None else ("PASS" if passed else "FAIL"),
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not passed_budget:
        raise SystemExit(f"Whole-system estimated FLOPs/frame {calibrated['calibrated_total_gflops_per_frame_2flop_per_mac']:.4f} exceeds budget {args.max_total_gflops}")
    if passed is False:
        raise SystemExit(
            f"Estimated {args.architecture} compute differs from D768 by "
            f"{compute['delta_vs_d768_percent']:.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
