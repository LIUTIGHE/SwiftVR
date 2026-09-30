#!/usr/bin/env python3
"""Audit Stage-A prompt-free/no-time DiT MACs on one canonical streaming MIDDLE chunk.

Additive B2-0 diagnostic only. It reuses the validated Stage-A streaming path and
MAC convention, but attributes self-attention QK/AV to individual transformer
blocks and classifies every executed DiT MAC into a named component. The report
is rejected unless those components reproduce the raw DiT MAC total exactly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BLOCK_COMPONENTS = (
    "q_proj", "k_proj", "v_proj", "qk_matmul", "av_matmul",
    "attn_out_proj", "adapter_down", "adapter_up", "ffn_up", "ffn_down",
)
ALL_COMPONENTS = ("patch_embedding",) + BLOCK_COMPONENTS + ("proj_out",)
_BLOCK_RE = re.compile(r"^transformer\.blocks\.(\d+)\.(.+)$")
_ATTN_RE = re.compile(r"^transformer\.blocks\.(\d+)\.self_attention\.(qk|av)$")


def parse_resolution(value: str) -> tuple[int, int]:
    w, sep, h = value.lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    try:
        w_i, h_i = int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must contain integers") from exc
    if w_i <= 0 or h_i <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return w_i, h_i


def _product(shape) -> int:
    value = 1
    for item in shape:
        value *= int(item)
    return value


def _attention_macs(query, key, value) -> tuple[int, int]:
    q, k, v = tuple(query.shape), tuple(key.shape), tuple(value.shape)
    batch_heads = _product(q[:-2])
    qk = batch_heads * int(q[-2]) * int(k[-2]) * int(q[-1])
    av = batch_heads * int(q[-2]) * int(k[-2]) * int(v[-1])
    return qk, av


def _split_fused_qkv_macs(macs: int) -> tuple[int, int, int]:
    share, remainder = divmod(int(macs), 3)
    if remainder:
        raise ValueError(f"Fused QKV MAC count is not divisible by 3: {macs}")
    return share, share, share


def _classify_linear(suffix: str, module: nn.Module | None, inner_dim: int, ffn_dim: int):
    exact = {
        "attn1.to_q": "q_proj", "attn1.to_k": "k_proj", "attn1.to_v": "v_proj",
        "attn1.to_out.0": "attn_out_proj",
        "prompt_free_adapter.down": "adapter_down",
        "prompt_free_adapter.up": "adapter_up",
    }
    if suffix in exact:
        return exact[suffix]
    if suffix.startswith("ffn.") and isinstance(module, nn.Linear):
        if (module.in_features, module.out_features) == (inner_dim, ffn_dim):
            return "ffn_up"
        if (module.in_features, module.out_features) == (ffn_dim, inner_dim):
            return "ffn_down"
    return None


def aggregate_transformer_macs(
    macs_by_name: dict[str, int],
    modules_by_name: dict[str, nn.Module],
    *,
    num_layers: int,
    inner_dim: int,
    ffn_dim: int,
):
    totals = {name: 0 for name in ALL_COMPONENTS}
    blocks = [{"block": i, **{name: 0 for name in BLOCK_COMPONENTS}} for i in range(num_layers)]
    unknown = []

    for name, value in sorted(macs_by_name.items()):
        if not name.startswith("transformer."):
            continue
        macs = int(value)
        if name == "transformer.patch_embedding":
            totals["patch_embedding"] += macs
            continue
        if name == "transformer.proj_out":
            totals["proj_out"] += macs
            continue

        match = _ATTN_RE.match(name)
        if match:
            block = int(match.group(1))
            component = "qk_matmul" if match.group(2) == "qk" else "av_matmul"
            if 0 <= block < num_layers:
                totals[component] += macs
                blocks[block][component] += macs
            else:
                unknown.append({"name": name, "macs": macs})
            continue

        match = _BLOCK_RE.match(name)
        if not match:
            unknown.append({"name": name, "macs": macs})
            continue
        block, suffix = int(match.group(1)), match.group(2)
        if not 0 <= block < num_layers:
            unknown.append({"name": name, "macs": macs})
            continue

        if suffix == "attn1.to_qkv":
            q, k, v = _split_fused_qkv_macs(macs)
            for component, part in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
                totals[component] += part
                blocks[block][component] += part
            continue

        component = _classify_linear(suffix, modules_by_name.get(name), inner_dim, ffn_dim)
        if component is None:
            unknown.append({"name": name, "macs": macs})
            continue
        totals[component] += macs
        blocks[block][component] += macs

    return totals, blocks, unknown


def architecture_summary(transformer) -> dict[str, object]:
    first = getattr(transformer.blocks[0], "_orig_mod", transformer.blocks[0])
    heads = int(first.attn1.heads)
    inner_dim = int(first.attn1.inner_dim)
    head_dim = inner_dim // heads
    ffn_candidates = [
        module.out_features
        for module in first.ffn.modules()
        if isinstance(module, nn.Linear)
        and module.in_features == inner_dim
        and module.out_features != inner_dim
    ]
    if len(ffn_candidates) != 1:
        raise RuntimeError(f"Cannot infer unique FFN width from block 0: {ffn_candidates}")
    return {
        "num_layers": len(transformer.blocks),
        "inner_dim": inner_dim,
        "num_attention_heads": heads,
        "attention_head_dim": head_dim,
        "ffn_dim": int(ffn_candidates[0]),
        "adapter_dim": int(first.prompt_free_adapter.down.out_features),
        "patch_size": list(transformer.patch_embedding.kernel_size),
        "self_attn_window_hw": list(transformer._self_attn_window_hw),
        "in_channels": int(transformer.patch_embedding.in_channels),
        "out_channels": int(transformer.proj_out.out_features // _product(transformer.patch_embedding.kernel_size)),
        "time_condition_folded": bool(getattr(transformer.config, "time_condition_folded", False)),
        "has_condition_embedder": hasattr(transformer, "condition_embedder"),
    }


def install_block_attention_labels(counter, transformer):
    active = {"block": None}
    handles = []
    for index, block in enumerate(transformer.blocks):
        block = getattr(block, "_orig_mod", block)
        handles.append(block.register_forward_pre_hook(
            lambda _m, _i, index=index: active.__setitem__("block", index)
        ))
        handles.append(block.register_forward_hook(
            lambda _m, _i, _o: active.__setitem__("block", None)
        ))

    original = counter._count_sdpa
    def count_sdpa(self, query, key, value):
        block = active["block"]
        if block is None:
            self.count_errors.append("Observed SDPA outside transformer block")
            return original(query, key, value)
        qk, av = _attention_macs(query, key, value)
        self._add(f"transformer.blocks.{block}.self_attention.qk", "self_attn_qk", qk)
        self._add(f"transformer.blocks.{block}.self_attention.av", "self_attn_av", av)
    counter._count_sdpa = types.MethodType(count_sdpa, counter)

    def close():
        counter._count_sdpa = original
        for handle in handles:
            handle.remove()
    return close


class PatchGeometryCapture:
    def __init__(self, counter, patch_embedding):
        self.counter = counter
        self.input_shape = None
        self.output_shape = None
        self.pre = patch_embedding.register_forward_pre_hook(self._pre)
        self.post = patch_embedding.register_forward_hook(self._post)

    def _pre(self, _module, inputs):
        if self.counter.enabled:
            self.input_shape = tuple(int(x) for x in inputs[0].shape)

    def _post(self, _module, _inputs, output):
        if self.counter.enabled:
            self.output_shape = tuple(int(x) for x in output.shape)

    def close(self):
        self.pre.remove()
        self.post.remove()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Full Stage-A prompt-free/no-time checkpoint root used by inference.")
    p.add_argument("--resolution", type=parse_resolution, default=(1920, 1080))
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--clip-len", type=int, default=24)
    p.add_argument("--dit-overlap", type=int, default=0)
    p.add_argument("--warmup-middle", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    p.add_argument("--attention-backend", choices=("sdpa",), default="sdpa")
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--expected-dit-gmac-per-frame", type=float, default=None)
    p.add_argument("--expected-tolerance-gmac", type=float, default=0.05)
    p.add_argument("--output-json", type=Path, default=Path("outputs/b2_dit_mac_anatomy.json"))
    return p


def _gmac_per_frame(macs: int, frames: int) -> float:
    return macs / frames / 1e9


def print_report(report):
    a, g, t, s = report["architecture"], report["dit_geometry"], report["component_gmac_per_output_frame"], report["sanity"]
    print("\n=== B2-0 DiT architecture ===")
    print(f"layers={a['num_layers']} hidden={a['inner_dim']} ffn={a['ffn_dim']} adapter={a['adapter_dim']}")
    print(f"heads={a['num_attention_heads']} x {a['attention_head_dim']} patch={a['patch_size']} window={a['self_attn_window_hw']}")
    print(f"time_folded={a['time_condition_folded']} condition_embedder={a['has_condition_embedder']}")
    print("\n=== Counted DiT geometry ===")
    print(f"input BCFHW={g['patch_input_shape']} patch BDFHW={g['patch_output_shape']} tokens={g['tokens_per_chunk']}")
    print("\n=== GMAC / output frame ===")
    for name in ALL_COMPONENTS:
        print(f"{name:20s} {t[name]:12.6f}")
    print(f"{'TOTAL':20s} {s['dit_gmac_per_output_frame']:12.6f}")
    print(f"{'QK+AV':20s} {t['qk_matmul'] + t['av_matmul']:12.6f}")
    print("\nblk |   Q    K    V   QK   AV  Aout  AdDn  AdUp   FFup   FFdn  total")
    for row in report["blocks"]:
        total = sum(row[name] for name in BLOCK_COMPONENTS)
        values = " ".join(f"{row[name]:5.2f}" for name in BLOCK_COMPONENTS)
        print(f"{row['block']:3d} | {values} {total:6.2f}")
    print(f"\nexact_sum={s['exact_component_sum']} unclassified={s['unclassified_count']} qk_calls={s['self_attn_qk_calls']} av_calls={s['self_attn_av_calls']}")


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_len <= 0 or args.clip_len % 4:
        raise ValueError("--clip-len must be a positive multiple of 4")
    if args.dit_overlap != 0:
        raise ValueError("Canonical B2-0 anatomy requires --dit-overlap 0")

    from swiftvr.models import ReAE
    from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
    from tools.profile_stage_a_streaming_macs import (
        DTYPES, PromptFreeMiddleSession, _prepare_video_geometry,
        _run_until_counted_middle, canonical_parameter_summary,
    )
    from tools.runtime_macs import RuntimeMacCounter

    device, dtype = torch.device(args.device), DTYPES[args.dtype]
    root = args.checkpoint.expanduser().resolve()
    geometry = _prepare_video_geometry(args)
    reae = ReAE(str(root / args.reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(root), subfolder=args.transformer_subfolder,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    architecture = architecture_summary(transformer)
    if not architecture["time_condition_folded"] or architecture["has_condition_embedder"]:
        raise RuntimeError("Checkpoint is not the folded prompt-free/no-time Stage-A architecture")
    architecture["transformer_parameters"] = canonical_parameter_summary(reae, transformer)["transformer_params"]

    reae.to(device=device, dtype=dtype).eval()
    transformer.to(device=device, dtype=dtype).eval()
    transformer.prepare_for_inference(attention_backend="sdpa", use_torch_compile=False)
    session = PromptFreeMiddleSession(
        reae, transformer, device=device, dtype=dtype,
        out_w=int(geometry["out_w"]), out_h=int(geometry["out_h"]),
        pad_w=int(geometry["pad_w"]), pad_h=int(geometry["pad_h"]),
        overlap=0,
    )

    counter = RuntimeMacCounter()
    counter.add_module("encoder", reae.encoder)
    counter.add_module("transformer", transformer)
    counter.add_module("decoder", reae.decoder)
    close_labels = install_block_attention_labels(counter, transformer)
    capture = PatchGeometryCapture(counter, transformer.patch_embedding)
    try:
        result = _run_until_counted_middle(
            args, geometry, session.step, counter,
            label="b2_stage_a_anatomy", expect_cross_attention=False,
        )
    finally:
        capture.close()
        close_labels()
        counter.close()

    if capture.input_shape is None or capture.output_shape is None:
        raise RuntimeError("Failed to capture counted DiT geometry")
    modules = {
        ("transformer" if not name else f"transformer.{name}"): module
        for name, module in transformer.named_modules()
    }
    component_macs, block_macs, unknown = aggregate_transformer_macs(
        dict(counter.macs_by_name), modules,
        num_layers=int(architecture["num_layers"]),
        inner_dim=int(architecture["inner_dim"]),
        ffn_dim=int(architecture["ffn_dim"]),
    )
    raw_dit = sum(v for k, v in counter.macs_by_name.items() if k.startswith("transformer"))
    component_sum = sum(component_macs.values())
    if unknown:
        raise RuntimeError("Unclassified transformer MACs: " + json.dumps(unknown, indent=2))
    if component_sum != raw_dit:
        raise RuntimeError(f"Component sum mismatch: {component_sum} != {raw_dit}")

    calls = result["macs"]["calls_by_type"]
    layers = int(architecture["num_layers"])
    if int(calls.get("self_attn_qk", 0)) != layers or int(calls.get("self_attn_av", 0)) != layers:
        raise RuntimeError(f"Expected one QK/AV call per block, got {calls}")
    if int(calls.get("cross_attn_qk", 0)):
        raise RuntimeError("Prompt-free/no-time source unexpectedly executed cross-attention")
    for row in block_macs:
        missing = [name for name in BLOCK_COMPONENTS if row[name] <= 0]
        if missing:
            raise RuntimeError(f"Block {row['block']} missing components: {missing}")

    frames = int(result["output_frames"])
    measured = _gmac_per_frame(raw_dit, frames)
    if args.expected_dit_gmac_per_frame is not None:
        error = abs(measured - args.expected_dit_gmac_per_frame)
        if error > args.expected_tolerance_gmac:
            raise RuntimeError(
                f"DiT total mismatch: measured={measured:.6f}, expected={args.expected_dit_gmac_per_frame:.6f}, "
                f"error={error:.6f} > tolerance={args.expected_tolerance_gmac:.6f} GMAC/frame"
            )

    report = {
        "kind": "swiftvr_b2_dit_mac_anatomy",
        "checkpoint": str(root),
        "architecture": architecture,
        "protocol": {
            "counted_chunk_type": result["counted_chunk_type"],
            "target_resolution": [geometry["out_w"], geometry["out_h"]],
            "internal_compute_resolution": [geometry["compute_w"], geometry["compute_h"]],
            "clip_len": args.clip_len, "dit_overlap": 0,
            "input_frames": result["input_frames"], "output_frames": frames,
            "attention_backend": "sdpa", "dtype": args.dtype,
            "mac_convention": result["macs"]["mac_convention"],
        },
        "dit_geometry": {
            "patch_input_shape": list(capture.input_shape),
            "patch_output_shape": list(capture.output_shape),
            "tokens_per_chunk": int(capture.output_shape[2] * capture.output_shape[3] * capture.output_shape[4]),
        },
        "component_gmac_per_output_frame": {
            name: _gmac_per_frame(value, frames) for name, value in component_macs.items()
        },
        "blocks": [
            {"block": row["block"], **{name: _gmac_per_frame(row[name], frames) for name in BLOCK_COMPONENTS}}
            for row in block_macs
        ],
        "sanity": {
            "raw_transformer_macs": raw_dit,
            "component_sum_macs": component_sum,
            "exact_component_sum": component_sum == raw_dit,
            "dit_gmac_per_output_frame": measured,
            "unclassified_count": len(unknown),
            "self_attn_qk_calls": int(calls.get("self_attn_qk", 0)),
            "self_attn_av_calls": int(calls.get("self_attn_av", 0)),
            "cross_attn_qk_calls": int(calls.get("cross_attn_qk", 0)),
            "count_errors": result["macs"]["count_errors"],
            "expected_dit_gmac_per_frame": args.expected_dit_gmac_per_frame,
            "expected_tolerance_gmac": args.expected_tolerance_gmac,
        },
    }
    out = args.output_json.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
