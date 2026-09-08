#!/usr/bin/env python3
"""Formal M8-C joint-recovery entrypoint.

This intentionally reuses ``train_m8_joint_coadapt_ddp.py`` and only locks the
validated M8-C operating point:

* M8 D1024/L20 + Decoder76;
* local batch 4 x 4 GPUs x grad-accum 4 = global effective batch 64;
* early adapter/router LR 1e-6, tail-6 full-block LR 2e-6;
* warmed Decoder76 LR 2e-5;
* 5k-step cosine schedule with 100-step warm-up;
* validation every 250 steps, full checkpoint every 500 steps;
* FP16 GradScaler starts at 1 and does not grow during this short gate.

The fixed low loss scale is deliberate: M8-B empirically converged stably at
scale=1, while the legacy decoder trainer failed before its first update when the
default large loss scale overflowed.  A genuine non-finite gradient at scale=1 is
therefore treated as a real numerical failure by the reused core trainer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_m8_joint_coadapt_ddp as core


_BASE_BUILD_PARSER = core.build_parser


def _set_default(parser, dest: str, value) -> None:
    found = False
    for action in parser._actions:
        if action.dest == dest:
            action.default = value
            found = True
            break
    if not found:
        raise RuntimeError(f"M8-C core parser is missing --{dest.replace('_', '-')}")


def build_formal_parser():
    parser = _BASE_BUILD_PARSER()
    parser.description = __doc__
    defaults = {
        "batch_size": 4,
        "num_workers": 8,
        "seed": 20260908,
        "tail_full_blocks": 6,
        "transformer_light_learning_rate": 1e-6,
        "transformer_tail_learning_rate": 2e-6,
        "decoder_learning_rate": 2e-5,
        "gradient_accumulation_steps": 4,
        "expected_global_batch_size": 64,
        "lr_warmup_steps": 100,
        "min_lr_ratio": 0.1,
        "velocity_nmse_weight": 0.25,
        "velocity_cosine_weight": 0.25,
        "latent_spatial_weight": 0.5,
        "latent_temporal_weight": 0.5,
        "teacher_rgb_l1_weight": 1.0,
        "teacher_lpips_weight": 0.1,
        "teacher_rgb_temporal_weight": 1.0,
        "router_balance_weight": 0.01,
        "lpips_microbatch_frames": 16,
        "max_steps": 5000,
        "log_every": 20,
        "validate_every": 250,
        "save_every": 500,
        "dtype": "float16",
    }
    for dest, value in defaults.items():
        _set_default(parser, dest, value)
    return parser


def _fixed_fp16_grad_scaler(device: torch.device, runtime_dtype: torch.dtype):
    enabled = device.type == "cuda" and runtime_dtype == torch.float16
    kwargs = {
        "enabled": enabled,
        "init_scale": 1.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 1_000_000_000,
    }
    try:
        return torch.amp.GradScaler(device.type, **kwargs)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(**kwargs)


def main() -> int:
    # Keep one canonical implementation of the actual joint loop.  Only replace
    # its parser defaults and FP16 scaler policy for the formal M8-C experiment.
    core.build_parser = build_formal_parser
    core.build_grad_scaler = _fixed_fp16_grad_scaler
    print(
        "[M8-C] formal profile: D1024/L20 + Decoder76, global batch 64, "
        "LR(light/tail/decoder)=1e-6/2e-6/2e-5, fixed FP16 scale=1, "
        "validate=250, save=500, schedule=5000 steps",
        flush=True,
    )
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
