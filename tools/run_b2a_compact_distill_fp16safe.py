#!/usr/bin/env python3
"""Launch B2-A distillation with an explicit FP16 GradScaler initial scale.

This is an additive diagnostic launcher.  It preserves the B2-A trainer, model,
loss, optimizer, DDP settings, and checkpoint format, and changes only the
initial AMP loss scale used when the resolved runtime dtype is float16.

Use this to verify the diagnosed first-step overflow before folding the setting
into the formal B2-A trainer configuration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _pop_init_scale(argv: list[str]) -> float:
    flag = "--grad-scaler-init-scale"
    if flag not in argv:
        return 4096.0
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise SystemExit(f"{flag} requires a positive numeric value")
    try:
        value = float(argv[index + 1])
    except ValueError as exc:
        raise SystemExit(f"{flag} must be numeric") from exc
    if not (value > 0.0):
        raise SystemExit(f"{flag} must be positive")
    del argv[index : index + 2]
    return value


def main() -> int:
    init_scale = _pop_init_scale(sys.argv)

    import train_b2a_compact_distill_ddp as trainer

    def build_b2a_grad_scaler(device: torch.device, runtime_dtype: torch.dtype):
        enabled = device.type == "cuda" and runtime_dtype == torch.float16
        kwargs = {"enabled": enabled, "init_scale": float(init_scale)}
        try:
            return torch.amp.GradScaler(device.type, **kwargs)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(**kwargs)

    trainer.build_grad_scaler = build_b2a_grad_scaler

    rank = os.environ.get("RANK", "?")
    if rank in {"0", "?"}:
        print(
            f"B2-A FP16-safe launcher: grad_scaler_init_scale={init_scale:g}",
            flush=True,
        )
    return int(trainer.main())


if __name__ == "__main__":
    raise SystemExit(main())
