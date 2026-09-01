#!/usr/bin/env python3
"""Cache-safe wrapper for D1024-MoE TA distillation.

Rank-0 validation runs under ``torch.inference_mode()`` while training resumes
with autograd enabled. SwiftVR shifted-window attention keeps process-local CUDA
index/meta caches; tensors created by validation must not be reused by a later
training forward because index_select backward needs the cached indices.

This wrapper keeps the validated MoE trainer unchanged and only clears the
shifted-window caches immediately before and after every rank-0 validation call.
The next training forward therefore rebuilds its indices in the normal autograd
context on rank 0, matching ranks 1..N.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools import train_b2b_moe_ta_distill_ddp as trainer
from swiftvr.models import transformer as transformer_ops


_original_validate_rank0 = trainer.base.validate_rank0


def _clear_window_caches() -> None:
    transformer_ops._WindowIndexCache.clear()
    transformer_ops._WindowRuntimeMetaCache.clear()


def _validate_rank0_cache_safe(*args, **kwargs):
    # Do not let an older training cache affect validation, and more importantly
    # do not let inference-mode tensors survive into the next autograd forward.
    _clear_window_caches()
    try:
        return _original_validate_rank0(*args, **kwargs)
    finally:
        _clear_window_caches()


def main() -> int:
    trainer.base.validate_rank0 = _validate_rank0_cache_safe
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
