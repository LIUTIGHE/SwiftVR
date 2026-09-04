#!/usr/bin/env python3
"""Trace one D1024-MoE TA training step without changing trainer semantics.

Launch this script with the exact arguments accepted by
``train_b2b_moe_ta_distill_ddp.py``.  It monkey-patches only timing/logging
wrappers around the existing trainer so a DDP stall can be localized to:

  data iterator -> TA cache load -> H2D batch move -> model forward
  -> distillation objective -> backward -> optimizer step.

No tensors, losses, routing decisions, optimizer settings, or synchronization
semantics are intentionally changed.  Use a fresh output directory and normally
set ``--max-steps 1 --log-every 1 --validate-every 0 --save-every 1``.  The
optional ``--validate-at-start`` may be kept when reproducing a stall that appears
only after rank-0 validation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools import train_b2b_moe_ta_distill_ddp as trainer


def _rank() -> str:
    return os.environ.get("RANK", "?")


def _stamp(stage: str, event: str, started: float | None = None) -> None:
    now = time.perf_counter()
    suffix = "" if started is None else f" dt={now - started:.6f}s"
    print(f"[TRACE rank={_rank()}] {stage} {event}{suffix}", flush=True)


class _TracedLoader:
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        index = 0
        while True:
            started = time.perf_counter()
            _stamp("data_next", f"begin index={index}")
            try:
                batch = next(iterator)
            except StopIteration:
                _stamp("data_next", f"stop index={index}", started)
                return
            _stamp("data_next", f"end index={index}", started)
            index += 1
            yield batch


_original_make_train_loader = trainer.stage_a.make_train_loader


def _make_train_loader(*args, **kwargs):
    started = time.perf_counter()
    _stamp("make_train_loader", "begin")
    loader = _original_make_train_loader(*args, **kwargs)
    _stamp("make_train_loader", "end", started)
    return _TracedLoader(loader)


_original_cache_load_batch = trainer.TeacherVelocityCache.load_batch


def _cache_load_batch(self, *args, **kwargs):
    started = time.perf_counter()
    _stamp("ta_cache_load_batch", "begin")
    value = _original_cache_load_batch(self, *args, **kwargs)
    _stamp("ta_cache_load_batch", "end", started)
    return value


_original_move_video_batch = trainer.move_video_batch


def _move_video_batch(*args, **kwargs):
    started = time.perf_counter()
    _stamp("move_video_batch", "begin")
    value = _original_move_video_batch(*args, **kwargs)
    _stamp("move_video_batch", "end", started)
    return value


_original_forward = trainer.B2BMoEVelocityDistillationForward.forward


def _forward(self, *args, **kwargs):
    started = time.perf_counter()
    _stamp("model_forward", "begin")
    value = _original_forward(self, *args, **kwargs)
    _stamp("model_forward", "end", started)
    return value


_original_objective = trainer.velocity_distillation_objective


def _objective(*args, **kwargs):
    started = time.perf_counter()
    _stamp("velocity_objective", "begin")
    value = _original_objective(*args, **kwargs)
    _stamp("velocity_objective", "end", started)
    return value


def _patch_backward() -> None:
    import torch

    original = torch.autograd.backward

    def traced_backward(*args, **kwargs):
        started = time.perf_counter()
        _stamp("autograd_backward", "begin")
        try:
            return original(*args, **kwargs)
        finally:
            _stamp("autograd_backward", "end", started)

    torch.autograd.backward = traced_backward


def _patch_optimizer_step() -> None:
    import torch

    original = torch.optim.AdamW.step

    def traced_step(self, *args, **kwargs):
        started = time.perf_counter()
        _stamp("adamw_step", "begin")
        try:
            return original(self, *args, **kwargs)
        finally:
            _stamp("adamw_step", "end", started)

    torch.optim.AdamW.step = traced_step


def main() -> int:
    trainer.stage_a.make_train_loader = _make_train_loader
    trainer.TeacherVelocityCache.load_batch = _cache_load_batch
    trainer.move_video_batch = _move_video_batch
    trainer.B2BMoEVelocityDistillationForward.forward = _forward
    trainer.velocity_distillation_objective = _objective
    _patch_backward()
    _patch_optimizer_step()
    _stamp("wrapper", "instrumentation_installed")
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
