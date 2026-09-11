#!/usr/bin/env python3
"""Trace the B2B-1B validation->training transition without changing training logic.

Run this wrapper with the same CLI as train_b2b_joint_recovery_ddp.py.  It
monkey-patches only logging around barriers, train-loader construction, teacher
cache loads, device moves, B2B forwards and explicit Tensor.backward calls.
The underlying recovery trainer, optimizer, losses, DDP configuration and model
states are otherwise unchanged.

This is intended for a 1-step diagnostic with --lr-warmup-steps 0 and a fresh
output directory.  Every trace line is flushed immediately and contains rank,
local rank, elapsed time and a monotonically increasing local phase counter.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
for search_root in (ROOT, TOOLS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from tools import train_b2b_joint_recovery_ddp as recovery


_STARTED = time.perf_counter()
_PHASE = 0
_BARRIER = 0
_CACHE_LOAD = 0
_FORWARD = 0
_BACKWARD = 0
_MOVE = 0
_LOADER = 0


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", -1))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", -1))


def _trace(message: str) -> None:
    global _PHASE
    _PHASE += 1
    print(
        f"[B2B-TRACE rank={_rank()} local={_local_rank()} "
        f"t={time.perf_counter() - _STARTED:8.3f}s phase={_PHASE:03d}] {message}",
        flush=True,
    )


def _install_traces() -> None:
    global _BARRIER, _CACHE_LOAD, _FORWARD, _BACKWARD, _MOVE, _LOADER

    original_barrier = dist.barrier

    def traced_barrier(*args, **kwargs):
        global _BARRIER
        _BARRIER += 1
        current = _BARRIER
        _trace(f"barrier#{current} ENTER kwargs={kwargs}")
        result = original_barrier(*args, **kwargs)
        _trace(f"barrier#{current} EXIT")
        return result

    # recovery.dist is the torch.distributed module object, so patching dist once
    # covers every barrier call made by the trainer.
    dist.barrier = traced_barrier

    original_make_loader = recovery.stage_a.make_train_loader

    def traced_make_loader(*args, **kwargs):
        global _LOADER
        _LOADER += 1
        current = _LOADER
        _trace(f"make_train_loader#{current} ENTER epoch={kwargs.get('epoch')}")
        result = original_make_loader(*args, **kwargs)
        _trace(f"make_train_loader#{current} EXIT len={len(result)}")
        return result

    recovery.stage_a.make_train_loader = traced_make_loader

    original_cache_load = recovery.TeacherVelocityCache.load_batch

    def traced_cache_load(self, *args, **kwargs):
        global _CACHE_LOAD
        _CACHE_LOAD += 1
        current = _CACHE_LOAD
        _trace(f"teacher_cache.load_batch#{current} ENTER root={self.root}")
        result = original_cache_load(self, *args, **kwargs)
        _trace(
            f"teacher_cache.load_batch#{current} EXIT "
            f"shape={tuple(result.shape)} dtype={result.dtype} device={result.device}"
        )
        return result

    recovery.TeacherVelocityCache.load_batch = traced_cache_load

    original_move = recovery.move_video_batch

    def traced_move(*args, **kwargs):
        global _MOVE
        _MOVE += 1
        current = _MOVE
        _trace(f"move_video_batch#{current} ENTER")
        result = original_move(*args, **kwargs)
        _trace(f"move_video_batch#{current} EXIT")
        return result

    recovery.move_video_batch = traced_move

    original_forward = recovery.B2BJointForward.forward

    def traced_forward(self, *args, **kwargs):
        global _FORWARD
        _FORWARD += 1
        current = _FORWARD
        _trace(f"B2BJointForward#{current} ENTER training={self.training}")
        result = original_forward(self, *args, **kwargs)
        velocity = result.get("velocity") if isinstance(result, dict) else None
        prediction = result.get("prediction") if isinstance(result, dict) else None
        _trace(
            f"B2BJointForward#{current} EXIT "
            f"velocity={None if velocity is None else tuple(velocity.shape)} "
            f"prediction={None if prediction is None else tuple(prediction.shape)}"
        )
        return result

    recovery.B2BJointForward.forward = traced_forward

    original_backward = torch.Tensor.backward

    def traced_backward(self, *args, **kwargs):
        global _BACKWARD
        _BACKWARD += 1
        current = _BACKWARD
        _trace(f"Tensor.backward#{current} ENTER shape={tuple(self.shape)} dtype={self.dtype}")
        result = original_backward(self, *args, **kwargs)
        _trace(f"Tensor.backward#{current} EXIT")
        return result

    torch.Tensor.backward = traced_backward


if __name__ == "__main__":
    _install_traces()
    _trace("diagnostic wrapper installed; entering recovery.main()")
    raise SystemExit(recovery.main())
