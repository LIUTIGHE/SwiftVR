#!/usr/bin/env python3
"""Cache-safe D1024-MoE distillation entry point for M5/M6.

Default mode keeps the validated M5 curriculum: train from the D1536 teaching
assistant cache and validate against Stage-A D3072.

Passing ``--stage-a-refine`` switches only the training teacher role to the
original Stage-A D3072 cache.  This is the M6 direct-refinement phase after a
D1536-TA-bootstrapped MoE checkpoint.  The flag is consumed by this wrapper
before the underlying validated trainer parses its normal arguments.

Rank-0 validation runs under ``torch.inference_mode()`` while training resumes
with autograd enabled. SwiftVR shifted-window attention keeps process-local CUDA
index/meta caches; tensors created by validation must not be reused by a later
training forward because index_select backward needs the cached indices.

This wrapper therefore clears the shifted-window caches immediately before and
after every rank-0 validation call. The next training forward rebuilds its
indices in the normal autograd context on rank 0, matching ranks 1..N.
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
_original_write_json = trainer.base._write_json
_original_save_snapshot = trainer.base._save_snapshot
_STAGE_A_REFINE_FLAG = "--stage-a-refine"


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


def _consume_stage_a_refine_flag() -> bool:
    count = sys.argv.count(_STAGE_A_REFINE_FLAG)
    if count > 1:
        raise ValueError(f"{_STAGE_A_REFINE_FLAG} may be specified at most once")
    if count == 0:
        return False
    sys.argv.remove(_STAGE_A_REFINE_FLAG)
    return True


def _configure_stage_a_refinement() -> None:
    """Retarget only the training teacher from D1536 TA to Stage-A D3072."""

    # The underlying trainer uses TA_CACHE_KIND only to validate/record the
    # training cache role. Validation already requires STAGE_A_CACHE_KIND.
    trainer.TA_CACHE_KIND = trainer.STAGE_A_CACHE_KIND

    def write_json(path: Path, value) -> None:
        payload = dict(value) if isinstance(value, dict) else value
        if isinstance(payload, dict) and Path(path).name == "run_config.json":
            payload["trainer"] = "b2b_d1024_moe_stage_a_refine_ddp_v1"
            payload["experiment"] = "m6_d1024_1s12e2a_stage_a_direct_refinement"
            payload["curriculum_phase"] = "M6_stage_a_direct_refinement"
            payload["source_pretraining"] = "D1536_TA_bootstrap"
            payload["training_teacher"] = "stage_a_d3072_reference"
            payload["training_teacher_cache_kind"] = trainer.STAGE_A_CACHE_KIND
        _original_write_json(path, payload)

    def save_snapshot(*args, **kwargs):
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            metadata = dict(metadata)
            metadata["trainer"] = "b2b_d1024_moe_stage_a_refine_ddp_v1"
            metadata["curriculum_phase"] = "M6_stage_a_direct_refinement"
            metadata["training_teacher"] = "stage_a_d3072_reference"
            if "velocity_relative_l2_train_to_ta" in metadata:
                metadata["velocity_relative_l2_train_to_stage_a"] = metadata.pop(
                    "velocity_relative_l2_train_to_ta"
                )
            kwargs["metadata"] = metadata
        return _original_save_snapshot(*args, **kwargs)

    trainer.base._write_json = write_json
    trainer.base._save_snapshot = save_snapshot


def main() -> int:
    stage_a_refine = _consume_stage_a_refine_flag()
    trainer.base.validate_rank0 = _validate_rank0_cache_safe
    if stage_a_refine:
        _configure_stage_a_refinement()
        print(
            "[M6] Stage-A direct refinement enabled: training teacher = Stage-A D3072",
            flush=True,
        )
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
