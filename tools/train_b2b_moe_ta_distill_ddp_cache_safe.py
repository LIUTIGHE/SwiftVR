#!/usr/bin/env python3
"""Cache-safe sparse-MoE distillation entry point for M5/M6/M7-A.

Architecture modes:

* default / ``--architecture m5-d1024-l30`` keeps the validated M5
  D1024/H8/L30 1S12E2A student;
* ``--architecture m7a-d1152-l25`` selects the M7-A D1152/H9/L25 1S12E2A
  compute-matched width/depth reallocation.

Both architecture modes train from the cached D1536 teaching assistant and
validate against Stage-A D3072.  ``--stage-a-refine`` remains the M6 direct
Stage-A-refinement mode and is intentionally restricted to the M5 architecture;
M7-A must first be compared under the same D1536-TA bootstrap protocol as M5.

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
from swiftvr.training.b2b_moe import (
    M5_MOE_ARCHITECTURE,
    M7A_MOE_ARCHITECTURE,
    MOE_ARCHITECTURES,
    moe_spec_from_name,
)


_original_validate_rank0 = trainer.base.validate_rank0
_original_write_json = trainer.base._write_json
_original_save_snapshot = trainer.base._save_snapshot
_STAGE_A_REFINE_FLAG = "--stage-a-refine"
_ARCHITECTURE_FLAG = "--architecture"


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


def _consume_architecture() -> str:
    """Consume wrapper-only architecture syntax before the base parser runs."""

    values: list[str] = []
    retained = [sys.argv[0]]
    index = 1
    while index < len(sys.argv):
        token = sys.argv[index]
        if token == _ARCHITECTURE_FLAG:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{_ARCHITECTURE_FLAG} requires a value")
            values.append(sys.argv[index + 1])
            index += 2
            continue
        prefix = _ARCHITECTURE_FLAG + "="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
            index += 1
            continue
        retained.append(token)
        index += 1
    if len(values) > 1:
        raise ValueError(f"{_ARCHITECTURE_FLAG} may be specified at most once")
    sys.argv[:] = retained
    architecture = values[0] if values else M5_MOE_ARCHITECTURE
    if architecture not in MOE_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; expected one of "
            f"{sorted(MOE_ARCHITECTURES)}"
        )
    return architecture


def _configure_architecture(architecture: str) -> None:
    """Switch only the locked MoE shape; preserve the validated trainer loop."""

    spec = moe_spec_from_name(architecture)
    trainer.LOCKED_SPEC = spec
    if architecture == M5_MOE_ARCHITECTURE:
        return
    if architecture != M7A_MOE_ARCHITECTURE:
        raise ValueError(f"Unsupported architecture metadata mode: {architecture}")

    def write_json(path: Path, value) -> None:
        payload = dict(value) if isinstance(value, dict) else value
        if isinstance(payload, dict):
            name = Path(path).name
            if name == "run_config.json":
                payload["trainer"] = "b2b_m7a_d1152_l25_moe_d1536_ta_distill_ddp_v1"
                payload["experiment"] = "m7a_d1152_l25_vs_m5_d1024_l30_compute_matched_race"
                payload["curriculum_phase"] = "M7A_width_depth_architecture_gate"
                payload["architecture"] = architecture
                payload["training_teacher"] = "b2a_d1536_teaching_assistant"
                payload["training_teacher_cache_kind"] = trainer.TA_CACHE_KIND
            elif name in {"best.json", "summary.json"}:
                payload["architecture"] = architecture
                payload["curriculum_phase"] = "M7A_width_depth_architecture_gate"
        _original_write_json(path, payload)

    def save_snapshot(*args, **kwargs):
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            metadata = dict(metadata)
            metadata["trainer"] = "b2b_m7a_d1152_l25_moe_d1536_ta_distill_ddp_v1"
            metadata["architecture"] = architecture
            metadata["curriculum_phase"] = "M7A_width_depth_architecture_gate"
            metadata["training_teacher"] = "b2a_d1536_teaching_assistant"
            kwargs["metadata"] = metadata
        return _original_save_snapshot(*args, **kwargs)

    trainer.base._write_json = write_json
    trainer.base._save_snapshot = save_snapshot


def _configure_stage_a_refinement() -> None:
    """Retarget only the M5 training teacher from D1536 TA to Stage-A D3072."""

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
    architecture = _consume_architecture()
    stage_a_refine = _consume_stage_a_refine_flag()
    if stage_a_refine and architecture != M5_MOE_ARCHITECTURE:
        raise ValueError(
            "--stage-a-refine is intentionally restricted to m5-d1024-l30; "
            "M7-A must first use the matched D1536-TA bootstrap protocol"
        )

    trainer.base.validate_rank0 = _validate_rank0_cache_safe
    _configure_architecture(architecture)
    if stage_a_refine:
        _configure_stage_a_refinement()
        print(
            "[M6] Stage-A direct refinement enabled: training teacher = Stage-A D3072",
            flush=True,
        )
    elif architecture == M7A_MOE_ARCHITECTURE:
        print(
            "[M7-A] D1152/H9/L25 1S12E2A enabled: training teacher = D1536 TA",
            flush=True,
        )
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
