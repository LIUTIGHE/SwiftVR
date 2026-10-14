#!/usr/bin/env python3
"""Train D768 from the B2-A D1536 teaching assistant and validate vs Stage-A.

Training supervision:
    frozen ReAE encoder -> D768 student -> cached D1536 B2-A velocity

Validation reference:
    the original Stage-A D3072 velocity cache and original frozen ReAE decoder

Keeping Stage-A as the validation reference makes every checkpoint directly
comparable with the D3 Direct-KD baseline.  Decoder and GT losses remain absent
from training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
for search_root in (ROOT, TOOLS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from tools import train_b2a_compact_distill_ddp as base


D768_SHAPE = {
    "student_hidden_dim": 768,
    "student_num_heads": 6,
    "student_head_dim": 128,
    "student_ffn_dim": 4080,
    "student_num_layers": 30,
    "student_adapter_dim": 128,
}
TA_CACHE_KIND = "swiftvr_b2b_d1536_ta_velocity"
STAGE_A_CACHE_KIND = "swiftvr_b2a_stage_a_teacher_velocity"

_original_build_parser = base.build_parser
_original_write_json = base._write_json
_original_teacher_velocity_cache = base.TeacherVelocityCache
_train_cache_root: Path | None = None
_val_cache_root: Path | None = None


def build_parser():
    parser = _original_build_parser()
    parser.description = __doc__
    parser.set_defaults(**D768_SHAPE)
    return parser


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    payload = dict(value)
    if path.name == "run_config.json":
        payload["experiment"] = "b2b_d768_d1536_teaching_assistant_v1"
        payload["deployment_priority"] = "stage_a_teacher_behavior"
        payload["training_decoder"] = "none"
        payload["training_teacher"] = "b2a_d1536_teaching_assistant"
        payload["training_teacher_cache_kind"] = TA_CACHE_KIND
        payload["validation_teacher"] = "stage_a_d3072_reference"
        payload["validation_teacher_cache_kind"] = STAGE_A_CACHE_KIND
        payload["validation_decoder"] = "original_frozen_reae"
        payload["gt_role"] = "diagnostic_only"
        payload["checkpoint_selection_metric"] = "stage_a_velocity_relative_l2"
        payload["locked_student_shape"] = {
            "hidden_dim": 768,
            "num_heads": 6,
            "head_dim": 128,
            "ffn_dim": 4080,
            "num_layers": 30,
            "adapter_dim": 128,
        }
    _original_write_json(path, payload)


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


class _RoleAwareTeacherVelocityCache(_original_teacher_velocity_cache):
    """Enforce TA-for-train and Stage-A-for-val while reusing the locked trainer.

    The base B2-A trainer historically hard-codes the Stage-A cache kind.  This
    wrapper keeps on-disk cache metadata untouched and only presents the legacy
    kind in memory for the training cache after verifying that it is the expected
    D1536 TA cache.  Validation must remain the real Stage-A cache.
    """

    def __init__(self, root):
        super().__init__(root)
        resolved = _resolve(Path(root))
        actual_kind = self.metadata.get("kind")
        if _train_cache_root is not None and resolved == _train_cache_root:
            if actual_kind != TA_CACHE_KIND:
                raise ValueError(
                    f"TA training cache kind mismatch: {actual_kind!r} != {TA_CACHE_KIND!r}"
                )
            metadata = dict(self.metadata)
            metadata["actual_kind"] = actual_kind
            metadata["kind"] = STAGE_A_CACHE_KIND
            self.metadata = metadata
            return
        if _val_cache_root is not None and resolved == _val_cache_root:
            if actual_kind != STAGE_A_CACHE_KIND:
                raise ValueError(
                    "TA validation must use the original Stage-A cache; got "
                    f"{actual_kind!r}"
                )
            return
        raise ValueError(f"Unexpected teacher-cache path in TA trainer: {resolved}")


def _cache_paths_from_argv() -> tuple[Path, Path | None]:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--teacher-cache", type=Path, required=True)
    probe.add_argument("--val-teacher-cache", type=Path, default=None)
    known, _unknown = probe.parse_known_args()
    train = _resolve(known.teacher_cache)
    val = None if known.val_teacher_cache is None else _resolve(known.val_teacher_cache)
    if val is not None and train == val:
        raise ValueError("TA training cache and Stage-A validation cache must be different")
    return train, val


def main() -> int:
    global _train_cache_root, _val_cache_root
    _train_cache_root, _val_cache_root = _cache_paths_from_argv()

    # Patch only in this standalone wrapper process.  The validated B2-A and D3
    # Direct trainers are unchanged on disk and on import.
    base.build_parser = build_parser
    base._write_json = _write_json
    base.TeacherVelocityCache = _RoleAwareTeacherVelocityCache
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
