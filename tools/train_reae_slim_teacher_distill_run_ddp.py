#!/usr/bin/env python3
"""Validated CLI entry point for ReAE-Slim teacher-only distillation.

The implementation lives in ``train_reae_slim_teacher_distill_ddp.py``.  This
wrapper intentionally replaces only its parser construction so the shared formal
Stage-B parser's existing ``--teacher-l2-weight`` option is reused rather than
registered twice.  Legacy GT/dual-LPIPS options are set to zero because they are
not part of the ReAE-Slim optimization objective.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_reae_slim_teacher_distill_ddp as base


def _build_parser() -> argparse.ArgumentParser:
    parser = base.formal.build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "init_decoder":
            action.required = False
            action.default = None
            action.help = argparse.SUPPRESS
        elif action.dest == "teacher_l2_weight":
            action.default = 10.0
        elif action.dest in ("gt_l2_weight", "lpips_weight"):
            action.default = 0.0
        elif action.dest == "learning_rate":
            action.default = 3e-5
    parser.add_argument("--variant", choices=tuple(base.VARIANT_CHANNELS), required=True)
    parser.add_argument("--prune-calibration-samples", type=int, default=64)
    parser.add_argument("--teacher-lpips-weight", type=float, default=0.1)
    parser.add_argument("--teacher-temporal-weight", type=float, default=1.0)
    return parser


def main() -> int:
    base.build_parser = _build_parser
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
