"""CPU tests for UltraVideo materialization audit compatibility helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "audit_ultravideo_lr_materialization.py"
_SPEC = importlib.util.spec_from_file_location("audit_ultravideo_lr_materialization", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class UltraVideoLRMaterializationAuditTest(unittest.TestCase):
    def test_canonical_geometry_matches_planner_rule(self):
        raw_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
        raw_8k = np.zeros((4320, 7680, 3), dtype=np.uint8)
        self.assertEqual(_TOOL._canonical_geometry_from_raw(raw_4k), (3840, 2160))
        self.assertEqual(_TOOL._canonical_geometry_from_raw(raw_8k), (3840, 2160))

    def test_canonical_geometry_preserves_non_16_9_height(self):
        raw = np.zeros((2026, 3840, 3), dtype=np.uint8)
        width, height = _TOOL._canonical_geometry_from_raw(raw)
        self.assertEqual(width, 3840)
        self.assertEqual(width % 3, 0)
        self.assertEqual(height % 3, 0)
        self.assertLessEqual(2026 - height, 2)


if __name__ == "__main__":
    unittest.main()
