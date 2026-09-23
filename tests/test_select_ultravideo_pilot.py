"""CPU tests for tools/select_ultravideo_pilot.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "select_ultravideo_pilot.py"
_SPEC = importlib.util.spec_from_file_location("select_ultravideo_pilot", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class UltraVideoPilotSelectorTest(unittest.TestCase):
    def test_spread_positions_are_not_prefix_only(self):
        self.assertEqual(_TOOL._spread_positions(10, 2), [3, 6])
        self.assertEqual(_TOOL._spread_positions(5, 1), [2])

    def test_select_group_uses_time_order(self):
        rows = [
            {"clip_id": "c", "start_time": 30},
            {"clip_id": "a", "start_time": 0},
            {"clip_id": "b", "start_time": 10},
            {"clip_id": "d", "start_time": 40},
        ]
        chosen = _TOOL._select_group(rows, 2)
        self.assertEqual([row["clip_id"] for row in chosen], ["b", "c"])

    def test_eligibility_filters_low_fps(self):
        row = {"fps": 12.0, "frame_width": 3840}
        self.assertFalse(
            _TOOL._eligible(row, min_fps=20.0, max_fps=60.1, min_width=3840)
        )
        row["fps"] = 24.0
        self.assertTrue(
            _TOOL._eligible(row, min_fps=20.0, max_fps=60.1, min_width=3840)
        )

    def test_select_preserves_source_coverage(self):
        rows = []
        for source in ("s1", "s2"):
            for index in range(4):
                rows.append(
                    {
                        "source_group_uid": source,
                        "clip_id": f"{source}-{index}",
                        "start_time": float(index),
                        "fps": 24.0,
                        "frame_width": 3840,
                    }
                )
        selected, summary = _TOOL._select(
            rows,
            clips_per_source=2,
            min_fps=20.0,
            max_fps=60.1,
            min_width=3840,
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(summary["selected_source_group_count"], 2)
        self.assertEqual(summary["underfilled_source_group_count"], 0)


if __name__ == "__main__":
    unittest.main()
