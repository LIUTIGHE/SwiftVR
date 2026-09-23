"""CPU tests for tools/plan_ultravideo_training_views.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "plan_ultravideo_training_views.py"
_SPEC = importlib.util.spec_from_file_location("plan_ultravideo_training_views", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class UltraVideoTrainingViewPlannerTest(unittest.TestCase):
    def test_cadence_stride_normalizes_high_fps(self):
        self.assertEqual(_TOOL._cadence_stride(24.0), 1)
        self.assertEqual(_TOOL._cadence_stride(30.0), 1)
        self.assertEqual(_TOOL._cadence_stride(50.0), 2)
        self.assertEqual(_TOOL._cadence_stride(60.0), 2)

    def test_canonical_size_preserves_aspect_before_divisibility_crop(self):
        self.assertEqual(_TOOL._canonical_size(7680, 4320, 3840), (3840, 2160))
        width, height = _TOOL._canonical_size(3840, 2026, 3840)
        self.assertEqual(width, 3840)
        self.assertEqual(width % 3, 0)
        self.assertEqual(height % 3, 0)
        self.assertLessEqual(2026 - height, 2)

    def test_candidate_specs_are_reproducible_and_valid(self):
        left = _TOOL._candidate_specs(
            frame_count=100,
            fps=60.0,
            canonical_hq_size=(1280, 720),
            clip_length=13,
            crop_size=128,
            candidate_count=12,
            seed=7,
        )
        right = _TOOL._candidate_specs(
            frame_count=100,
            fps=60.0,
            canonical_hq_size=(1280, 720),
            clip_length=13,
            crop_size=128,
            candidate_count=12,
            seed=7,
        )
        self.assertEqual(left, right)
        self.assertEqual(len(left), 12)
        for item in left:
            positions = item["raw_frame_positions"]
            self.assertEqual(len(positions), 13)
            self.assertEqual(positions[1] - positions[0], 2)
            top, x, height, width = item["crop_box_hq"]
            self.assertGreaterEqual(top, 0)
            self.assertGreaterEqual(x, 0)
            self.assertEqual((height, width), (128, 128))

    def test_selector_returns_4_2_2_without_duplicates(self):
        candidates = []
        for index in range(16):
            candidates.append(
                {
                    "candidate_index": index,
                    "raw_frame_start": index * 2,
                    "raw_frame_stride": 1,
                    "raw_frame_positions": list(range(index * 2, index * 2 + 13)),
                    "crop_box_hq": [index * 3, index * 5, 128, 128],
                    "horizontal_flip": False,
                    "hr_highpass_l1": 0.01 + index * 0.001,
                    "clean_sr_highpass_gap": 0.002 + index * 0.001,
                    "structural_motion_l1": 0.003 + (15 - index) * 0.001,
                    "raw_motion_l1": 0.01,
                    "luma_motion_l1": 0.001,
                    "temporal_spike_ratio": 1.5,
                }
            )
        selected = _TOOL._select_views(
            candidates,
            clip_length=13,
            stride=1,
            detail_count=4,
            detail_motion_count=2,
            random_count=2,
            diversity_weight=0.35,
            spike_limit=4.0,
            seed=9,
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({item["candidate_index"] for item in selected}), 8)
        counts = {}
        for item in selected:
            counts[item["selection_category"]] = counts.get(item["selection_category"], 0) + 1
        self.assertEqual(counts, {"detail": 4, "detail_motion": 2, "random": 2})


if __name__ == "__main__":
    unittest.main()
