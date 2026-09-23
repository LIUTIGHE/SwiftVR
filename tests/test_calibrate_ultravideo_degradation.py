"""CPU tests for tools/calibrate_ultravideo_degradation.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "calibrate_ultravideo_degradation.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_ultravideo_degradation", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class UltraVideoDegradationCalibrationTest(unittest.TestCase):
    def test_direct_script_help_imports_from_repo_root(self):
        completed = subprocess.run(
            [sys.executable, str(_TOOL_PATH), "--help"],
            cwd=str(_REPO_ROOT),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("Calibrate a deterministic synthetic UltraVideo degradation", completed.stdout)

    def test_source_balanced_selection_uses_distinct_groups(self):
        rows = [
            {"source_group_uid": "a", "clip_id": "a0"},
            {"source_group_uid": "a", "clip_id": "a1"},
            {"source_group_uid": "b", "clip_id": "b0"},
        ]
        selected = _TOOL._select_source_balanced(rows, 2, 7)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({row["source_group_uid"] for row in selected}), 2)

    def test_canonical_hr_preserves_aspect_and_divisibility(self):
        frame = np.zeros((2160, 7680, 3), dtype=np.uint8)
        image = _TOOL._canonical_hr(frame, 3840)
        self.assertEqual(image.size[0], 3840)
        self.assertEqual(image.size[0] % 3, 0)
        self.assertEqual(image.size[1] % 3, 0)

    def test_degradation_is_deterministic_for_same_inputs(self):
        image = Image.fromarray(
            np.tile(np.arange(96, dtype=np.uint8)[None, :, None], (72, 1, 3)),
            mode="RGB",
        )
        left = _TOOL._apply_degradation(
            image,
            preset=_TOOL.PRESETS["base"],
            severity=0.6,
            noise_seed=123,
        )
        right = _TOOL._apply_degradation(
            image,
            preset=_TOOL.PRESETS["base"],
            severity=0.6,
            noise_seed=123,
        )
        self.assertTrue(np.array_equal(np.asarray(left), np.asarray(right)))

    def test_distance_prefers_exact_target(self):
        target = {
            "hq_lr_psnr": {"p10": 25.0, "median": 32.0, "p90": 39.0},
            "highpass_retention_ratio": {"p10": 0.2, "median": 0.5, "p90": 0.7},
        }
        generated = {"metrics": target}
        self.assertEqual(_TOOL._distance(generated, target)["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
