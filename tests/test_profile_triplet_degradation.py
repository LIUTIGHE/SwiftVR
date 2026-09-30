"""CPU tests for tools/profile_triplet_degradation.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "profile_triplet_degradation.py"
_SPEC = importlib.util.spec_from_file_location("profile_triplet_degradation", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class TripletDegradationProfileTest(unittest.TestCase):
    def test_identical_pair_has_perfect_pixel_metrics(self):
        image = np.full((16, 20, 3), 128, dtype=np.uint8)
        metrics = _TOOL._pair_metrics(image, image.copy())
        self.assertTrue(np.isinf(metrics["hq_lr_psnr"]))
        self.assertEqual(metrics["hq_lr_mae_01"], 0.0)
        self.assertAlmostEqual(metrics["hq_lr_ssim_gray_global"], 1.0, places=7)
        self.assertEqual(metrics["mean_luma_abs_shift"], 0.0)

    def test_added_pattern_changes_highpass_and_pixel_metrics(self):
        hq = np.full((16, 20, 3), 128, dtype=np.uint8)
        lr = hq.copy()
        lr[:, ::2] = 160
        metrics = _TOOL._pair_metrics(hq, lr)
        self.assertTrue(np.isfinite(metrics["hq_lr_psnr"]))
        self.assertGreater(metrics["hq_lr_mae_01"], 0.0)
        self.assertGreater(metrics["lr_highpass_l1"], metrics["hq_highpass_l1"])

    def test_positions_cover_sequence_endpoints(self):
        self.assertEqual(_TOOL._positions(10, 3), [0, 4, 9])
        self.assertEqual(_TOOL._positions(1, 3), [0])


if __name__ == "__main__":
    unittest.main()
