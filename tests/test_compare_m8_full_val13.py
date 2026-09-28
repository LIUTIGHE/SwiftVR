"""CPU tests for tools/compare_m8_full_val13.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "compare_m8_full_val13.py"
_SPEC = importlib.util.spec_from_file_location("compare_m8_full_val13", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class CompareM8FullVal13Test(unittest.TestCase):
    def test_parse_models_preserves_order_and_baseline(self):
        values = [
            "M8A=outputs/m8a",
            "TA1500=outputs/ta/checkpoints/step_00001500",
            "StageA4000=outputs/stagea/checkpoints/step_00004000",
        ]
        parsed = _TOOL._parse_models(values)
        self.assertEqual(
            [label for label, _ in parsed],
            ["M8A", "TA1500", "StageA4000"],
        )
        self.assertTrue(str(parsed[0][1]).endswith("outputs/m8a"))

    def test_parse_models_rejects_duplicate_labels(self):
        with self.assertRaisesRegex(ValueError, "Duplicate model label"):
            _TOOL._parse_models(["same=a", "same=b"])

    def test_flatten_record_reports_deltas_against_first_model(self):
        def metrics(stage_psnr: float, gt_psnr: float):
            return {
                "velocity_stage_a": {
                    "velocity_relative_l2": 0.5,
                    "velocity_cosine": 0.8,
                },
                "stage_a": {"psnr": stage_psnr, "ssim": 0.9},
                "gt": {"psnr": gt_psnr, "ssim": 0.8},
                "gt_temporal_difference_mse": 0.01,
            }

        record = {
            "index": 0,
            "sample": "sample0",
            "models": {
                "base": metrics(31.0, 24.7),
                "new": metrics(31.2, 24.8),
            },
        }
        row = _TOOL._flatten_record(record, ["base", "new"], "base")
        self.assertAlmostEqual(row["new_delta_stage_a_psnr_vs_base"], 0.2)
        self.assertAlmostEqual(row["new_delta_gt_psnr_vs_base"], 0.1)


if __name__ == "__main__":
    unittest.main()
