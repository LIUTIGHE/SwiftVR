"""CPU tests for tools/audit_triplet_alignment.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "audit_triplet_alignment.py"
_SPEC = importlib.util.spec_from_file_location("audit_triplet_alignment", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class AuditTripletAlignmentTest(unittest.TestCase):
    def test_fraction_parser(self):
        self.assertAlmostEqual(_TOOL._fraction_to_float("30000/1001"), 29.97002997)
        self.assertIsNone(_TOOL._fraction_to_float("0/0"))
        self.assertIsNone(_TOOL._fraction_to_float("N/A"))

    def test_identical_pair_metrics(self):
        frame = np.full((12, 16, 3), 127, dtype=np.uint8)
        metrics = _TOOL.pair_metrics(frame, frame.copy())
        self.assertEqual(metrics.mae, 0.0)
        self.assertEqual(metrics.rmse, 0.0)
        self.assertEqual(metrics.psnr, float("inf"))
        self.assertAlmostEqual(metrics.ssim_gray_global, 1.0)

    def test_resize_rgb_uses_height_width_order(self):
        frame = np.zeros((8, 12, 3), dtype=np.uint8)
        resized = _TOOL.resize_rgb(frame, (4, 6))
        self.assertEqual(resized.shape, (4, 6, 3))

    def test_offset_search_recovers_known_shift(self):
        height, width = 24, 32
        base = {}
        for index in range(12):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, (index * 2) % width : ((index * 2) % width) + 2] = 255
            base[index] = frame
        # Reference frame t corresponds to source frame t+2.
        reference = {index: base[index + 2] for index in range(2, 8)}
        anchors = list(range(2, 8))
        best, scores = _TOOL.find_best_offset_from_sequences(
            base,
            reference,
            anchors,
            range(-3, 4),
            max_side=64,
        )
        self.assertEqual(best, 2)
        self.assertEqual(scores[2], 0.0)

    def test_manifest_filter_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            rows = [
                {
                    "sample_id": "a",
                    "hr": "a_hr",
                    "hq": "a_hq",
                    "lr": "a_lr",
                    "split": "train",
                },
                {
                    "sample_id": "b",
                    "hr": "b_hr",
                    "hq": "b_hq",
                    "lr": "b_lr",
                    "split": "val",
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            records = _TOOL.read_manifest(path, split="val")
            self.assertEqual([record["sample_id"] for record in records], ["b"])

    def test_summary_counts_status_and_offsets(self):
        results = [
            {
                "status": "pass",
                "temporal_alignment": {
                    "best_hr_offset_relative_to_hq": 0,
                    "best_hq_offset_relative_to_lr": 0,
                },
                "metrics": {
                    "hr_downsample_vs_hq": {"psnr": 40.0},
                    "hq_vs_lr": {"psnr": 25.0},
                },
            },
            {
                "status": "fail",
                "temporal_alignment": {
                    "best_hr_offset_relative_to_hq": 2,
                    "best_hq_offset_relative_to_lr": -1,
                },
                "metrics": {
                    "hr_downsample_vs_hq": {"psnr": 38.0},
                    "hq_vs_lr": {"psnr": 24.0},
                },
            },
        ]
        summary = _TOOL.summarize_results(results)
        self.assertEqual(summary["status_counts"]["pass"], 1)
        self.assertEqual(summary["status_counts"]["fail"], 1)
        self.assertEqual(summary["best_hr_hq_offset_counts"], {"0": 1, "2": 1})
        self.assertEqual(summary["best_hq_lr_offset_counts"], {"0": 1, "-1": 1})
        self.assertEqual(summary["mean_metrics"]["hr_downsample_vs_hq_psnr"], 39.0)

    def test_help_executes(self):
        completed = subprocess.run(
            [sys.executable, str(_TOOL_PATH), "--help"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertIn("--manifest", completed.stdout)
        self.assertIn("--offset-radius", completed.stdout)


if __name__ == "__main__":
    unittest.main()
