from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.audit_stage_a_distillation import (
    parse_frame_indices,
    parse_model_specs,
    write_markdown,
)


class StageAAuditTest(unittest.TestCase):
    def test_parse_model_specs_preserves_order_and_base(self):
        specs = parse_model_specs(
            [
                "init=base",
                "step992=/tmp/run/checkpoints/step_00000992",
                "long=/tmp/run/checkpoints/step_00170000",
            ]
        )
        self.assertEqual([label for label, _ in specs], ["init", "step992", "long"])
        self.assertIsNone(specs[0][1])
        self.assertEqual(specs[1][1].name, "step_00000992")
        self.assertEqual(specs[2][1].name, "step_00170000")

    def test_parse_model_specs_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "Duplicate model label"):
            parse_model_specs(["same=base", "same=/tmp/checkpoint"])

    def test_frame_indices_are_unique_and_ordered(self):
        self.assertEqual(parse_frame_indices("0,6,12,6"), (0, 6, 12))
        with self.assertRaises(ValueError):
            parse_frame_indices("-1")

    def test_markdown_contains_quality_and_compute_tables(self):
        report = {
            "models": [
                {
                    "label": "step992",
                    "metrics": {
                        "velocity_relative_l2": 0.25,
                        "velocity_cosine": 0.97,
                        "student_teacher_psnr": 40.0,
                        "student_teacher_ssim": 0.98,
                        "student_gt_psnr": 24.5,
                        "student_gt_ssim": 0.79,
                        "student_gt_temporal_difference_mse": 0.001,
                    },
                    "parameters": {
                        "total_parameters": 100,
                        "trainable_parameters": 10,
                    },
                    "profile": {
                        "encoder_latency": {"median_ms": 1.0},
                        "transformer_latency": {"median_ms": 2.0},
                        "decoder_latency": {"median_ms": 3.0},
                        "end_to_end_latency": {"median_ms": 6.0},
                        "effective_fps": 100.0,
                        "peak_allocated_gb": 1.5,
                    },
                }
            ],
            "teacher": {
                "metrics": {
                    "teacher_gt_psnr": 24.2,
                    "teacher_gt_ssim": 0.78,
                    "teacher_gt_temporal_difference_mse": 0.002,
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.md"
            write_markdown(report, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("Stage-A Distillation Audit", text)
        self.assertIn("step992", text)
        self.assertIn("Conditional teacher", text)
        self.assertIn("Effective FPS", text)


if __name__ == "__main__":
    unittest.main()
