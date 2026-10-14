from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.finalize_stage_a_audit_flops import (
    _flops_value,
    _ratio_to_teacher,
    write_markdown,
)


def _profile(total: int, *, frames: int = 13):
    return {
        "input_shape": [1, frames, 3, 128, 128],
        "flops": {
            "encoder": {"reported_flops": total // 10, "error": None},
            "transformer": {"reported_flops": total * 7 // 10, "error": None},
            "decoder": {"reported_flops": total * 2 // 10, "error": None},
            "end_to_end": {"reported_flops": total, "error": None},
        },
    }


class StageAAuditFlopsTest(unittest.TestCase):
    def test_flops_value_and_ratio(self):
        profile = _profile(1_000_000_000_000)
        self.assertEqual(_flops_value(profile, "end_to_end"), 1_000_000_000_000)
        ratio, reduction = _ratio_to_teacher(750, 1000)
        self.assertEqual(ratio, "0.7500×")
        self.assertEqual(reduction, "25.00%")

    def test_markdown_places_teacher_in_primary_compute_table(self):
        teacher_profile = _profile(1_000_000_000_000)
        student_profile = _profile(800_000_000_000)
        student_profile.update(
            {
                "encoder_latency": {"median_ms": 1.0},
                "transformer_latency": {"median_ms": 2.0},
                "decoder_latency": {"median_ms": 3.0},
                "end_to_end_latency": {"median_ms": 6.0},
                "effective_fps": 100.0,
                "peak_allocated_gb": 1.5,
            }
        )
        report = {
            "models": [
                {
                    "label": "long",
                    "metrics": {
                        "velocity_relative_l2": 0.15,
                        "velocity_cosine": 0.99,
                        "student_teacher_psnr": 40.0,
                        "student_teacher_ssim": 0.98,
                        "student_gt_psnr": 24.3,
                        "student_gt_ssim": 0.78,
                        "student_gt_temporal_difference_mse": 0.001,
                    },
                    "parameters": {"total_parameters": 800, "trainable_parameters": 10},
                    "profile": student_profile,
                }
            ],
            "teacher": {
                "metrics": {
                    "teacher_gt_psnr": 24.2,
                    "teacher_gt_ssim": 0.77,
                    "teacher_gt_temporal_difference_mse": 0.002,
                },
                "parameters": {"total_parameters": 1000, "trainable_parameters": 0},
                "profile": teacher_profile,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.md"
            write_markdown(report, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("Compute — primary comparison", text)
        self.assertIn("Conditional teacher", text)
        self.assertIn("FLOPs / Teacher", text)
        self.assertIn("20.00%", text)
        self.assertIn("Supplementary hardware profile", text)


if __name__ == "__main__":
    unittest.main()
