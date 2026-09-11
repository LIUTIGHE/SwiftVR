from __future__ import annotations

import unittest

from swiftvr.training.telemetry import scalar_tags_from_record


class TrainingTelemetryTest(unittest.TestCase):
    def test_train_mapping(self):
        tags = scalar_tags_from_record(
            {
                "global_step": 3,
                "loss": 0.2,
                "pixel_l1": 0.1,
                "gradient_norm": 0.5,
                "sample_id": ["ignored"],
            },
            stream="train",
        )
        self.assertEqual(tags["train/loss"], 0.2)
        self.assertEqual(tags["train/pixel_l1"], 0.1)
        self.assertEqual(tags["train/gradient_norm"], 0.5)
        self.assertNotIn("sample_id", tags)

    def test_flat_validation_is_student_gt(self):
        tags = scalar_tags_from_record(
            {"global_step": 20, "psnr": 27.0, "ssim": 0.8, "rmse": 0.04},
            stream="val",
        )
        self.assertEqual(tags["val/student_gt/psnr"], 27.0)
        self.assertEqual(tags["val/student_gt/ssim"], 0.8)
        self.assertEqual(tags["val/student_gt/rmse"], 0.04)

    def test_nested_reference_groups(self):
        tags = scalar_tags_from_record(
            {
                "global_step": 300,
                "student_gt": {"psnr": 28.0},
                "reference_gt": {"psnr": 27.5},
                "student_reference": {"velocity_mse": 0.01},
                "gap": {"psnr": 0.5},
            },
            stream="val",
        )
        self.assertEqual(tags["val/student_gt/psnr"], 28.0)
        self.assertEqual(tags["val/reference_gt/psnr"], 27.5)
        self.assertEqual(tags["val/student_reference/velocity_mse"], 0.01)
        self.assertEqual(tags["val/gap/psnr"], 0.5)


if __name__ == "__main__":
    unittest.main()
