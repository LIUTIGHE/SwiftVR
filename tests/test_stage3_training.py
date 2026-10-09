"""CPU tests for SwiftVR Stage-3 reconstruction objectives and metrics."""

from __future__ import annotations

import math
import unittest

import torch

from swiftvr.training import (
    TrainingCursor,
    VideoMetricAccumulator,
    advance_cursor_batches,
    stage3_reconstruction_objective,
    temporal_difference_mse,
    video_ssim,
)


class Stage3TrainingTest(unittest.TestCase):
    def test_temporal_difference_mse_is_zero_for_identical_video(self):
        video = torch.rand(2, 5, 3, 8, 8)
        value = temporal_difference_mse(video, video.clone())
        self.assertEqual(float(value), 0.0)

    def test_temporal_difference_mse_matches_known_motion_error(self):
        target = torch.zeros(1, 2, 3, 4, 4)
        prediction = target.clone()
        prediction[:, 1:] = 1.0
        value = temporal_difference_mse(prediction, target)
        self.assertAlmostEqual(float(value), 1.0, places=6)

    def test_stage3_objective_combines_pixel_and_temporal_terms(self):
        target = torch.zeros(1, 2, 3, 2, 2)
        prediction = target.clone()
        prediction[:, 1:] = 1.0
        pixel = torch.nn.functional.l1_loss(prediction, target)
        objective = stage3_reconstruction_objective(
            {
                "prediction": prediction,
                "target": target,
                "pixel_l1": pixel,
            },
            pixel_weight=2.0,
            temporal_weight=3.0,
        )
        self.assertAlmostEqual(float(objective["pixel_l1"]), 0.5, places=6)
        self.assertAlmostEqual(float(objective["temporal_mse"]), 1.0, places=6)
        self.assertAlmostEqual(float(objective["loss"]), 4.0, places=6)

    def test_video_ssim_is_one_for_identical_video(self):
        video = torch.rand(1, 3, 3, 16, 16)
        value = video_ssim(video, video.clone())
        self.assertAlmostEqual(float(value), 1.0, places=5)

    def test_metric_accumulator_reports_known_error(self):
        target = torch.zeros(1, 2, 3, 8, 8)
        prediction = torch.full_like(target, 0.5)
        accumulator = VideoMetricAccumulator()
        accumulator.update(prediction, target)
        result = accumulator.compute()
        self.assertAlmostEqual(float(result["mae"]), 0.5, places=6)
        self.assertAlmostEqual(float(result["mse"]), 0.25, places=6)
        self.assertAlmostEqual(float(result["rmse"]), 0.5, places=6)
        self.assertAlmostEqual(float(result["psnr"]), 10 * math.log10(4), places=6)
        self.assertEqual(result["batches"], 1)
        self.assertEqual(result["frames"], 2)
        self.assertTrue(math.isfinite(float(result["ssim"])))

    def test_accumulation_cursor_separates_batches_and_optimizer_steps(self):
        cursor = TrainingCursor(global_step=4, epoch=1, batch_in_epoch=8)
        advanced = advance_cursor_batches(
            cursor,
            consumed_batches=4,
            batches_per_epoch=10,
            optimizer_steps=1,
        )
        self.assertEqual(advanced.global_step, 5)
        self.assertEqual(advanced.epoch, 2)
        self.assertEqual(advanced.batch_in_epoch, 2)

    def test_accumulation_cursor_can_drop_epoch_tail_without_step(self):
        cursor = TrainingCursor(global_step=4, epoch=1, batch_in_epoch=8)
        advanced = advance_cursor_batches(
            cursor,
            consumed_batches=2,
            batches_per_epoch=10,
            optimizer_steps=0,
        )
        self.assertEqual(advanced.global_step, 4)
        self.assertEqual(advanced.epoch, 2)
        self.assertEqual(advanced.batch_in_epoch, 0)


if __name__ == "__main__":
    unittest.main()
