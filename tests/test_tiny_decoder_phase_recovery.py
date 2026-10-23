from __future__ import annotations

import unittest

import torch

from swiftvr.training.tiny_decoder_phase import (
    calibrated_phase_weight,
    residual_new_p8_phase_loss,
)
from tools import train_tiny_decoder_resize_conv_phase_recovery_ddp as phase_trainer


class TinyDecoderPhaseRecoveryTests(unittest.TestCase):
    @staticmethod
    def _rgb_from_luma_pattern(pattern: torch.Tensor, *, frames: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        teacher = torch.full((1, frames, 3, *pattern.shape), 0.5, dtype=torch.float32)
        prediction = teacher.clone()
        prediction += pattern.reshape(1, 1, 1, *pattern.shape)
        return prediction, teacher

    def test_parent_period4_component_is_removed(self):
        parent4 = torch.tensor(
            [
                [0.02, -0.01, 0.03, -0.02],
                [0.01, 0.04, -0.03, 0.02],
                [-0.02, 0.03, 0.01, -0.04],
                [0.04, -0.02, 0.02, -0.01],
            ],
            dtype=torch.float32,
        )
        pattern = parent4.repeat(4, 4)  # 16x16, exactly period 4.
        prediction, teacher = self._rgb_from_luma_pattern(pattern)
        loss = residual_new_p8_phase_loss(prediction, teacher)
        self.assertLess(float(loss.item()), 1e-7)

    def test_new_period8_component_is_penalized(self):
        phase8 = torch.zeros((8, 8), dtype=torch.float32)
        phase8[0, 0] = 0.08
        phase8[4, 0] = -0.08
        pattern = phase8.repeat(2, 2)
        prediction, teacher = self._rgb_from_luma_pattern(pattern)
        loss = residual_new_p8_phase_loss(prediction, teacher)
        self.assertGreater(float(loss.item()), 1e-3)

    def test_opposite_frame_phase_does_not_cancel(self):
        phase8 = torch.zeros((8, 8), dtype=torch.float32)
        phase8[0, 0] = 0.08
        phase8[4, 0] = -0.08
        pattern = phase8.repeat(2, 2)
        teacher = torch.full((1, 2, 3, 16, 16), 0.5, dtype=torch.float32)
        prediction = teacher.clone()
        prediction[:, 0] += pattern.reshape(1, 1, 16, 16)
        prediction[:, 1] -= pattern.reshape(1, 1, 16, 16)
        loss = residual_new_p8_phase_loss(prediction, teacher)
        self.assertGreater(float(loss.item()), 1e-3)

    def test_phase_loss_is_differentiable(self):
        phase8 = torch.zeros((8, 8), dtype=torch.float32)
        phase8[1, 2] = 0.05
        phase8[5, 2] = -0.05
        pattern = phase8.repeat(2, 2)
        prediction, teacher = self._rgb_from_luma_pattern(pattern)
        prediction.requires_grad_(True)
        loss = residual_new_p8_phase_loss(prediction, teacher)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(float(prediction.grad.abs().sum().item()), 0.0)

    def test_calibrated_weight_matches_target_gradient_ratio(self):
        weight = calibrated_phase_weight(2.0, 4.0, target_ratio=0.1)
        self.assertAlmostEqual(weight, 0.05)
        self.assertAlmostEqual(weight * 4.0 / 2.0, 0.1)

    def test_phase_trainer_defaults_to_auto_calibration(self):
        parser = phase_trainer.build_parser()
        args = parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--init-decoder", "init",
                "--train-cache", "train_cache",
                "--val-cache", "val_cache",
                "--manifest", "train.jsonl",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
            ]
        )
        self.assertIsNone(args.phase_loss_weight)
        self.assertEqual(args.phase_gradient_target_ratio, 0.1)
        phase_trainer._validate_args(args)


if __name__ == "__main__":
    unittest.main()
