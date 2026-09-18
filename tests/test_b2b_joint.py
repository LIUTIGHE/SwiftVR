from __future__ import annotations

import unittest

import torch

from swiftvr.training.b2b_joint import (
    B2B_EXTREME_DECODER_CHANNELS,
    B2B_TINY_SPEC,
    b2b_compute_budget,
    b2b_joint_objective,
)


class B2BJointTest(unittest.TestCase):
    def test_strict_210_gmac_budget(self) -> None:
        self.assertEqual(B2B_EXTREME_DECODER_CHANNELS, (96, 48, 24, 16))
        self.assertEqual(B2B_TINY_SPEC.hidden_dim, 768)
        self.assertEqual(B2B_TINY_SPEC.num_heads, 6)
        self.assertEqual(B2B_TINY_SPEC.head_dim, 128)
        self.assertEqual(B2B_TINY_SPEC.ffn_dim, 4080)
        self.assertEqual(B2B_TINY_SPEC.num_layers, 30)
        budget = b2b_compute_budget()
        self.assertAlmostEqual(budget["dit_gmac_per_frame"], 196.29195264, places=8)
        self.assertAlmostEqual(
            budget["dit_plus_decoder_gmac_per_frame"],
            209.64980736,
            places=8,
        )
        self.assertGreater(budget["headroom_to_210_gmac"], 0.0)

    def test_joint_objective_backpropagates_student_rgb_and_velocity(self) -> None:
        torch.manual_seed(7)
        student_velocity = torch.randn(2, 3, 2, 4, 4, requires_grad=True)
        teacher_velocity = torch.randn_like(student_velocity)
        student_rgb_source = torch.randn(2, 5, 3, 8, 8, requires_grad=True)
        student_rgb = torch.sigmoid(student_rgb_source)
        teacher_rgb = torch.rand_like(student_rgb)
        target = torch.rand_like(student_rgb)
        objective = b2b_joint_objective(
            student_velocity,
            teacher_velocity,
            student_rgb,
            teacher_rgb,
            target,
        )
        self.assertTrue(torch.isfinite(objective["loss"]))
        objective["loss"].backward()
        self.assertIsNotNone(student_velocity.grad)
        self.assertIsNotNone(student_rgb_source.grad)
        self.assertGreater(float(student_velocity.grad.abs().sum()), 0.0)
        self.assertGreater(float(student_rgb_source.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
