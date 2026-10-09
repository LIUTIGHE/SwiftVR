from __future__ import annotations

import unittest

import torch

from swiftvr.training.lift_distillation import (
    lift_velocity_distillation_objective,
    lift_velocity_terms,
)


class LiftVelocityDistillationTest(unittest.TestCase):
    def test_identity_mapping_has_zero_coarse_and_fine_loss(self) -> None:
        torch.manual_seed(0)
        teacher = torch.randn(2, 3, 4, 5)
        terms = lift_velocity_terms(teacher.clone(), teacher)
        self.assertLess(float(terms["lift_coarse_loss"]), 1e-5)
        self.assertLess(float(terms["lift_fine_loss"]), 1e-6)
        self.assertAlmostEqual(float(terms["lift_fine_weight"]), 1.0, places=5)

    def test_linear_mismatch_is_exposed_by_coarse_term(self) -> None:
        torch.manual_seed(1)
        student = torch.randn(2, 3, 4, 5)
        teacher = 0.25 + 1.5 * student
        terms = lift_velocity_terms(student, teacher)
        self.assertAlmostEqual(float(terms["lift_beta1"]), 1.5, places=4)
        self.assertAlmostEqual(float(terms["lift_beta0_abs"]), 0.25, places=4)
        self.assertLess(float(terms["lift_fine_loss"]), 1e-6)
        self.assertGreater(float(terms["lift_coarse_loss"]), 0.7)

    def test_objective_backpropagates_to_student(self) -> None:
        torch.manual_seed(2)
        student = torch.randn(2, 3, 4, 5, requires_grad=True)
        teacher = torch.randn_like(student)
        objective = lift_velocity_distillation_objective(
            student,
            teacher,
            velocity_mse_weight=1.0,
            velocity_cosine_weight=1.0,
            lift_weight=1.0,
        )
        objective["loss"].backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
