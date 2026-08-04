from __future__ import annotations

import unittest

import torch
from torch.utils.data import Dataset

from swiftvr.training.distillation import (
    DeterministicTripletViewDataset,
    DistillationMetricAccumulator,
    velocity_distillation_objective,
)


class _RandomDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "index": index,
            "random": torch.rand(4),
            "random_int": torch.randint(0, 10000, (1,)),
        }


class DeterministicViewTests(unittest.TestCase):
    def test_view_is_independent_of_access_order(self) -> None:
        dataset = DeterministicTripletViewDataset(
            _RandomDataset(), views_per_record=2, view_seed=123
        )
        first = dataset[3]
        _ = dataset[0]
        second = dataset[3]
        torch.testing.assert_close(first["random"], second["random"])
        torch.testing.assert_close(first["random_int"], second["random_int"])
        self.assertEqual(first["distillation_record_index"], 1)
        self.assertEqual(first["distillation_view_index"], 1)

    def test_distinct_views_use_distinct_seeds(self) -> None:
        dataset = DeterministicTripletViewDataset(
            _RandomDataset(), views_per_record=2, view_seed=123
        )
        self.assertNotEqual(
            dataset[0]["distillation_view_seed"],
            dataset[1]["distillation_view_seed"],
        )


class DistillationObjectiveTests(unittest.TestCase):
    def test_identical_velocity_has_zero_core_loss(self) -> None:
        teacher = torch.randn(2, 3, 4, 5, 6)
        result = velocity_distillation_objective(teacher.clone(), teacher)
        self.assertAlmostEqual(float(result["velocity_normalized_mse"]), 0.0, places=7)
        self.assertAlmostEqual(float(result["velocity_cosine"]), 1.0, places=6)
        self.assertAlmostEqual(float(result["loss"]), 0.0, places=6)

    def test_opposite_velocity_has_cosine_penalty(self) -> None:
        teacher = torch.randn(2, 3, 2, 2, 2)
        result = velocity_distillation_objective(
            -teacher,
            teacher,
            velocity_mse_weight=0.0,
            velocity_cosine_weight=1.0,
        )
        self.assertAlmostEqual(float(result["velocity_cosine"]), -1.0, places=6)
        self.assertAlmostEqual(float(result["loss"]), 2.0, places=6)

    def test_output_losses_require_predictions(self) -> None:
        teacher = torch.randn(1, 3, 2, 2, 2)
        with self.assertRaises(ValueError):
            velocity_distillation_objective(
                teacher,
                teacher,
                output_l1_weight=1.0,
            )

    def test_metric_accumulator_matches_relative_l2(self) -> None:
        teacher = torch.ones(1, 2, 1, 2, 2)
        student = torch.zeros_like(teacher)
        accumulator = DistillationMetricAccumulator()
        accumulator.update(student, teacher)
        result = accumulator.compute()
        self.assertAlmostEqual(float(result["velocity_relative_l2"]), 1.0)
        self.assertAlmostEqual(float(result["velocity_normalized_mse"]), 1.0)


if __name__ == "__main__":
    unittest.main()
