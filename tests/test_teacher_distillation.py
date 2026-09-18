from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from swiftvr.training.distillation import (
    DeterministicTripletViewDataset,
    DistillationMetricAccumulator,
    gt_reconstruction_constraint,
    velocity_distillation_objective,
)
from swiftvr.training.distillation_visuals import export_validation_visuals
from swiftvr.training.loop import build_grad_scaler


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
        self.assertEqual(result["loss"].dtype, torch.float32)

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

    def test_half_inputs_reduce_in_fp32(self) -> None:
        teacher = torch.randn(1, 2, 1, 2, 2).half()
        student = (teacher.float() + 0.1).half()
        result = velocity_distillation_objective(student, teacher)
        for key in ("loss", "velocity_mse", "velocity_cosine"):
            self.assertEqual(result[key].dtype, torch.float32)


class GTConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = torch.zeros(2, 3, 3, 4, 4)
        self.teacher = torch.full_like(self.target, 0.2)

    def test_guard_is_zero_when_student_is_better(self) -> None:
        student = torch.full_like(self.target, 0.1)
        result = gt_reconstruction_constraint(
            student,
            self.teacher,
            self.target,
            mode="guard",
        )
        self.assertAlmostEqual(float(result["gt_pixel_guard"]), 0.0, places=7)
        self.assertAlmostEqual(float(result["gt_pixel_loss"]), 0.0, places=7)

    def test_guard_is_teacher_relative(self) -> None:
        student = torch.full_like(self.target, 0.3)
        result = gt_reconstruction_constraint(
            student,
            self.teacher,
            self.target,
            mode="guard",
        )
        self.assertAlmostEqual(float(result["gt_pixel_guard"]), 0.5, places=5)
        self.assertAlmostEqual(
            float(result["gt_pixel_violation_rate"]), 1.0, places=7
        )

    def test_direct_mode_uses_student_gt_error(self) -> None:
        student = torch.full_like(self.target, 0.3)
        result = gt_reconstruction_constraint(
            student,
            self.teacher,
            self.target,
            mode="direct",
        )
        self.assertAlmostEqual(float(result["gt_pixel_loss"]), 0.3, places=6)

    def test_gt_objective_requires_rgb_when_enabled(self) -> None:
        teacher_velocity = torch.randn(1, 2, 1, 2, 2)
        with self.assertRaises(ValueError):
            velocity_distillation_objective(
                teacher_velocity,
                teacher_velocity,
                gt_loss_mode="guard",
                gt_pixel_weight=0.1,
            )


class PrecisionAndVisualTests(unittest.TestCase):
    def test_bfloat16_disables_grad_scaler(self) -> None:
        scaler = build_grad_scaler(torch.device("cuda"), torch.bfloat16)
        self.assertFalse(scaler.is_enabled())

    def test_visual_export_writes_fixed_pngs(self) -> None:
        video = torch.linspace(0, 1, 3 * 3 * 8 * 8).reshape(3, 3, 8, 8)
        sample = {
            "record_uid": "plain:example",
            "lq_input": video,
            "target": video,
            "teacher_prediction": video,
            "student_prediction": video,
        }
        with tempfile.TemporaryDirectory() as temporary:
            report = export_validation_visuals(
                [sample],
                output_root=temporary,
                step=20,
                frame_indices=(0, 2),
                write_videos=False,
            )
            root = Path(temporary) / "validation_visuals" / "step_00000020"
            pngs = sorted(root.rglob("*.png"))
            self.assertEqual(len(pngs), 4)
            self.assertTrue((root / "metadata.json").is_file())
            self.assertEqual(len(report["samples"]), 1)


if __name__ == "__main__":
    unittest.main()
