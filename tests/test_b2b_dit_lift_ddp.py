from __future__ import annotations

import unittest

import torch

from tools import train_b2b_dit_lift_ddp as wrapper


class B2BDiTLiftWrapperTest(unittest.TestCase):
    def _args(self, *extra: str):
        parser = wrapper.build_parser()
        return parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--student-init", "student",
                "--teacher-cache", "cache",
                "--manifest", "train.jsonl",
                "--max-steps", "10",
                "--output-dir", "out",
                *extra,
            ]
        )

    def test_d768_defaults_and_lift_weight(self) -> None:
        args = self._args()
        self.assertEqual(args.student_hidden_dim, 768)
        self.assertEqual(args.student_num_heads, 6)
        self.assertEqual(args.student_ffn_dim, 4080)
        self.assertEqual(args.lift_weight, 1.0)

    def test_negative_lift_weight_rejected(self) -> None:
        args = self._args("--lift-weight", "-1")
        with self.assertRaises(ValueError):
            wrapper._validate_args(args)

    def test_adapter_rejects_gt_or_rgb_training_losses(self) -> None:
        student = torch.randn(1, 2, 3, 4)
        teacher = torch.randn_like(student)
        with self.assertRaises(ValueError):
            wrapper._lift_objective(
                student,
                teacher,
                output_l1_weight=1.0,
            )


if __name__ == "__main__":
    unittest.main()
