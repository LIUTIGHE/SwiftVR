from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from tools import train_b2b_aggressive_compare_ddp as compare


class B2BAggressiveCompareTest(unittest.TestCase):
    def _args(self, *extra: str):
        parser = compare.build_parser()
        return parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--student-init", "student",
                "--decoder-init", "decoder",
                "--teacher-cache", "train_cache",
                "--manifest", "train.jsonl",
                "--val-teacher-cache", "val_cache",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
                *extra,
            ]
        )

    def test_aggressive_budget(self) -> None:
        budget = compare.b2b_aggressive_compute_budget()
        self.assertAlmostEqual(budget["dit_gmac_per_frame"], 196.29195264, places=5)
        self.assertAlmostEqual(budget["decoder_gmac_per_frame"], 86.79211008, places=5)
        self.assertAlmostEqual(budget["whole_model_gflops_per_frame"], 650.72412544, places=5)

    def test_joint_branch_preserves_decoder_lr(self) -> None:
        args = self._args()
        compare._validate_args(args)
        self.assertFalse(args.freeze_decoder)
        self.assertAlmostEqual(args.decoder_learning_rate, 1e-5)

    def test_staged_branch_forces_decoder_lr_zero(self) -> None:
        args = self._args("--freeze-decoder")
        compare._validate_args(args)
        self.assertTrue(args.freeze_decoder)
        self.assertEqual(args.decoder_learning_rate, 0.0)

    def test_zero_gt_loss_is_valid_for_teacher_behavior_training(self) -> None:
        args = self._args("--gt-rgb-l1-weight", "0")
        compare._validate_args(args)
        self.assertEqual(args.gt_rgb_l1_weight, 0.0)
        self.assertGreater(args.teacher_rgb_l1_weight, 0.0)

    def test_base_best_selection_remains_teacher_behavior(self) -> None:
        source = inspect.getsource(compare.base.main)
        self.assertIn('validation["student_teacher_psnr"]', source)
        self.assertIn("best_teacher_psnr", source)
        self.assertIn("best selected by Student->Teacher RGB PSNR", source)

    def test_run_config_declares_teacher_behavior_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_config.json"
            compare._write_json(path, {"trainer": "test"})
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["deployment_priority"], "teacher_behavior")
            self.assertEqual(payload["checkpoint_selection_metric"], "student_teacher_psnr")
            self.assertEqual(payload["gt_role"], "diagnostic_or_explicit_optional_loss")


if __name__ == "__main__":
    unittest.main()
