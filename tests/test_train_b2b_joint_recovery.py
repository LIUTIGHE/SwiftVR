from __future__ import annotations

import unittest

from tools import train_b2b_joint_recovery_ddp as recovery


class B2BJointRecoveryProtocolTest(unittest.TestCase):
    def test_recovery_defaults_are_locked(self) -> None:
        parser = recovery.build_parser()
        defaults = {action.dest: action.default for action in parser._actions}
        self.assertEqual(defaults["max_steps"], 500)
        self.assertEqual(defaults["lr_warmup_steps"], 50)
        self.assertEqual(defaults["validate_every"], 100)
        self.assertEqual(defaults["save_every"], 100)
        self.assertEqual(defaults["dtype"], "bfloat16")
        self.assertAlmostEqual(defaults["dit_learning_rate"], 2e-5)
        self.assertAlmostEqual(defaults["decoder_learning_rate"], 1e-5)
        self.assertAlmostEqual(defaults["representation_mse_weight"], 0.05)
        self.assertAlmostEqual(defaults["representation_cosine_weight"], 0.05)
        self.assertAlmostEqual(defaults["teacher_rgb_l1_weight"], 1.0)
        self.assertAlmostEqual(defaults["gt_rgb_l1_weight"], 0.5)

    def test_lr_multiplier_warms_then_decays(self) -> None:
        parser = recovery.build_parser()
        args = parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--student-init", "student",
                "--decoder-init", "decoder",
                "--teacher-cache", "train_cache",
                "--manifest", "train.jsonl",
                "--val-teacher-cache", "val_cache",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
            ]
        )
        self.assertAlmostEqual(recovery._lr_multiplier(args, 1), 1.0 / 50.0)
        self.assertAlmostEqual(recovery._lr_multiplier(args, 50), 1.0)
        self.assertLess(recovery._lr_multiplier(args, 500), 1.0)
        self.assertAlmostEqual(recovery._lr_multiplier(args, 500), 0.1)


if __name__ == "__main__":
    unittest.main()
