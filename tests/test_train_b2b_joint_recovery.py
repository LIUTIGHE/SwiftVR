from __future__ import annotations

import inspect
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

    def test_lr_scale_warms_then_decays(self) -> None:
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
        self.assertAlmostEqual(recovery._lr_scale(args, 1), 1.0 / 50.0)
        self.assertAlmostEqual(recovery._lr_scale(args, 50), 1.0)
        self.assertLess(recovery._lr_scale(args, 500), 1.0)
        self.assertAlmostEqual(recovery._lr_scale(args, 500), 0.1)

    def test_validation_gate_is_rank_invariant(self) -> None:
        source = inspect.getsource(recovery.main)
        self.assertIn(
            "validation_configured = bool(args.val_manifest and args.val_teacher_cache)",
            source,
        )
        self.assertNotIn(
            "validation_configured = val_loader is not None and val_cache is not None",
            source,
        )

    def test_validation_uses_no_grad_not_inference_mode(self) -> None:
        source = inspect.getsource(recovery._validate_rank0)
        self.assertIn("with torch.no_grad():", source)
        self.assertNotIn("with torch.inference_mode():", source)


if __name__ == "__main__":
    unittest.main()
