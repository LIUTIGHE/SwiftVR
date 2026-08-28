from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
