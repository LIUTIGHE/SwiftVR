"""M12-2 CPU regression: temporal supervision, scene-cut mask, and trainer safety."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import tempfile
import unittest

import torch

from swiftvr.training.m12_slim_objective import m12_slim_objective, scene_pair_valid
from tools import train_m12_slim_temporal as trainer


class M12ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261103)
        self.gt = torch.rand(1, 13, 3, 16, 16)

    def loss(self, p, gt, t, mask=None):
        return m12_slim_objective(
            p, gt, t, teacher_lpips_weight=0.0, pair_valid=mask,
        )

    def test_exact_gt_teacher_has_zero_loss(self):
        prediction = self.gt.clone().requires_grad_(True)
        terms = self.loss(prediction, self.gt, self.gt)
        self.assertAlmostEqual(float(terms["loss"]), 0.0, places=6)

    def test_natural_motion_is_not_penalized(self):
        gt = torch.rand_like(self.gt)
        # Real frame-to-frame motion and real brightness changes are in GT.
        pred = gt.clone().requires_grad_(True)
        out = self.loss(pred, gt, gt)
        self.assertAlmostEqual(float(out["gt_relative_luma_temporal_l1"]), 0.0, places=7)

    def test_flicker_is_penalized_and_student_gradient_exists(self):
        gt = self.gt
        original = gt.detach() + 0.02
        pulse = ((torch.arange(13) % 2).float() * 2 - 1).view(1, 13, 1, 1, 1)
        prediction = (gt + 0.06 * pulse).detach().requires_grad_(True)
        terms = self.loss(prediction, gt, original)
        self.assertGreater(float(terms["gt_relative_luma_temporal_l1"]), 0.10)
        terms["loss"].backward()
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)
        self.assertIsNone(original.grad)

    def test_cross_shot_pair_excluded_from_loss(self):
        gt = torch.zeros(1, 13, 3, 16, 16)
        # The error has a single discontinuity at t=5 -> 6.
        p = gt.clone()
        p[:, 6:] += 0.2
        valid = torch.ones(1, 12, dtype=torch.bool)
        valid[:, 5] = False
        unmasked = self.loss(p, gt, gt)["gt_relative_luma_temporal_l1"]
        masked = self.loss(p, gt, gt, mask=valid)["gt_relative_luma_temporal_l1"]
        self.assertGreater(float(unmasked), 0.01)
        self.assertLess(float(masked), 1e-7)

    def test_histogram_cut_gate_preserves_same_distribution_motion(self):
        # Translated checkerboard: huge local pixel displacement, but the
        # color histogram remains unchanged. Must not be treated as a cut.
        gt = torch.zeros(1, 13, 3, 32, 32)
        checker = (torch.arange(32)[:, None] + torch.arange(32)[None, :]) % 2
        for t in range(13):
            gt[:, t] = checker.roll(shifts=t, dims=1).float()
        valid = scene_pair_valid(gt)
        self.assertTrue(bool(valid.all()))
        # A strong global color-distribution cut must be excluded.
        gt[:, 7:] = torch.tensor([1.0, 0.0, 0.0]).view(1, 1, 3, 1, 1)
        valid = scene_pair_valid(gt)
        self.assertFalse(bool(valid[:, 6].item()))

    def test_rejects_bad_mask_shape_and_missing_lpips(self):
        with self.assertRaisesRegex(ValueError, "pair_valid"):
            self.loss(self.gt, self.gt, self.gt,
                      mask=torch.ones(1, 13, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "LPIPS"):
            m12_slim_objective(self.gt, self.gt, self.gt, teacher_lpips_weight=0.1)


class M12TrainerContractTests(unittest.TestCase):
    def test_single_gpu_and_fixed_cache_defaults(self):
        args = trainer.parser().parse_args([
            "--m11b-checkpoint", "m11b",
            "--slim-init", "slim100",
            "--legacy-train", "legacy",
            "--ultra-train", "ultra",
            "--legacy-val", "val13",
            "--lpips-weights", "alex.pt",
            "--output-dir", "new_run",
        ])
        self.assertEqual(args.max_steps, 500)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertEqual(args.teacher_lpips_weight, 0.1)
        self.assertEqual(args.cut_hist_threshold, 0.65)
        self.assertEqual(args.save_every, 125)

    def test_output_tree_guard(self):
        args = trainer.parser().parse_args([
            "--m11b-checkpoint", "m11b",
            "--slim-init", "slim100",
            "--legacy-train", "legacy",
            "--ultra-train", "ultra",
            "--legacy-val", "val13",
            "--lpips-weights", "alex.pt",
            "--output-dir", "slim100/outputs",
        ])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            trainer.validate_args(args)

    def test_checkpoint_source_lineage_and_no_transformer_optimizer(self):
        code = inspect.getsource(trainer)
        for name in (
            "SlimReAEDecoder.from_pretrained",
            "M11BPairedLatentDataset",
            "scene_pair_valid",
            "m12_slim_objective",
            "source_sha256",
            "optimizer.state_dict()",
            "restore_rng",
            "output_frames=13",
        ):
            self.assertIn(name, code)
        self.assertIn("torch.optim.AdamW(student.parameters()", code)
        self.assertNotIn("WanTransformer3DModel", code)
        self.assertNotIn("teacher_optimizer", code)


if __name__ == "__main__":
    unittest.main()
