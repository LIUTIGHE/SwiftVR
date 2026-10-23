"""T1B-A/B CPU tests: all-phase teacher supervision, frequency split, gradients."""
from __future__ import annotations

import unittest

import torch

from swiftvr.training.m11b_t1b_objective import t1b_objective


class T1BObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261021)
        self.gt = torch.rand(1, 13, 3, 16, 16)
        self.teacher = self.gt.detach().clone()

    def test_a_exact_teacher_and_gt_video_has_zero_loss(self):
        output = self.gt.clone().requires_grad_(True)
        losses = t1b_objective(output, self.gt, self.teacher, variant="A")
        self.assertAlmostEqual(float(losses["loss"]), 0.0, places=6)

    def test_b_exact_teacher_and_gt_video_has_zero_loss(self):
        output = self.gt.clone().requires_grad_(True)
        losses = t1b_objective(output, self.gt, self.teacher, variant="B")
        self.assertAlmostEqual(float(losses["loss"]), 0.0, places=6)

    def test_b_does_not_inherit_teacher_constant_brightness_bias(self):
        gt=self.gt
        original=(gt+0.2).detach()
        output=gt.clone().requires_grad_(True)
        result=t1b_objective(output,gt,original,variant="B")
        self.assertLess(float(result["teacher_highfreq_l1"]), 1e-6)
        self.assertLess(float(result["gt_lowfreq_l1"]), 1e-6)
        self.assertLess(float(result["loss"]), 1e-6)
        a=t1b_objective(output,gt,original,variant="A")
        self.assertGreater(float(a["teacher_l1"]), 0.19)

    def test_b_gt_lowfreq_anchor_rejects_teacher_brightness_bias(self):
        # A can copy a constant offset present in Original. B's GT LF term
        # must reject it even when student and Original HF are identical.
        gt = self.gt
        original = gt + 0.12
        pred = original.clone().requires_grad_(True)
        b = t1b_objective(pred, gt, original.detach(), variant="B")
        self.assertLess(float(b["teacher_highfreq_l1"]), 1e-6)
        self.assertGreater(float(b["gt_lowfreq_l1"]), 0.11)
        b["loss"].backward()
        self.assertGreater(float(pred.grad.abs().sum()), 0)

    def test_teacher_hf_supervision_reaches_all_four_phases(self):
        # Base constant GT; Original has distinct checkerboard texture on
        # every phase. B should respond to P0 AND P3 without phase filtering.
        gt=torch.full((1,13,3,16,16),0.4)
        original=gt.clone()
        checker=((torch.arange(16)%2)*2-1).float().view(1,1,1,1,16)
        original += 0.08 * checker
        output=gt.clone().requires_grad_(True)
        out=t1b_objective(output,gt,original,variant="B",
                          lowfreq_gt_weight=0,temporal_weight=0)
        self.assertGreater(float(out["teacher_highfreq_l1"]),0.05)
        out["loss"].backward()
        self.assertIsNotNone(output.grad)
        self.assertGreater(float(output.grad[:,1].abs().sum()),0.0)  # P0
        self.assertGreater(float(output.grad[:,0].abs().sum()),0.0)  # P3
        for phase in range(4):
            self.assertIn(f"phase_{phase}_original_hf_energy",out)
            self.assertIn(f"phase_{phase}_hf_teacher_l1",out)
        self.assertIsNone(original.grad)

    def test_a_teacher_only_and_temporal_gt_correctness(self):
        # GT brightness can change; prediction=GT incurs no relative flicker.
        pred=self.gt.clone().requires_grad_(True)
        a=t1b_objective(pred,self.gt,self.teacher,variant="A")
        self.assertLess(float(a["gt_luma_temporal"]),1e-7)
        with torch.no_grad():
            pred[:,::2]+=0.05
        a=t1b_objective(pred,self.gt,self.teacher,variant="A")
        self.assertGreater(float(a["gt_luma_temporal"]),0.02)

    def test_original_remains_frozen_and_a_gradients_exist(self):
        original=(self.teacher+0.02).clone().requires_grad_(True)
        pred=self.gt.clone().requires_grad_(True)
        result=t1b_objective(pred,self.gt,original,variant="A")
        result["loss"].backward()
        self.assertIsNone(original.grad)
        self.assertGreater(float(pred.grad.abs().sum()),0.0)

    def test_rejects_invalid_inputs_and_weights(self):
        with self.assertRaisesRegex(ValueError,"variant"):
            t1b_objective(self.gt,self.gt,self.gt,variant="C")
        with self.assertRaisesRegex(ValueError,"weights"):
            t1b_objective(self.gt,self.gt,self.gt,variant="A",teacher_weight=-1)
        with self.assertRaisesRegex(ValueError,"odd"):
            t1b_objective(self.gt,self.gt,self.gt,variant="B",highpass_kernel=4)
        with self.assertRaisesRegex(ValueError,"LPIPS"):
            t1b_objective(self.gt,self.gt,self.gt,variant="B",gt_lpips_weight=.1)


if __name__ == "__main__":
    unittest.main()
