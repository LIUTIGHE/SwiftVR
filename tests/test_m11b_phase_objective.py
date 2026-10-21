"""CPU tests for M11B-S1-T1 phase-aware decoder objective."""
from __future__ import annotations
import unittest
import torch

from swiftvr.training.m11b_phase_objective import (
    _lowpass, phase_metrics, phase_objective,
)


class PhaseObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261020)
        self.gt = torch.rand(1, 13, 3, 12, 12)
        self.teacher = self.gt.clone()

    def objective(self, pred, **kwargs):
        return phase_objective(pred,self.gt,self.teacher,weight_lpips=0,**kwargs)

    def test_static_gt_zero_phase_and_flicker(self):
        pred=self.gt.clone().requires_grad_(True)
        terms=self.objective(pred)
        self.assertAlmostEqual(terms["luma_temporal_gt"].item(),0,places=7)
        self.assertAlmostEqual(terms["highfreq_gt_l1"].item(),0,places=7)
        self.assertAlmostEqual(terms["phase_highfreq_imbalance"].item(),0,places=6)
        self.assertAlmostEqual(terms["loss"].item(),0,places=6)

    def test_phase0_specific_loss_and_decoder_gradient(self):
        pred=self.gt.clone()
        # With trim=3, T=1/5/9 corresponds to phase0.
        pred[:,[1,5,9],:,:,::2]+=0.1
        pred.requires_grad_(True)
        terms=self.objective(pred)
        self.assertGreater(terms["phase_0_mse"].item(),terms["phase_3_mse"].item())
        self.assertGreater(terms["phase_0_hf_l1"].item(),terms["phase_3_hf_l1"].item())
        self.assertGreater(terms["phase_highfreq_imbalance"].item(),0)
        terms["loss"].backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(pred.grad.abs().sum().item(),0)

    def test_relative_luminance_does_not_penalize_gt_video_motion(self):
        # Real GT changes arbitrarily over time: prediction identical to GT.
        pred=self.gt.clone()
        self.assertLess(self.objective(pred)["luma_temporal_gt"].item(),1e-7)
        pred[:,::2]+=0.1
        self.assertGreater(self.objective(pred)["luma_temporal_gt"].item(),0.02)

    def test_teacher_never_receives_gradient(self):
        teacher=self.teacher.clone().requires_grad_(True)
        pred=(self.gt+0.03).clone().requires_grad_(True)
        terms=phase_objective(pred,self.gt,teacher,weight_lpips=0)
        terms["loss"].backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(pred.grad)

    def test_constant_border_remains_constant(self):
        white=torch.ones(1,13,1,12,12)
        self.assertTrue(torch.allclose(_lowpass(white,7),white))

    def test_need_explicit_lpips_when_weight_nonzero(self):
        with self.assertRaisesRegex(ValueError,"LPIPS"):
            phase_objective(self.gt,self.gt,self.gt,weight_lpips=0.1)


if __name__=="__main__":
    unittest.main()
