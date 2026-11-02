"""CPU contracts for independent M11B-S1-T1B A/B fixed-latent trainer."""
from __future__ import annotations

import argparse
import ast
import inspect
import unittest

from tools import train_m11b_s1_t1b_decoder as trainer


class T1BTrainerTests(unittest.TestCase):
    def test_two_variants_and_250_update_defaults(self):
        parser = trainer.parser()
        cases = []
        for variant in ("A", "B"):
            a = parser.parse_args([
                "--variant", variant,
                "--m11b-checkpoint", "bootstrap",
                "--a1-init", "e99", "--legacy-train", "legacy",
                "--ultra-train", "ultra", "--legacy-val", "val",
                "--output-dir", "output"
            ])
            self.assertEqual(a.max_steps, 250)
            self.assertEqual(a.gt_lpips_weight, 0.0)
            self.assertEqual(a.gradient_accumulation_steps, 8)
            self.assertEqual(a.teacher_weight, 1.0)
            self.assertEqual(a.temporal_weight, 0.5)
            self.assertEqual(trainer.t1b_loss_kwargs(a)["variant"], variant)
            cases.append(a)
        self.assertNotEqual(cases[0].variant, cases[1].variant)

    def test_stateless_sampler_resume_and_50_50_paired_domains(self):
        kwargs=dict(legacy=15896,ultra=20088,batch_size=1,rank=0,world=1,seed=20261021)
        a=list(trainer.BalancedBatchSampler(start=0,stop=12,**kwargs))
        b=list(trainer.BalancedBatchSampler(start=5,stop=12,**kwargs))
        self.assertEqual(a[5:],b)
        self.assertEqual(sum(batch[0]<15896 for batch in a),6)
        self.assertEqual(sum(batch[0]>=15896 for batch in a),6)

    def test_freeze_transformer_and_frozen_original(self):
        src=inspect.getsource(trainer)
        tree=ast.parse(src)
        names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        self.assertNotIn("WanTransformer3DModelPromptFreeNoTimeMoE",names)
        self.assertIn("M11BPairedLatentDataset",names)
        self.assertIn("t1b_objective",names)
        self.assertIn("decode_reae_clip",names)
        self.assertIn("M9A1FactorizedReAEDecoder",names)
        self.assertIn("reae.requires_grad_(False)",src)
        self.assertIn('"tiny_decoder"',src)

    def test_variant_and_objective_identity_binds_checkpoint_resume(self):
        src=inspect.getsource(trainer.identity_config)
        self.assertIn("m11b_t1b_objective.py",src)
        self.assertIn('"kind"',src)
        self.assertIn("vars(a).items()",src)
        self.assertIn("world_size",src)

    def test_metrics_retained_from_t1(self):
        src=inspect.getsource(trainer.evaluate)
        self.assertIn("a1_gt_metrics",src)
        self.assertIn("original_gt_metrics",src)
        self.assertIn("t1b_objective",src)
        self.assertIn("VideoMetricAccumulator",src)
        self.assertIn("a1_gt_lpips_eval",src)
        self.assertIn("original_gt_lpips_eval",src)
        self.assertIn("perceptual(pred, gt)",src)
        self.assertIn("perceptual(target, gt)",src)


if __name__ == "__main__":
    unittest.main()
