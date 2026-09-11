from __future__ import annotations

import math
import unittest

import torch

from swiftvr.models.transformer_prompt_free_no_time_moe import (
    WanTransformer3DModelPromptFreeNoTimeMoE,
)
from swiftvr.training.b2b_moe_training import forward_moe_transformer_training
from swiftvr.training.forward import prepare_prompt_free_no_time_transformer_for_training
from tools.train_b2b_moe_ta_distill_ddp import (
    LOCKED_SPEC,
    TA_CACHE_KIND,
    STAGE_A_CACHE_KIND,
    _router_metrics_from_counts,
    build_parser,
)


class MoETATrainerConfigTests(unittest.TestCase):
    def test_parser_locks_compute_matched_student(self):
        parser = build_parser()
        args = parser.parse_args([
            "--base-checkpoint", "base",
            "--student-init", "student",
            "--teacher-cache", "ta-cache",
            "--manifest", "train.jsonl",
            "--max-steps", "20",
            "--output-dir", "out",
        ])
        self.assertEqual(args.student_hidden_dim, 1024)
        self.assertEqual(args.student_num_heads, 8)
        self.assertEqual(args.student_head_dim, 128)
        self.assertEqual(args.student_ffn_dim, 1536)
        self.assertEqual(args.student_num_layers, 30)
        self.assertAlmostEqual(args.router_balance_weight, 0.01)
        self.assertEqual(TA_CACHE_KIND, "swiftvr_b2b_d1536_ta_velocity")
        self.assertEqual(STAGE_A_CACHE_KIND, "swiftvr_b2a_stage_a_teacher_velocity")
        self.assertEqual(LOCKED_SPEC.top_k, 2)
        self.assertEqual(LOCKED_SPEC.num_experts, 12)

    def test_router_metric_balanced_case(self):
        counts = [100.0] * 12
        result = _router_metrics_from_counts(counts, 1200.0, math.log(12.0))
        self.assertAlmostEqual(result["router_normalized_entropy"], 1.0, places=6)
        self.assertAlmostEqual(result["router_load_cv"], 0.0, places=7)
        self.assertAlmostEqual(result["router_min_fraction"], 1.0 / 12.0, places=7)
        self.assertAlmostEqual(result["router_max_fraction"], 1.0 / 12.0, places=7)


class MoETrainingForwardTests(unittest.TestCase):
    def _tiny_model(self):
        return WanTransformer3DModelPromptFreeNoTimeMoE(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            ffn_dim=24,
            num_layers=2,
            rope_max_seq_len=32,
            enable_swa=False,
            self_attn_window_hw=(2, 2),
            adapter_dim=4,
            shared_expert_dim=16,
            normal_expert_dim=4,
            num_experts=4,
            top_k=2,
        )

    def test_balance_loss_is_differentiable_to_router(self):
        torch.manual_seed(0)
        model = self._tiny_model().train()
        prepare_prompt_free_no_time_transformer_for_training(model, attention_backend="sdpa")
        x = torch.randn(1, 4, 2, 4, 4)
        velocity, balance = forward_moe_transformer_training(
            model, x, gradient_checkpointing=False
        )
        self.assertEqual(tuple(velocity.shape), tuple(x.shape))
        self.assertEqual(balance.ndim, 0)
        self.assertTrue(torch.isfinite(balance))
        loss = velocity.float().square().mean() + 0.01 * balance.float()
        loss.backward()
        for raw_block in model.blocks:
            block = getattr(raw_block, "_orig_mod", raw_block)
            grad = block.ffn.router.weight.grad
            self.assertIsNotNone(grad)
            assert grad is not None
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
