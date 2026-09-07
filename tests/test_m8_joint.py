from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.models.reae_slim_decoder import M8_DECODER76_CHANNELS, VARIANT_CHANNELS
from swiftvr.models.transformer_prompt_free_no_time_moe import WanTransformer3DModelPromptFreeNoTimeMoE
from swiftvr.training.m8_joint import M8JointLossWeights, m8_joint_objective
from tools.train_m8_joint_coadapt_ddp import _configure_trainable_scope


class M8DecoderArchitectureTests(unittest.TestCase):
    def test_decoder76_variant_is_locked(self):
        self.assertEqual(M8_DECODER76_CHANNELS, (128, 96, 64, 64))
        self.assertEqual(tuple(VARIANT_CHANNELS["m8decoder76"]), M8_DECODER76_CHANNELS)


class M8JointScopeTests(unittest.TestCase):
    def _tiny_transformer(self):
        return WanTransformer3DModelPromptFreeNoTimeMoE(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            ffn_dim=24,
            num_layers=4,
            rope_max_seq_len=32,
            enable_swa=False,
            self_attn_window_hw=(2, 2),
            adapter_dim=4,
            shared_expert_dim=16,
            normal_expert_dim=4,
            num_experts=4,
            top_k=2,
        )

    def test_early_blocks_only_adapter_router_and_tail_is_full(self):
        transformer = self._tiny_transformer()
        decoder = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU(), nn.Conv2d(4, 4, 1))
        groups, report = _configure_trainable_scope(transformer, decoder, tail_full_blocks=2)

        self.assertEqual(report["tail_start_block"], 2)
        self.assertTrue(groups["transformer_light"])
        self.assertTrue(groups["transformer_tail"])
        self.assertTrue(groups["decoder"])

        for block_index, block in enumerate(transformer.blocks):
            for name, parameter in block.named_parameters():
                if block_index >= 2:
                    self.assertTrue(parameter.requires_grad, f"tail parameter frozen: {block_index}.{name}")
                elif "prompt_free_adapter" in name or name.startswith("ffn.router"):
                    self.assertTrue(parameter.requires_grad, f"light parameter frozen: {block_index}.{name}")
                else:
                    self.assertFalse(parameter.requires_grad, f"early trunk unexpectedly trainable: {block_index}.{name}")

        self.assertTrue(all(parameter.requires_grad for parameter in decoder.parameters()))


class M8JointObjectiveTests(unittest.TestCase):
    def test_teacher_only_objective_is_finite_and_differentiable(self):
        torch.manual_seed(0)
        student_velocity = torch.randn(2, 4, 3, 5, 6, requires_grad=True)
        teacher_velocity = torch.randn_like(student_velocity)
        z_lq = torch.randn_like(student_velocity)
        student_rgb = torch.rand(2, 5, 3, 12, 14, requires_grad=True)
        teacher_rgb = torch.rand_like(student_rgb)
        router_balance = student_velocity.new_tensor(1.05, requires_grad=True)
        weights = M8JointLossWeights(teacher_lpips=0.0)

        objective = m8_joint_objective(
            student_velocity=student_velocity,
            teacher_velocity=teacher_velocity,
            z_lq=z_lq,
            student_prediction=student_rgb,
            teacher_prediction=teacher_rgb,
            router_balance_loss=router_balance,
            perceptual=None,
            weights=weights,
        )
        self.assertTrue(torch.isfinite(objective["loss"]))
        self.assertNotIn("gt", " ".join(objective.keys()).lower())
        objective["loss"].backward()
        self.assertIsNotNone(student_velocity.grad)
        self.assertIsNotNone(student_rgb.grad)
        self.assertIsNotNone(router_balance.grad)
        self.assertTrue(torch.isfinite(student_velocity.grad).all())
        self.assertTrue(torch.isfinite(student_rgb.grad).all())


if __name__ == "__main__":
    unittest.main()
