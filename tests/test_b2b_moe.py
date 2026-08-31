from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from swiftvr.models.transformer_prompt_free_no_time_moe import (
    SparseMoEFFN,
    WanTransformer3DModelPromptFreeNoTimeMoE,
)
from swiftvr.training.b2b_moe import (
    B2BMoESpec,
    parameter_accounting,
    partition_ffn_neurons,
    transformer_moe_shape,
)


class SparseMoEFFNTests(unittest.TestCase):
    def test_sparse_moe_forward_and_routing_accounting(self):
        torch.manual_seed(0)
        moe = SparseMoEFFN(
            16,
            shared_expert_dim=16,
            normal_expert_dim=4,
            num_experts=4,
            top_k=2,
        )
        x = torch.randn(2, 7, 16)
        y, balance = moe.forward_with_aux(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(balance.ndim, 0)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.isfinite(balance))
        stats = moe.last_router_stats()
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.token_count, 14)
        self.assertEqual(stats.assignment_count, 28)
        self.assertEqual(sum(stats.expert_counts), 28)

    def test_partition_ffn_neurons_is_disjoint_and_prioritizes_shared(self):
        score = torch.arange(64, dtype=torch.float32)
        spec = B2BMoESpec(
            hidden_dim=16,
            num_heads=2,
            head_dim=8,
            num_layers=2,
            adapter_dim=4,
            shared_expert_dim=16,
            normal_expert_dim=4,
            num_experts=4,
            top_k=2,
        )
        shared, experts = partition_ffn_neurons(score, spec)
        all_values = [set(shared.tolist())] + [set(v.tolist()) for v in experts]
        union = set().union(*all_values)
        self.assertEqual(len(union), spec.total_ffn_dim)
        for i, left in enumerate(all_values):
            for right in all_values[i + 1 :]:
                self.assertFalse(left & right)
        self.assertEqual(set(shared.tolist()), set(range(48, 64)))


class MoETransformerTests(unittest.TestCase):
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

    def test_shape_and_parameter_accounting(self):
        model = self._tiny_model()
        shape = transformer_moe_shape(model)
        self.assertEqual(shape["hidden_dim"], 16)
        self.assertEqual(shape["active_ffn_dim"], 24)
        self.assertEqual(shape["total_ffn_dim"], 32)
        accounting = parameter_accounting(model)
        self.assertLess(accounting["activated_parameters"], accounting["total_parameters"])
        self.assertGreater(accounting["activated_fraction"], 0.0)
        self.assertLess(accounting["activated_fraction"], 1.0)

    def test_model_forward_shape(self):
        torch.manual_seed(0)
        model = self._tiny_model().eval()
        x = torch.randn(1, 4, 2, 4, 4)
        with torch.no_grad():
            y = model(x).sample
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        stats = model.router_stats()
        self.assertEqual(len(stats), 2)
        self.assertTrue(all(item is not None for item in stats))

    def test_checkpoint_round_trip_preserves_moe_config(self):
        torch.manual_seed(0)
        model = self._tiny_model().eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model.save_pretrained(path, safe_serialization=True)
            loaded = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
                path,
                low_cpu_mem_usage=True,
            ).eval()
            self.assertEqual(transformer_moe_shape(loaded), transformer_moe_shape(model))
            self.assertEqual(int(loaded.config.shared_expert_dim), 16)
            self.assertEqual(int(loaded.config.normal_expert_dim), 4)
            self.assertEqual(int(loaded.config.num_experts), 4)
            self.assertEqual(int(loaded.config.top_k), 2)


if __name__ == "__main__":
    unittest.main()
