from __future__ import annotations

import unittest

import torch

from swiftvr.models.transformer_prompt_free_no_time_moe import (
    WanTransformer3DModelPromptFreeNoTimeMoE,
)
from tools.build_b2b_moe_depth_init import (
    _build_depth_student,
    _copy_depth_subset,
    select_blocks_constrained,
)


class ConstrainedDepthSelectionTests(unittest.TestCase):
    def test_constraints_protect_edges_regions_and_prune_runs(self):
        residual = torch.ones(30)
        cosine = torch.zeros(30)
        # Make the first half look artificially redundant so a pure global
        # ranking would over-prune it.
        for index in range(1, 15):
            residual[index] = 0.01 + index * 1e-4
            cosine[index] = 0.999
        result = select_blocks_constrained(
            residual,
            cosine,
            keep_layers=20,
            protect_edge_blocks=1,
            region_size=6,
            min_keep_per_region=3,
            max_consecutive_pruned=2,
        )
        kept = result["kept_source_blocks"]
        pruned = result["pruned_source_blocks"]
        self.assertEqual(len(kept), 20)
        self.assertEqual(len(pruned), 10)
        self.assertIn(0, kept)
        self.assertIn(29, kept)
        self.assertLessEqual(result["observed_max_consecutive_pruned"], 2)
        for region in result["regions"]:
            self.assertGreaterEqual(len(region["kept"]), 3)


class ExactDepthCopyTests(unittest.TestCase):
    def _source(self):
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

    def test_retained_blocks_are_bit_exact(self):
        torch.manual_seed(7)
        source = self._source().eval()
        # Give each block a recognizable router value.
        with torch.no_grad():
            for index, block in enumerate(source.blocks):
                block.ffn.router.weight.fill_(float(index + 1))
        student = _build_depth_student(source, 2).eval()
        _copy_depth_subset(source, student, [0, 3])
        self.assertTrue(
            torch.equal(student.blocks[0].ffn.router.weight, source.blocks[0].ffn.router.weight)
        )
        self.assertTrue(
            torch.equal(student.blocks[1].ffn.router.weight, source.blocks[3].ffn.router.weight)
        )
        self.assertTrue(torch.equal(student.patch_embedding.weight, source.patch_embedding.weight))
        self.assertTrue(torch.equal(student.proj_out.weight, source.proj_out.weight))


if __name__ == "__main__":
    unittest.main()
