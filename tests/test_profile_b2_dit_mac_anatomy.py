from __future__ import annotations

import unittest

import torch.nn as nn

from tools.profile_b2_dit_mac_anatomy import (
    BLOCK_COMPONENTS,
    _split_fused_qkv_macs,
    aggregate_transformer_macs,
    parse_resolution,
)


class B2DiTMacAnatomyHelpersTest(unittest.TestCase):
    def test_parse_resolution(self):
        self.assertEqual(parse_resolution("1080x720"), (1080, 720))
        with self.assertRaises(Exception):
            parse_resolution("1080")
        with self.assertRaises(Exception):
            parse_resolution("0x720")

    def test_fused_qkv_split_is_exact(self):
        self.assertEqual(_split_fused_qkv_macs(300), (100, 100, 100))
        with self.assertRaisesRegex(ValueError, "not divisible by 3"):
            _split_fused_qkv_macs(301)

    def test_aggregate_reproduces_raw_transformer_total(self):
        inner_dim = 8
        ffn_dim = 12
        modules = {
            "transformer.blocks.0.ffn.up": nn.Linear(inner_dim, ffn_dim),
            "transformer.blocks.0.ffn.down": nn.Linear(ffn_dim, inner_dim),
        }
        raw = {
            "transformer.patch_embedding": 11,
            "transformer.blocks.0.attn1.to_qkv": 300,
            "transformer.blocks.0.self_attention.qk": 31,
            "transformer.blocks.0.self_attention.av": 37,
            "transformer.blocks.0.attn1.to_out.0": 41,
            "transformer.blocks.0.prompt_free_adapter.down": 43,
            "transformer.blocks.0.prompt_free_adapter.up": 47,
            "transformer.blocks.0.ffn.up": 53,
            "transformer.blocks.0.ffn.down": 59,
            "transformer.proj_out": 61,
        }
        totals, blocks, unclassified = aggregate_transformer_macs(
            raw,
            modules,
            num_layers=1,
            inner_dim=inner_dim,
            ffn_dim=ffn_dim,
        )
        self.assertEqual(unclassified, [])
        self.assertEqual(sum(totals.values()), sum(raw.values()))
        self.assertEqual(blocks[0]["q_proj"], 100)
        self.assertEqual(blocks[0]["k_proj"], 100)
        self.assertEqual(blocks[0]["v_proj"], 100)
        self.assertTrue(all(blocks[0][name] > 0 for name in BLOCK_COMPONENTS))

    def test_aggregate_exposes_unknown_transformer_macs(self):
        _, _, unclassified = aggregate_transformer_macs(
            {"transformer.blocks.0.unknown": 7},
            {},
            num_layers=1,
            inner_dim=8,
            ffn_dim=12,
        )
        self.assertEqual(len(unclassified), 1)
        self.assertEqual(unclassified[0]["name"], "transformer.blocks.0.unknown")


if __name__ == "__main__":
    unittest.main()
