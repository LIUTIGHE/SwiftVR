"""CPU tests; real source/checkpoint/dataset tests run on the user's GPU server."""
from __future__ import annotations

import copy
import importlib.util
import tempfile
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import m10_layer_search_core as core
from tools.select_m10_layers import build_parser


class ToyBlock(nn.Module):
    def __init__(self, value, shift):
        super().__init__()
        self.attn1 = nn.Module()
        self.attn1._do_shift = shift
        self.weight = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return x * self.weight + (0.7 if self.attn1._do_shift else -0.3)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock(1 + i / 10, bool(i % 2)) for i in range(6)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class LayerSelectionTests(unittest.TestCase):
    def test_historical_mask_valid(self):
        mask = [0,3,4,5,6,7,8,11,13,16,17,18,20,23,24,25,26,27,28,29]
        self.assertEqual(len(core.checked_mask(mask, keep=20)), 20)
        for bad in ([0,0,29], [29,0], [-1,0,29], [0,30], [1,29]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                core.checked_mask(bad)

    def test_grouped_split_prevents_variant_and_val_leakage(self):
        samples = [{"sample_id": f'source{i}', "record_uid": f'{i}:variant{v}',
                    "distillation_index": i * 6 + v * 3 + view}
                   for i in range(12) for v in range(2) for view in range(3)]
        val = [{"sample_id": "source4", "distillation_index": 0}]
        result = core.source_disjoint_split(samples, val, probe=2, rank=3, adapt=18, seed=9)
        a, b, c = (set(result["source_groups"][k]) for k in ("probe", "rank", "adapt"))
        self.assertFalse(a & b or a & c or b & c)
        self.assertNotIn('source4', a | b | c)
        for k in ('probe', 'rank', 'adapt'):
            chosen = [s for s in samples if s['distillation_index'] in result[k]]
            self.assertEqual(len(chosen), len(result[k]))
            self.assertTrue({s['sample_id'] for s in chosen} <= set(result['source_groups'][k]))
        self.assertEqual(result, core.source_disjoint_split(list(reversed(samples)), val,
                                                           probe=2, rank=3, adapt=18, seed=9))

    def test_split_rejects_missing_identity_and_insufficient_sources(self):
        with self.assertRaises(ValueError):
            core.source_disjoint_split([{'distillation_index': 0}], [])
        with self.assertRaises(ValueError):
            core.source_disjoint_split([{'distillation_index': 0, 'sample_id': 'a'}], [])

    def test_swaps_cross_regions_preserve_count_and_endpoints(self):
        mask = (0,1,2,3,6,9)
        candidates = core.propose_swaps(mask, {1: 1, 2: 2}, {8: 0.1, 7: 0.2}, width=2, total=10)
        self.assertIn((0,2,3,6,8,9), candidates)  # early slot transferred to late region
        for m in candidates:
            self.assertEqual(len(m), len(mask))
            self.assertEqual(m[0], 0)
            self.assertEqual(m[-1], 9)
            self.assertEqual(len(set(m) - set(mask)), 1)
        with self.assertRaises(ValueError):
            core.propose_swaps(mask, {1: float('nan')}, {8: 0.1}, total=10)

    def test_shortlist_always_contains_heuristic_even_if_screen_score_worse(self):
        old, new, third = (0,1,4), (0,2,4), (0,3,4)
        self.assertEqual(core.shortlist({old: 9, new: 1, third: 2}, old, 2), [old, new])
        with self.assertRaises(ValueError):
            core.shortlist({old: float('nan'), new: 1}, old, 2)

    def test_recovery_can_reverse_immediate_ranking(self):
        # Initial endpoint order must NOT dictate the post-recovery choice.
        records = [dict(id='heuristic', steps=10, lpips=.3, rgb_mse=.01, before=0.1),
                   dict(id='new', steps=10, lpips=.2, rgb_mse=.02, before=0.9)]
        self.assertEqual(core.recovered_winner(records)['id'], 'new')
        records[0]['steps'] = 20
        with self.assertRaises(ValueError):
            core.recovered_winner(records)

    def test_training_order_identical_across_candidates(self):
        order = core.training_order(9, 10, 4, 12)
        self.assertEqual(len(order), 40)
        self.assertEqual(order, core.training_order(9, 10, 4, 12))
        self.assertEqual(set(order[:9]), set(range(9)))

    def test_compact_view_matches_physical_export_not_parent_parity(self):
        parent = ToyModel()
        original = parent.blocks
        state = {k: v.clone() for k,v in parent.state_dict().items()}
        kept = (0,2,3,5)
        exported = copy.deepcopy(parent)
        exported.blocks = nn.ModuleList([exported.blocks[i] for i in kept])
        for i,b in enumerate(exported.blocks):
            b.attn1._do_shift = bool(i % 2)
        x = torch.tensor(2.0)
        with core.compact_block_view(parent, kept):
            self.assertEqual(len(parent.blocks), 4)
            torch.testing.assert_close(parent(x), exported(x))
            self.assertEqual([b.attn1._do_shift for b in parent.blocks], [False,True,False,True])
            self.assertIs(parent.blocks[1], original[2])
        self.assertIs(parent.blocks, original)
        self.assertEqual([b.attn1._do_shift for b in parent.blocks], [False,True,False,True,False,True])
        for key,value in parent.state_dict().items():
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)

    def test_compact_view_restores_after_exception(self):
        model = ToyModel()
        original = model.blocks
        with self.assertRaises(RuntimeError):
            with core.compact_block_view(model, (0,2,5)):
                raise RuntimeError('evaluation failed')
        self.assertIs(model.blocks, original)
        self.assertFalse(model.blocks[2].attn1._do_shift)

    def test_parser_stages_do_not_silently_train(self):
        parser = build_parser()
        a = parser.parse_args(['recover', '--work-dir', 'run'])
        self.assertEqual(a.steps, 250)
        self.assertEqual(a.accumulation, 4)
        a = parser.parse_args(['validate', '--work-dir', 'run'])
        self.assertEqual(a.stage, 'validate')

    def test_real_moe_view_materialization_and_save_load(self):
        if importlib.util.find_spec("diffusers") is None:
            self.skipTest("Full SwiftVR/diffusers dependencies not installed")
        from swiftvr.models.transformer_prompt_free_no_time_moe import WanTransformer3DModelPromptFreeNoTimeMoE
        from swiftvr.training.forward import prepare_prompt_free_no_time_transformer_for_training
        from swiftvr.training.b2b_moe_training import forward_moe_transformer_training
        from tools.build_b2b_moe_depth_init import _build_depth_student, _copy_depth_subset
        torch.manual_seed(7)
        parent = WanTransformer3DModelPromptFreeNoTimeMoE(
            patch_size=(1,2,2), num_attention_heads=2, attention_head_dim=8,
            in_channels=4, out_channels=4, ffn_dim=24, num_layers=6,
            rope_max_seq_len=32, enable_swa=True, self_attn_window_hw=(2,2),
            adapter_dim=4, shared_expert_dim=16, normal_expert_dim=4,
            num_experts=4, top_k=2, time_condition_folded=True,
        ).eval()
        prepare_prompt_free_no_time_transformer_for_training(parent, attention_backend="sdpa")
        mask = (0,2,3,5)
        x = torch.randn(1,4,2,8,8)
        with torch.no_grad(), core.compact_block_view(parent, mask):
            expected, _ = forward_moe_transformer_training(parent, x, gradient_checkpointing=False)
        student = _build_depth_student(parent, len(mask))
        _copy_depth_subset(parent, student, list(mask))
        prepare_prompt_free_no_time_transformer_for_training(student, attention_backend="sdpa")
        with torch.no_grad():
            actual, _ = forward_moe_transformer_training(student, x, gradient_checkpointing=False)
        torch.testing.assert_close(actual, expected)
        with tempfile.TemporaryDirectory() as path:
            student.save_pretrained(path, safe_serialization=True)
            loaded = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(path, local_files_only=True)
            self.assertEqual(len(loaded.blocks), 4)
            prepare_prompt_free_no_time_transformer_for_training(loaded, attention_backend="sdpa")
            with torch.no_grad():
                output, _ = forward_moe_transformer_training(loaded, x, gradient_checkpointing=False)
            torch.testing.assert_close(output, expected)

    def test_synthetic_recovery_and_fixed_decoder(self):
        torch.manual_seed(2)
        parent = ToyModel().requires_grad_(False)
        source_weights = {k: v.clone() for k,v in parent.state_dict().items()}
        decoder = nn.Linear(1, 1).requires_grad_(False)
        decoder_state = copy.deepcopy(decoder.state_dict())
        x = torch.linspace(-.5, .5, 12).reshape(-1,1)
        target = parent(x).detach()
        for mask in ((0,1,3,5), (0,2,4,5)):
            student = copy.deepcopy(parent)
            student.blocks = nn.ModuleList([student.blocks[i] for i in mask])
            for i,b in enumerate(student.blocks):
                b.attn1._do_shift = bool(i % 2)
            student.requires_grad_(True)
            optimizer = torch.optim.AdamW(student.parameters(), lr=.02, weight_decay=0)
            before = (student(x)-target).square().mean().item()
            for _ in range(40):
                optimizer.zero_grad()
                loss = (student(x)-target).square().mean()
                loss.backward()
                optimizer.step()
            self.assertLess((student(x)-target).square().mean().item(), before)
            self.assertTrue(all(p.grad is None for p in decoder.parameters()))
            for k,v in parent.state_dict().items():
                torch.testing.assert_close(v, source_weights[k], atol=0, rtol=0)
        for k,v in decoder.state_dict().items():
            torch.testing.assert_close(v, decoder_state[k], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
