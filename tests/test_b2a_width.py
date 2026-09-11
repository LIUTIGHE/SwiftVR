import unittest

import torch
import torch.nn as nn

from swiftvr.training.b2a_width import (
    ActivationImportanceCollector,
    B2AWidthSpec,
    _linear_pair,
    _topk_sorted,
    expand_head_indices,
    transfer_structured_width,
    validate_b2a_teacher_shape,
)


class ToyAdapter(nn.Module):
    def __init__(self, dim: int, adapter_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, dim)


class ToyAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
        self.norm_q = nn.RMSNorm(dim)
        self.norm_k = nn.RMSNorm(dim)


class ToyBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int, adapter_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.attn1 = ToyAttention(dim, heads)
        self.prompt_free_adapter = ToyAdapter(dim, adapter_dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim))


class ToyTransformer(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int, adapter_dim: int, layers: int):
        super().__init__()
        self.patch_embedding = nn.Conv3d(3, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.blocks = nn.ModuleList(
            [ToyBlock(dim, ffn_dim, heads, adapter_dim) for _ in range(layers)]
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim))
        self.proj_out = nn.Linear(dim, 12)


class B2AWidthTest(unittest.TestCase):
    def test_topk_sorted(self):
        self.assertEqual(_topk_sorted(torch.tensor([1.0, 5.0, 3.0, 4.0]), 2).tolist(), [1, 3])

    def test_expand_complete_heads(self):
        self.assertEqual(expand_head_indices(torch.tensor([0, 2]), 2).tolist(), [0, 1, 4, 5])

    def test_hidden_importance_hook_uses_norm1_input(self):
        teacher = ToyTransformer(dim=8, ffn_dim=10, heads=4, adapter_dim=2, layers=2)
        collector = ActivationImportanceCollector(teacher)
        try:
            teacher.blocks[0].norm1(torch.randn(2, 5, 8))
            self.assertGreater(float(collector.hidden_count[0].item()), 0.0)
            self.assertEqual(float(collector.hidden_count[1].item()), 0.0)
        finally:
            collector.close()

    def test_structured_transfer_is_exact_slice(self):
        teacher = ToyTransformer(dim=8, ffn_dim=10, heads=4, adapter_dim=2, layers=2)
        student = ToyTransformer(dim=4, ffn_dim=6, heads=2, adapter_dim=2, layers=2)
        spec = B2AWidthSpec(
            hidden_dim=4,
            num_heads=2,
            ffn_dim=6,
            head_dim=2,
            num_layers=2,
            adapter_dim=2,
        )
        with torch.no_grad():
            for parameter in teacher.parameters():
                parameter.copy_(torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter))

        hidden = torch.tensor([0, 2, 5, 7])
        heads = [torch.tensor([0, 3]), torch.tensor([1, 2])]
        ffn = [
            torch.tensor([0, 1, 3, 4, 7, 9]),
            torch.tensor([2, 3, 4, 5, 6, 8]),
        ]
        transfer_structured_width(
            teacher,
            student,
            hidden_indices=hidden,
            head_indices_by_block=heads,
            ffn_indices_by_block=ffn,
            spec=spec,
        )

        self.assertTrue(
            torch.equal(
                student.patch_embedding.weight,
                teacher.patch_embedding.weight.index_select(0, hidden),
            )
        )
        q_indices = expand_head_indices(heads[0], 2)
        self.assertTrue(
            torch.equal(
                student.blocks[0].attn1.to_q.weight,
                teacher.blocks[0].attn1.to_q.weight.index_select(0, q_indices).index_select(1, hidden),
            )
        )
        teacher_up, teacher_down = _linear_pair(teacher.blocks[0].ffn, 8, 10)
        student_up, student_down = _linear_pair(student.blocks[0].ffn, 4, 6)
        self.assertTrue(
            torch.equal(
                student_up.weight,
                teacher_up.weight.index_select(0, ffn[0]).index_select(1, hidden),
            )
        )
        self.assertTrue(
            torch.equal(
                student_down.weight,
                teacher_down.weight.index_select(0, hidden).index_select(1, ffn[0]),
            )
        )

    def test_rejects_depth_change(self):
        teacher = ToyTransformer(dim=8, ffn_dim=10, heads=4, adapter_dim=2, layers=2)
        spec = B2AWidthSpec(hidden_dim=4, num_heads=2, ffn_dim=6, head_dim=2, num_layers=3, adapter_dim=2)
        with self.assertRaises(ValueError):
            validate_b2a_teacher_shape(teacher, spec)


if __name__ == "__main__":
    unittest.main()
