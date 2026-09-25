from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.runtime_macs import RuntimeMacCounter


class RuntimeMacCounterTest(unittest.TestCase):
    def test_linear_and_conv_macs(self):
        module = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True),
            nn.Flatten(2),
        )
        linear = nn.Linear(4, 5, bias=True)
        counter = RuntimeMacCounter()
        counter.add_module("encoder", module)
        counter.add_module("transformer", linear)
        try:
            x = torch.randn(2, 3, 8, 8)
            with counter.count(reset=True):
                y = module[0](x)
                z = linear(y.permute(0, 2, 3, 1))
            expected_conv = 2 * 4 * 8 * 8 * 3 * 3 * 3
            expected_linear = 2 * 8 * 8 * 5 * 4
            self.assertEqual(counter.macs_by_type["conv2d"], expected_conv)
            self.assertEqual(counter.macs_by_type["linear"], expected_linear)
            self.assertEqual(counter.total_macs(), expected_conv + expected_linear)
            summary = counter.summary()
            self.assertAlmostEqual(
                summary["by_root_gmacs"]["encoder"], expected_conv / 1e9
            )
            self.assertAlmostEqual(
                summary["by_root_gmacs"]["transformer"], expected_linear / 1e9
            )
        finally:
            counter.close()

    def test_sdpa_self_attention_macs(self):
        q = torch.randn(2, 3, 5, 4)
        k = torch.randn(2, 3, 7, 4)
        v = torch.randn(2, 3, 7, 6)
        counter = RuntimeMacCounter()
        try:
            with counter.count(reset=True):
                F.scaled_dot_product_attention(q, k, v)
            expected_qk = 2 * 3 * 5 * 7 * 4
            expected_av = 2 * 3 * 5 * 7 * 6
            self.assertEqual(counter.macs_by_type["self_attn_qk"], expected_qk)
            self.assertEqual(counter.macs_by_type["self_attn_av"], expected_av)
            self.assertEqual(counter.calls_by_type["self_attn_qk"], 1)
            self.assertEqual(counter.calls_by_type["self_attn_av"], 1)
            self.assertFalse(counter.count_errors)
        finally:
            counter.close()

    def test_dispatch_attention_is_counted_once(self):
        import swiftvr.models.transformer as transformer_ops

        original = transformer_ops.dispatch_attention_fn

        def fake_dispatch(q, k, v, *args, **kwargs):
            return F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2)

        transformer_ops.dispatch_attention_fn = fake_dispatch
        q = torch.randn(2, 5, 3, 4)
        k = torch.randn(2, 7, 3, 4)
        v = torch.randn(2, 7, 3, 6)
        counter = RuntimeMacCounter()
        try:
            with counter.count(reset=True):
                transformer_ops.dispatch_attention_fn(q, k, v)
            expected_qk = 2 * 3 * 5 * 7 * 4
            expected_av = 2 * 3 * 5 * 7 * 6
            self.assertEqual(counter.macs_by_type["cross_attn_qk"], expected_qk)
            self.assertEqual(counter.macs_by_type["cross_attn_av"], expected_av)
            self.assertEqual(counter.macs_by_type.get("self_attn_qk", 0), 0)
            self.assertEqual(counter.macs_by_type.get("self_attn_av", 0), 0)
            self.assertFalse(counter.count_errors)
        finally:
            counter.close()
            transformer_ops.dispatch_attention_fn = original

    def test_reset_prevents_accumulation(self):
        linear = nn.Linear(4, 4, bias=False)
        counter = RuntimeMacCounter()
        counter.add_module("transformer", linear)
        x = torch.randn(2, 4)
        try:
            with counter.count(reset=True):
                linear(x)
            first = counter.total_macs()
            with counter.count(reset=True):
                linear(x)
            second = counter.total_macs()
            self.assertEqual(first, second)
        finally:
            counter.close()


if __name__ == "__main__":
    unittest.main()
