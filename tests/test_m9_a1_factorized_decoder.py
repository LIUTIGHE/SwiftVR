from __future__ import annotations

import unittest

import torch

from swiftvr.models.m9_factorized_decoder import (
    M9_A1_GMAC_1920X1088,
    CausalTemporalAdapter,
    M9A1FactorizedReAEDecoder,
    m9_a1_compute_breakdown_1920x1088,
)
from swiftvr.streaming import StreamingTAE


class M9A1FactorizedDecoderTest(unittest.TestCase):
    def test_compute_budget_matches_declared_value(self):
        total = sum(m9_a1_compute_breakdown_1920x1088().values())
        self.assertAlmostEqual(total, M9_A1_GMAC_1920X1088, places=6)
        self.assertLess(M9_A1_GMAC_1920X1088, 76.45175808)

    def test_temporal_adapter_preserves_shape_and_starts_as_identity(self):
        torch.manual_seed(0)
        module = CausalTemporalAdapter(8).eval()
        x = torch.rand(1, 5, 8, 4, 4)
        y = module.forward_clip(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        # Input is non-negative and the zero-initialized second temporal conv
        # makes the residual branch exactly zero at initialization.
        self.assertTrue(torch.equal(y, x))

    def test_decoder_forward_contract(self):
        model = M9A1FactorizedReAEDecoder().eval()
        z = torch.randn(1, 4, 48, 2, 2)
        with torch.no_grad():
            y = model(z, output_frames=13, clamp=True)
        self.assertEqual(tuple(y.shape[:3]), (1, 13, 3))
        self.assertEqual(tuple(y.shape[-2:]), (32, 32))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    def test_whole_and_streaming_match_with_active_temporal_adapter(self):
        torch.manual_seed(1)
        model = M9A1FactorizedReAEDecoder().eval()
        adapter = model.decoder[19]
        self.assertIsInstance(adapter, CausalTemporalAdapter)
        with torch.no_grad():
            torch.nn.init.normal_(adapter.conv1.weight, mean=0.0, std=0.01)
            torch.nn.init.normal_(adapter.conv2.weight, mean=0.0, std=0.01)
            adapter.conv1.bias.zero_()
            adapter.conv2.bias.zero_()

        z = torch.randn(1, 4, 48, 2, 2)
        with torch.no_grad():
            whole = model(z, clamp=True)
            stream = StreamingTAE(model)
            first = stream.decode_chunk(z[:, :2])
            second = stream.decode_chunk(z[:, 2:])
            streamed = torch.cat([first, second], dim=1)

        self.assertEqual(tuple(streamed.shape), tuple(whole.shape))
        self.assertTrue(torch.allclose(streamed, whole, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
