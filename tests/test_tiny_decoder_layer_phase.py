from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from tools import diagnose_tiny_decoder_layer_phase as diag
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder


class TinyDecoderLayerPhaseTests(unittest.TestCase):
    def test_feature_checkerboard_basis_is_isolated(self):
        feature = torch.empty(1, 1, 8, 8)
        for y in range(8):
            for x in range(8):
                feature[0, 0, y, x] = 2.0 if (y + x) % 2 == 0 else -2.0

        basis, rms = diag._feature_phase2_values(feature)
        self.assertAlmostEqual(float(rms.item()), 2.0, places=6)
        self.assertAlmostEqual(float(basis["checkerboard"].item()), 2.0, places=6)
        self.assertAlmostEqual(float(basis["horizontal"].item()), 0.0, places=6)
        self.assertAlmostEqual(float(basis["vertical"].item()), 0.0, places=6)
        self.assertAlmostEqual(float(basis["dc"].item()), 0.0, places=6)

    def test_prepixel_channel_order_matches_pixelshuffle(self):
        prepixel = torch.zeros(1, 12, 3, 3)
        phase_values = (1.0, -1.0, -1.0, 1.0)
        for rgb in range(3):
            for phase, value in enumerate(phase_values):
                prepixel[:, rgb * 4 + phase] = value

        basis = diag._prepixel_phase_values(prepixel, patch_size=2)
        self.assertTrue(torch.allclose(basis["checkerboard"], torch.ones(1, 3)))
        self.assertTrue(torch.allclose(basis["horizontal"], torch.zeros(1, 3)))
        self.assertTrue(torch.allclose(basis["vertical"], torch.zeros(1, 3)))

        pixels = F.pixel_shuffle(prepixel, 2)
        expected = torch.empty_like(pixels)
        for y in range(int(pixels.shape[-2])):
            for x in range(int(pixels.shape[-1])):
                expected[..., y, x] = 1.0 if (y + x) % 2 == 0 else -1.0
        self.assertTrue(torch.equal(pixels, expected))

    def test_frozen_compact_layout_has_expected_trace_points(self):
        model = TinyConditionalDecoder(
            block_mode="compact",
            block_internal_channels=(80, 48, 24, 16),
        )
        names = diag._semantic_layer_names(model)
        self.assertEqual(names[5], "upsample1_nearest")
        self.assertEqual(names[10], "upsample2_nearest")
        self.assertEqual(names[15], "upsample3_nearest")
        self.assertEqual(names[20], "pre_pixelshuffle_conv12")


if __name__ == "__main__":
    unittest.main()
