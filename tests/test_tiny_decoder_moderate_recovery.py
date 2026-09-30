from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from swiftvr.models.tiny_conditional_decoder_moderate_resize_conv import (
    MODERATE_CHANNELS,
    MODERATE_CONDITION_CHANNELS,
    MODERATE_INTERNAL_CHANNELS,
    ModerateResizeConvTinyConditionalDecoder,
)
from swiftvr.models.tiny_conditional_decoder_resize_conv import ResizeConvTinyConditionalDecoder
from swiftvr.models.tiny_decoder_sparsity import CompactMemBlock
from tools.profile_tiny_decoder_moderate_macs import estimate_resizeconv_decoder_macs


class ModerateDecoderRecoveryTest(unittest.TestCase):
    def _small_source(self) -> ResizeConvTinyConditionalDecoder:
        torch.manual_seed(11)
        return ResizeConvTinyConditionalDecoder(
            latent_channels=4,
            condition_channels=4,
            channels=(16, 12, 8, 4),
            blocks_per_stage=(1, 1, 1, 1),
            temporal_factor=4,
            spatial_factor=16,
            patch_size=2,
            frames_to_trim=3,
            block_mode="compact",
            block_internal_channels=(8, 4, 4, 2),
            resize_mode="nearest",
        ).float()

    def _small_target(self) -> ModerateResizeConvTinyConditionalDecoder:
        torch.manual_seed(29)
        return ModerateResizeConvTinyConditionalDecoder(
            latent_channels=4,
            condition_channels=8,
            channels=(16, 12, 8, 4),
            blocks_per_stage=(1, 1, 1, 1),
            temporal_factor=4,
            spatial_factor=16,
            patch_size=2,
            frames_to_trim=3,
            block_mode="compact",
            block_internal_channels=(12, 8, 8, 4),
            resize_mode="nearest",
        ).float()

    def test_widening_preserves_source_output_numerically(self) -> None:
        source = self._small_source().eval()
        target = self._small_target().eval()
        report = target.initialize_from_resizeconv_decoder(source)
        self.assertTrue(report["source_function_preserved_at_initialization"])

        torch.manual_seed(7)
        latents = torch.randn(1, 2, 4, 2, 2)
        condition = torch.rand(1, 5, 3, 32, 32)
        with torch.no_grad():
            expected = source(latents, condition, output_frames=5, clamp=False)
            actual = target(latents, condition, output_frames=5, clamp=False)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_new_capacity_is_feature_active_but_output_gated(self) -> None:
        source = self._small_source()
        target = self._small_target()
        target.initialize_from_resizeconv_decoder(source)

        old_cond = source.condition_channels
        latent = source.latent_channels
        self.assertGreater(
            float(target.condition_projection.weight[old_cond:].abs().sum().item()), 0.0
        )
        self.assertEqual(
            float(target.decoder[1].weight[:, latent + old_cond :].abs().sum().item()),
            0.0,
        )

        source_blocks = [m for m in source.modules() if isinstance(m, CompactMemBlock)]
        target_blocks = [m for m in target.modules() if isinstance(m, CompactMemBlock)]
        self.assertEqual(len(source_blocks), len(target_blocks))
        for src, dst in zip(source_blocks, target_blocks):
            old_k = src.internal_channels
            self.assertGreater(float(dst.conv[0].weight[old_k:].abs().sum().item()), 0.0)
            self.assertEqual(float(dst.conv[4].weight[:, old_k:].abs().sum().item()), 0.0)

    def test_save_load_roundtrip(self) -> None:
        source = self._small_source()
        target = self._small_target()
        target.initialize_from_resizeconv_decoder(source)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "moderate"
            target.save_pretrained(root)
            loaded = ModerateResizeConvTinyConditionalDecoder.from_pretrained(root)
            self.assertEqual(loaded.condition_channels, target.condition_channels)
            self.assertEqual(loaded.block_internal_channels, target.block_internal_channels)
            for name, value in target.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)

    def test_production_profile_is_sub100_and_reproduces_keep040(self) -> None:
        keep040 = estimate_resizeconv_decoder_macs(
            output_height=1088,
            output_width=1920,
            latent_channels=48,
            condition_channels=32,
            channels=(192, 128, 64, 32),
            internal_channels=(80, 48, 24, 16),
            blocks_per_stage=(2, 2, 2, 1),
        )
        moderate = estimate_resizeconv_decoder_macs(
            output_height=1088,
            output_width=1920,
            latent_channels=48,
            condition_channels=MODERATE_CONDITION_CHANNELS,
            channels=MODERATE_CHANNELS,
            internal_channels=MODERATE_INTERNAL_CHANNELS,
            blocks_per_stage=(2, 2, 2, 1),
        )
        self.assertAlmostEqual(keep040["total_gmac"], 47.94476544, places=6)
        self.assertAlmostEqual(moderate["total_gmac"], 66.54799872, places=6)
        self.assertLess(moderate["total_gmac"], 100.0)


if __name__ == "__main__":
    unittest.main()
