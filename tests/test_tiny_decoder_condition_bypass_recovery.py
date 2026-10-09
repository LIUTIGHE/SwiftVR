from __future__ import annotations

import unittest

import torch

from swiftvr.models.tiny_conditional_decoder_condition_bypass_resize_conv import (
    CONDITION_BYPASS_MODE,
    ConditionBypassResizeConvTinyConditionalDecoder,
)
from swiftvr.models.tiny_conditional_decoder_resize_conv import ResizeConvTinyConditionalDecoder
from tools import train_tiny_decoder_condition_bypass_recovery_ddp as trainer


class TinyDecoderConditionBypassRecoveryTests(unittest.TestCase):
    @staticmethod
    def _source() -> ResizeConvTinyConditionalDecoder:
        torch.manual_seed(7)
        return ResizeConvTinyConditionalDecoder(
            latent_channels=48,
            condition_channels=32,
            channels=(16, 12, 8, 4),
            blocks_per_stage=(1, 1, 1, 1),
            temporal_factor=4,
            spatial_factor=16,
            patch_size=2,
            frames_to_trim=3,
            block_mode="compact",
            block_internal_channels=(8, 6, 4, 2),
            resize_mode="nearest",
        ).eval()

    def test_zero_init_bypass_preserves_source_function(self):
        source = self._source()
        target = ConditionBypassResizeConvTinyConditionalDecoder(
            latent_channels=48,
            condition_channels=32,
            channels=source.channels,
            blocks_per_stage=source.blocks_per_stage,
            temporal_factor=4,
            spatial_factor=16,
            patch_size=2,
            frames_to_trim=3,
            block_mode="compact",
            block_internal_channels=source.block_internal_channels,
            resize_mode="nearest",
        ).eval()
        report = target.initialize_from_resizeconv_decoder(source)

        self.assertTrue(report["source_function_exact_at_initialization"])
        self.assertTrue(torch.count_nonzero(target.direct_condition_bypass.weight) == 0)

        latents = torch.randn(1, 2, 48, 2, 2)
        condition = torch.rand(1, 5, 3, 32, 32)
        with torch.no_grad():
            expected = source(latents, condition, output_frames=5, clamp=False)
            actual = target(latents, condition, output_frames=5, clamp=False)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_only_bypass_weight_is_trainable(self):
        source = self._source()
        target = ConditionBypassResizeConvTinyConditionalDecoder(
            latent_channels=48,
            condition_channels=32,
            channels=source.channels,
            blocks_per_stage=source.blocks_per_stage,
            temporal_factor=4,
            spatial_factor=16,
            patch_size=2,
            frames_to_trim=3,
            block_mode="compact",
            block_internal_channels=source.block_internal_channels,
            resize_mode="nearest",
        )
        target.initialize_from_resizeconv_decoder(source)
        report, parameters = trainer._set_bypass_trainable(target)
        names = {name for name, p in target.named_parameters() if p.requires_grad}
        self.assertEqual(names, {"direct_condition_bypass.weight"})
        self.assertEqual(len(parameters), 1)
        self.assertEqual(report["scope"], trainer.TRAINABLE_SCOPE)
        self.assertTrue(report["source_condition_projection_preserved"])

    def test_config_marks_bypass_variant(self):
        source = self._source()
        target = ConditionBypassResizeConvTinyConditionalDecoder(
            latent_channels=48,
            condition_channels=32,
            channels=source.channels,
            blocks_per_stage=source.blocks_per_stage,
            temporal_factor=4,
            spatial_factor=16,
            patch_size=2,
            frames_to_trim=3,
            block_mode="compact",
            block_internal_channels=source.block_internal_channels,
            resize_mode="nearest",
        )
        config = target.config_dict
        self.assertEqual(config["condition_injection"], CONDITION_BYPASS_MODE)
        self.assertEqual(config["packed_condition_channels"], 3072)
        self.assertTrue(config["source_condition_projection_preserved"])

    def test_parser_defaults(self):
        parser = trainer.build_parser()
        args = parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--init-decoder", "init",
                "--train-cache", "train_cache",
                "--val-cache", "val_cache",
                "--manifest", "train.jsonl",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
            ]
        )
        self.assertEqual(args.resize_mode, "nearest")
        self.assertEqual(args.bypass_learning_rate, 1e-5)
        trainer._validate_args(args)


if __name__ == "__main__":
    unittest.main()
