from __future__ import annotations

import unittest

import torch

from swiftvr.models.tiny_conditional_decoder_resize_conv import (
    ResizeConvTinyConditionalDecoder,
)
from tools import train_tiny_decoder_resize_conv_tail_recovery_ddp as tail


class TinyDecoderResizeConvTailRecoveryTests(unittest.TestCase):
    def _model(self) -> ResizeConvTinyConditionalDecoder:
        return ResizeConvTinyConditionalDecoder(
            latent_channels=48,
            condition_channels=32,
            channels=(192, 128, 64, 32),
            blocks_per_stage=(2, 2, 2, 1),
            temporal_factor=4,
            spatial_factor=16,
            patch_size=2,
            frames_to_trim=3,
            block_mode="compact",
            block_internal_channels=(80, 48, 24, 16),
            resize_mode="nearest",
        )

    def test_semantic_layout_matches_current_decoder(self):
        model = self._model()
        layout = tail._decoder_layout(model)
        self.assertEqual(layout["stage0"], (3, 4))
        self.assertEqual(layout["transition01"], (5, 6, 7))
        self.assertEqual(layout["stage1"], (8, 9))
        self.assertEqual(layout["transition12"], (10, 11, 12))
        self.assertEqual(layout["stage2"], (13, 14))
        self.assertEqual(layout["transition23"], (15, 16, 17))
        self.assertEqual(layout["stage3"], (18,))
        self.assertEqual(layout["output_relu"], 19)
        self.assertEqual(len(model.decoder), 20)

    def test_trainable_scope_freezes_early_decoder(self):
        model = self._model()
        report, groups = tail._set_tail_trainable(model)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        self.assertTrue(trainable)
        self.assertTrue(all(not name.startswith("condition_projection.") for name in trainable))
        self.assertTrue(all(not name.startswith("decoder.1.") for name in trainable))
        for prefix in ("decoder.3.", "decoder.4.", "decoder.7.", "decoder.8.", "decoder.9.", "decoder.12."):
            self.assertTrue(all(not name.startswith(prefix) for name in trainable), prefix)

        self.assertTrue(any(name.startswith("decoder.13.") for name in trainable))
        self.assertTrue(any(name.startswith("decoder.14.") for name in trainable))
        self.assertTrue(any(name.startswith("decoder.16.") for name in trainable))
        self.assertTrue(any(name.startswith("decoder.17.") for name in trainable))
        self.assertTrue(any(name.startswith("decoder.18.") for name in trainable))
        self.assertIn("output_head.weight", trainable)
        self.assertIn("output_head.bias", trainable)

        grouped = {id(parameter) for values in groups.values() for parameter in values}
        actual = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        self.assertEqual(grouped, actual)
        self.assertEqual(report["scope"], tail.TRAINABLE_SCOPE)

    def test_grouped_optimizer_uses_requested_learning_rates(self):
        model = self._model()
        _, groups = tail._set_tail_trainable(model)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

        parser = tail.build_parser()
        args = parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--init-decoder", "init",
                "--train-cache", "train_cache",
                "--val-cache", "val_cache",
                "--manifest", "train.jsonl",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
                "--stage2-learning-rate", "1e-5",
                "--transition-learning-rate", "2e-5",
                "--stage3-learning-rate", "3e-5",
                "--head-learning-rate", "1e-4",
            ]
        )
        optimizer = tail._build_grouped_adamw(groups, args)
        lrs = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(lrs["stage2"], 1e-5)
        self.assertEqual(lrs["transition23"], 2e-5)
        self.assertEqual(lrs["stage3"], 3e-5)
        self.assertEqual(lrs["output_head"], 1e-4)
        self.assertTrue(all(group["foreach"] is False for group in optimizer.param_groups))

    def test_parser_defaults_match_planned_recovery(self):
        parser = tail.build_parser()
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
        self.assertEqual(args.stage2_learning_rate, 1e-5)
        self.assertEqual(args.transition_learning_rate, 1e-5)
        self.assertEqual(args.stage3_learning_rate, 3e-5)
        self.assertEqual(args.head_learning_rate, 1e-4)


if __name__ == "__main__":
    unittest.main()
