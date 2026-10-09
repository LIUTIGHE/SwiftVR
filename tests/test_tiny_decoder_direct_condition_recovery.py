from __future__ import annotations

import unittest

import torch

from swiftvr.models.tiny_conditional_decoder_direct_condition_resize_conv import (
    DIRECT_CONDITION_MODE,
    DirectConditionResizeConvTinyConditionalDecoder,
)
from swiftvr.models.tiny_conditional_decoder_resize_conv import (
    ResizeConvTinyConditionalDecoder,
)
from tools import train_tiny_decoder_direct_condition_recovery_ddp as recovery


class DirectConditionResizeConvTinyDecoderTests(unittest.TestCase):
    def _source(self) -> ResizeConvTinyConditionalDecoder:
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

    def _direct(self) -> DirectConditionResizeConvTinyConditionalDecoder:
        return DirectConditionResizeConvTinyConditionalDecoder(
            latent_channels=48,
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

    def test_direct_condition_shape_and_no_projection(self):
        model = self._direct()
        condition = torch.rand(1, 1, 3, 16, 16)
        packed = model.project_condition(condition)
        self.assertEqual(tuple(packed.shape), (1, 1, 3072, 1, 1))
        self.assertEqual(model.condition_channels, 3072)
        self.assertEqual(model.packed_condition_channels, 3072)
        self.assertFalse(hasattr(model, "condition_projection"))
        self.assertEqual(model.decoder[1].in_channels, 48 + 3072)
        self.assertEqual(model.config_dict["condition_injection"], DIRECT_CONDITION_MODE)

    def test_resizeconv_transfer_preserves_nonfusion_weights(self):
        source = self._source()
        target = self._direct()
        report = target.initialize_from_resizeconv_decoder(source)

        source_state = source.state_dict()
        target_state = target.state_dict()
        for name, value in target_state.items():
            if name.startswith("decoder.1."):
                continue
            self.assertTrue(torch.equal(value, source_state[name]), name)

        self.assertFalse(report["condition_fold_function_exact"])
        self.assertEqual(report["packed_condition_channels"], 3072)
        self.assertGreater(
            report["new_input_conv_parameters"],
            report["source_input_conv_parameters"],
        )

    def test_fusion_weight_initialization_matches_linear_composition(self):
        torch.manual_seed(7)
        source = self._source()
        target = self._direct()
        target.initialize_from_resizeconv_decoder(source)

        source_input = source.decoder[1]
        projection = source.condition_projection
        expected = torch.einsum(
            "ockl,ci->oikl",
            source_input.weight[:, 48:].detach().float(),
            projection.weight.detach()[:, :, 0, 0].float(),
        )
        actual = target.decoder[1].weight[:, 48:].detach().float()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
        self.assertTrue(
            torch.equal(
                target.decoder[1].weight[:, :48],
                source_input.weight[:, :48],
            )
        )

    def test_recovery_scope_only_trains_input_fusion(self):
        model = self._direct()
        report, parameters = recovery._set_fusion_trainable(model)
        trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(trainable, {"decoder.1.weight", "decoder.1.bias"})
        self.assertEqual(report["scope"], recovery.TRAINABLE_SCOPE)
        self.assertEqual(
            {id(parameter) for parameter in parameters},
            {id(parameter) for parameter in model.decoder[1].parameters()},
        )

    def test_recovery_parser_defaults(self):
        parser = recovery.build_parser()
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
        self.assertEqual(args.fusion_learning_rate, 1e-5)
        recovery._validate_args(args)


if __name__ == "__main__":
    unittest.main()
