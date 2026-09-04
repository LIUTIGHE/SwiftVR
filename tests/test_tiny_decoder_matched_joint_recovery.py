from __future__ import annotations

import unittest

import torch

from swiftvr.models.tiny_conditional_decoder_condition_bypass_resize_conv import (
    ConditionBypassResizeConvTinyConditionalDecoder,
)
from swiftvr.models.tiny_conditional_decoder_resize_conv import (
    ResizeConvTinyConditionalDecoder,
)
from tools import train_tiny_decoder_matched_joint_recovery_ddp as joint


class TinyDecoderMatchedJointRecoveryTests(unittest.TestCase):
    @staticmethod
    def _r4() -> ResizeConvTinyConditionalDecoder:
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

    @staticmethod
    def _bypass() -> ConditionBypassResizeConvTinyConditionalDecoder:
        return ConditionBypassResizeConvTinyConditionalDecoder(
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

    def test_r4_joint_scope_covers_every_parameter_once(self):
        model = self._r4()
        report, groups = joint._set_joint_trainable(model, variant="r4")
        grouped = [parameter for values in groups.values() for parameter in values]
        self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertEqual(
            {id(parameter) for parameter in grouped},
            {id(parameter) for parameter in model.parameters()},
        )
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertNotIn("condition_bypass", groups)
        self.assertEqual(report["scope"], joint.TRAINABLE_SCOPE)

    def test_bypass_joint_scope_adds_only_bypass_group(self):
        model = self._bypass()
        report, groups = joint._set_joint_trainable(model, variant="condition-bypass")
        grouped = [parameter for values in groups.values() for parameter in values]
        self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertEqual(
            {id(parameter) for parameter in grouped},
            {id(parameter) for parameter in model.parameters()},
        )
        self.assertIn("condition_bypass", groups)
        self.assertEqual(
            {name for name, parameter in model.named_parameters() if id(parameter) in {id(p) for p in groups["condition_bypass"]}},
            {"direct_condition_bypass.weight"},
        )
        self.assertEqual(report["variant"], "condition-bypass")

    def test_old_decoder_learning_rates_are_matched(self):
        parser = joint.build_parser()
        common = [
            "--base-checkpoint", "base",
            "--init-decoder", "r4",
            "--train-cache", "train_cache",
            "--val-cache", "val_cache",
            "--manifest", "train.jsonl",
            "--val-manifest", "val.jsonl",
            "--output-dir", "out",
        ]
        r4_args = parser.parse_args([*common, "--variant", "r4"])
        bypass_args = parser.parse_args([*common, "--variant", "condition-bypass"])
        r4_rates = joint._learning_rates(r4_args, "r4")
        bypass_rates = joint._learning_rates(bypass_args, "condition-bypass")
        for name, value in r4_rates.items():
            self.assertEqual(bypass_rates[name], value)
        self.assertEqual(bypass_rates["condition_bypass"], 3e-5)
        self.assertEqual(r4_rates["condition_input"], 1e-5)
        self.assertEqual(r4_rates["early"], 5e-6)
        self.assertEqual(r4_rates["stage2"], 1e-5)
        self.assertEqual(r4_rates["transition23"], 1e-5)
        self.assertEqual(r4_rates["stage3"], 2e-5)
        self.assertEqual(r4_rates["output_head"], 3e-5)

    def test_zero_bypass_is_exact_source_function(self):
        torch.manual_seed(7)
        source = self._r4().eval()
        target = self._bypass().eval()
        report = target.initialize_from_resizeconv_decoder(source)
        self.assertTrue(report["source_function_exact_at_initialization"])
        self.assertEqual(float(target.direct_condition_bypass.weight.abs().max()), 0.0)

        latent = torch.randn(1, 2, 48, 2, 2)
        condition = torch.rand(1, 8, 3, 32, 32)
        with torch.no_grad():
            expected = source(latent, condition, output_frames=5, clamp=False)
            actual = target(latent, condition, output_frames=5, clamp=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
