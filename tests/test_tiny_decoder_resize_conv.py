from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tools import train_tiny_decoder_resize_conv_recovery_ddp as recovery
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder
from swiftvr.models.tiny_conditional_decoder_resize_conv import (
    SURGERY_SCHEME,
    ResizeConvTinyConditionalDecoder,
)


class TinyDecoderResizeConvTests(unittest.TestCase):
    def _kwargs(self):
        return dict(
            latent_channels=4,
            condition_channels=4,
            channels=(8, 8, 8, 8),
            blocks_per_stage=(0, 0, 0, 0),
            block_mode="compact",
            block_internal_channels=(4, 4, 4, 4),
        )

    def test_forward_geometry_matches_canonical_contract(self):
        torch.manual_seed(4)
        model = ResizeConvTinyConditionalDecoder(**self._kwargs()).eval()
        latents = torch.randn(2, 2, 4, 1, 1)
        condition = torch.rand(2, 5, 3, 16, 16)
        with torch.no_grad():
            output = model(latents, condition, output_frames=5, clamp=False)
        self.assertEqual(tuple(output.shape), (2, 5, 3, 16, 16))

    def test_surgery_copies_trunk_and_phase_averages_head(self):
        torch.manual_seed(7)
        source = TinyConditionalDecoder(**self._kwargs()).eval()
        old_head = source.decoder[-1]
        self.assertIsInstance(old_head, torch.nn.Conv2d)
        with torch.no_grad():
            # Make the four PixelShuffle phases deliberately different so the
            # expected RGB phase average is unambiguous.
            for rgb in range(3):
                for phase in range(4):
                    channel = rgb * 4 + phase
                    old_head.weight[channel].fill_(10.0 * rgb + float(phase))
                    old_head.bias[channel].fill_(100.0 * rgb + float(phase))

        target = ResizeConvTinyConditionalDecoder(**self._kwargs()).eval()
        report = target.initialize_from_pixelshuffle_decoder(source)
        self.assertEqual(report["scheme"], SURGERY_SCHEME)

        expected_weight = old_head.weight.detach().reshape(
            3, 4, *old_head.weight.shape[1:]
        ).mean(dim=1)
        expected_bias = old_head.bias.detach().reshape(3, 4).mean(dim=1)
        torch.testing.assert_close(target.output_head.weight, expected_weight)
        torch.testing.assert_close(target.output_head.bias, expected_bias)

        source_state = source.state_dict()
        for name, value in target.state_dict().items():
            if name.startswith("output_head."):
                continue
            torch.testing.assert_close(value, source_state[name], rtol=0.0, atol=0.0)

    def test_pixelshuffle_checkpoint_surgery_and_roundtrip(self):
        torch.manual_seed(11)
        source = TinyConditionalDecoder(**self._kwargs()).eval()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            variant_root = root / "variant"
            source.save_pretrained(source_root)
            variant, report = ResizeConvTinyConditionalDecoder.from_pixelshuffle_pretrained(
                source_root,
                resize_mode="nearest",
                device="cpu",
                dtype=torch.float32,
            )
            self.assertEqual(report["resize_mode"], "nearest")
            variant.save_pretrained(variant_root)
            loaded = ResizeConvTinyConditionalDecoder.from_pretrained(
                variant_root, device="cpu", dtype=torch.float32
            )
            self.assertEqual(loaded.resize_mode, "nearest")
            for name, value in variant.state_dict().items():
                torch.testing.assert_close(
                    loaded.state_dict()[name], value, rtol=0.0, atol=0.0
                )

    def test_head_only_freeze_is_exact(self):
        model = ResizeConvTinyConditionalDecoder(**self._kwargs())
        report = recovery._freeze_head_only(model)
        self.assertEqual(report["scope"], "output_head_only")
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(trainable, {"output_head.weight", "output_head.bias"})
        self.assertEqual(
            report["parameter_elements"],
            sum(parameter.numel() for parameter in model.output_head.parameters()),
        )

    def test_parser_defaults_to_nearest_resize(self):
        parser = recovery.build_parser()
        args = parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--init-decoder", "source",
                "--train-cache", "train",
                "--val-cache", "val",
                "--manifest", "train.jsonl",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
            ]
        )
        self.assertEqual(args.resize_mode, "nearest")
        recovery._validate_args(args)


if __name__ == "__main__":
    unittest.main()
