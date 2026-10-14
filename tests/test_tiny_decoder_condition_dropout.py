from __future__ import annotations

import unittest

import torch

from tools import train_tiny_decoder_condition_dropout_ddp as dropout
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder


class TinyDecoderConditionDropoutTests(unittest.TestCase):
    def _args(self, *extra: str):
        parser = dropout.build_parser()
        return parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--init-decoder", "tiny",
                "--train-cache", "train",
                "--val-cache", "val",
                "--manifest", "train.jsonl",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
                *extra,
            ]
        )

    def test_default_preserves_formal_recipe(self):
        args = self._args()
        self.assertEqual(args.condition_dropout_probability, 0.0)
        dropout._validate_args(args)

    def test_invalid_probability_is_rejected(self):
        for value in ("-0.01", "1.01"):
            args = self._args("--condition-dropout-probability", value)
            with self.assertRaisesRegex(ValueError, "condition-dropout-probability"):
                dropout._validate_args(args)

    def test_mask_is_deterministic_and_resume_independent(self):
        kwargs = dict(
            batch_size=64,
            probability=0.2,
            seed=20260812,
            epoch=7,
            batch_index=19,
            rank=2,
            device=torch.device("cpu"),
        )
        first = dropout._condition_keep_mask(**kwargs)
        second = dropout._condition_keep_mask(**kwargs)
        self.assertIsNotNone(first)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (64, 1, 1, 1, 1))

        disabled = dropout._condition_keep_mask(
            64,
            0.0,
            seed=20260812,
            epoch=7,
            batch_index=19,
            rank=2,
            device=torch.device("cpu"),
        )
        self.assertIsNone(disabled)

        all_dropped = dropout._condition_keep_mask(
            8,
            1.0,
            seed=20260812,
            epoch=0,
            batch_index=1,
            rank=0,
            device=torch.device("cpu"),
        )
        self.assertIsNotNone(all_dropped)
        self.assertEqual(int(all_dropped.sum().item()), 0)

    def test_no_mask_matches_base_forward(self):
        torch.manual_seed(3)
        kwargs = dict(
            latent_channels=4,
            condition_channels=4,
            channels=(8, 8, 8, 8),
            blocks_per_stage=(0, 0, 0, 0),
            block_mode="compact",
            block_internal_channels=(4, 4, 4, 4),
        )
        base = TinyConditionalDecoder(**kwargs).eval()
        candidate = dropout.ConditionDropoutTinyDecoder(**kwargs).eval()
        candidate.load_state_dict(base.state_dict(), strict=True)

        latents = torch.randn(2, 2, 4, 1, 1)
        condition = torch.rand(2, 5, 3, 16, 16)
        with torch.no_grad():
            expected = base(latents, condition, output_frames=5, clamp=False)
            actual = candidate(
                latents,
                condition,
                output_frames=5,
                clamp=False,
                condition_keep_mask=None,
            )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_zero_keep_mask_removes_projected_condition(self):
        torch.manual_seed(5)
        model = dropout.ConditionDropoutTinyDecoder(
            latent_channels=4,
            condition_channels=4,
            channels=(8, 8, 8, 8),
            blocks_per_stage=(0, 0, 0, 0),
            block_mode="compact",
            block_internal_channels=(4, 4, 4, 4),
        ).eval()
        latents = torch.randn(2, 2, 4, 1, 1)
        condition_a = torch.rand(2, 5, 3, 16, 16)
        condition_b = torch.rand(2, 5, 3, 16, 16)
        zero = torch.zeros(2, 1, 1, 1, 1, dtype=torch.bool)
        with torch.no_grad():
            output_a = model(
                latents,
                condition_a,
                output_frames=5,
                condition_keep_mask=zero,
            )
            output_b = model(
                latents,
                condition_b,
                output_frames=5,
                condition_keep_mask=zero,
            )
        torch.testing.assert_close(output_a, output_b, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
