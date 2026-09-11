"""CPU tests for the differentiable SwiftVR training forward."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from swiftvr.models.reae import ReAE
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training import (
    SwiftVRTrainingForward,
    WanShiftWindow2DTrainProcessor,
    encode_reae_clip,
    prepare_prompt_free_no_time_transformer_for_training,
    prepare_training_batch,
    required_pixel_multiple,
)


class _DummyVelocityModel(nn.Module):
    def __init__(self, channels: int = 8):
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=channels,
            patch_size=(1, 1, 1),
        )
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, hidden_states):
        return SimpleNamespace(sample=self.proj(hidden_states))


def _tiny_transformer() -> WanTransformer3DModelPromptFreeNoTime:
    return WanTransformer3DModelPromptFreeNoTime(
        patch_size=(1, 1, 1),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=8,
        out_channels=8,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=32,
        enable_swa=True,
        self_attn_window_hw=(2, 2),
        adapter_dim=4,
        folded_timestep=1000.0,
        time_condition_folded=True,
    ).float()


def _tiny_reae() -> ReAE:
    return ReAE(
        checkpoint_path=None,
        width_mult=1,
        patch_size=2,
        latent_channels=8,
    ).float()


def _batch(frames: int = 5, target_size: int = 16):
    torch.manual_seed(11)
    low_size = target_size // 2
    return {
        "lr": torch.rand(1, frames, 3, low_size, low_size),
        "hq": torch.rand(1, frames, 3, low_size, low_size),
        "hr": torch.rand(1, frames, 3, target_size, target_size),
    }


class TrainingForwardTest(unittest.TestCase):
    def test_prepare_batch_matches_deployment_geometry(self):
        batch = _batch()
        prepared = prepare_training_batch(batch)

        self.assertEqual(prepared["lq_input"].shape, (1, 5, 3, 16, 16))
        self.assertEqual(prepared["hq_reference"].shape, (1, 5, 3, 16, 16))
        self.assertIs(prepared["target"], batch["hr"])
        self.assertTrue(torch.isfinite(prepared["lq_input"]).all())

    def test_training_attention_is_unfused_and_alternating(self):
        transformer = _tiny_transformer()
        backend = prepare_prompt_free_no_time_transformer_for_training(
            transformer,
            attention_backend="sdpa",
        )

        self.assertEqual(backend, "sdpa")
        self.assertTrue(transformer.training)
        for index, block in enumerate(transformer.blocks):
            self.assertIsInstance(
                block.attn1.processor,
                WanShiftWindow2DTrainProcessor,
            )
            self.assertFalse(block.attn1.fused_projections)
            self.assertEqual(block.attn1._do_shift, bool(index % 2 == 1))
            self.assertFalse(hasattr(block.attn1, "to_qkv"))

    def test_fixed_time_pixel_forward_is_differentiable(self):
        reae = _tiny_reae()
        transformer = _tiny_transformer()
        model = SwiftVRTrainingForward(
            reae,
            transformer,
            attention_backend="sdpa",
        )

        output = model(_batch())
        self.assertEqual(output["prediction"].shape, (1, 5, 3, 16, 16))
        self.assertEqual(output["prediction_clamped"].shape, (1, 5, 3, 16, 16))
        self.assertEqual(output["target"].shape, (1, 5, 3, 16, 16))
        self.assertEqual(output["lq_input"].shape, (1, 5, 3, 16, 16))
        self.assertEqual(output["hq_reference"].shape, (1, 5, 3, 16, 16))
        self.assertEqual(output["z_lq"].shape, (1, 8, 2, 1, 1))
        self.assertEqual(output["velocity"].shape, output["z_lq"].shape)
        self.assertEqual(output["z_prediction"].shape, output["z_lq"].shape)
        self.assertIsNone(output["z_target"])
        self.assertIsNone(output["velocity_target"])
        self.assertTrue(torch.isfinite(output["loss"]))

        output["loss"].backward()
        self.assertIsNotNone(transformer.patch_embedding.weight.grad)
        self.assertGreater(
            transformer.patch_embedding.weight.grad.abs().sum().item(),
            0.0,
        )
        encoder_parameter = next(reae.encoder.parameters())
        decoder_parameter = next(reae.decoder.parameters())
        self.assertIsNotNone(encoder_parameter.grad)
        self.assertIsNotNone(decoder_parameter.grad)
        self.assertGreater(encoder_parameter.grad.abs().sum().item(), 0.0)
        self.assertGreater(decoder_parameter.grad.abs().sum().item(), 0.0)

    def test_optional_endpoint_velocity_target(self):
        reae = _tiny_reae()
        transformer = _DummyVelocityModel(channels=8)
        model = SwiftVRTrainingForward(
            reae,
            transformer,
            pixel_loss_weight=1.0,
            latent_loss_weight=0.5,
            training_safe_transformer=False,
            prepare_transformer=False,
        )

        output = model(_batch())
        self.assertIsNotNone(output["z_target"])
        self.assertIsNotNone(output["velocity_target"])
        self.assertEqual(output["z_target"].shape, output["z_lq"].shape)
        self.assertEqual(output["velocity_target"].shape, output["velocity"].shape)
        self.assertTrue(torch.isfinite(output["latent_velocity_mse"]))
        expected = output["pixel_l1"] + 0.5 * output["latent_velocity_mse"]
        torch.testing.assert_close(output["loss"], expected)

    def test_temporal_and_spatial_constraints_fail_early(self):
        reae = _tiny_reae()
        with self.assertRaisesRegex(ValueError, "T=4k\\+1"):
            encode_reae_clip(
                reae,
                torch.rand(1, 6, 3, 16, 16),
            )

        transformer = _DummyVelocityModel(channels=8)
        self.assertEqual(required_pixel_multiple(reae, transformer), (16, 16))
        model = SwiftVRTrainingForward(
            reae,
            transformer,
            training_safe_transformer=False,
            prepare_transformer=False,
        )
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            model(_batch(target_size=24))


if __name__ == "__main__":
    unittest.main()
