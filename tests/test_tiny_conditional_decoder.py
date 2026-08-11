from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from swiftvr.models.tiny_conditional_decoder import (
    TinyConditionalDecoder,
    pack_rgb_condition,
)
from swiftvr.streaming.tiny_decoder import StreamingTinyConditionalDecoder
from swiftvr.training.tiny_decoder import tiny_decoder_objective


class FakePerceptual(nn.Module):
    def forward_video(self, prediction, target, *, microbatch_frames=16):
        del microbatch_frames
        return (prediction.float() - target.float()).abs().mean()


def make_model() -> TinyConditionalDecoder:
    return TinyConditionalDecoder(
        latent_channels=6,
        condition_channels=4,
        channels=(8, 8, 4, 4),
        blocks_per_stage=(1, 1, 1, 0),
        temporal_factor=4,
        spatial_factor=16,
        patch_size=2,
        frames_to_trim=3,
    )


class TinyConditionalDecoderTests(unittest.TestCase):
    def test_condition_pack_matches_reae_grid(self):
        condition = torch.randn(2, 13, 3, 32, 48)
        packed = pack_rgb_condition(condition, temporal_factor=4, spatial_factor=16)
        self.assertEqual(tuple(packed.shape), (2, 4, 3072, 2, 3))

    def test_whole_clip_shape_4k_plus_1(self):
        torch.manual_seed(1)
        model = make_model().eval()
        condition = torch.randn(1, 13, 3, 32, 32)
        latents = torch.randn(1, 4, 6, 2, 2)
        with torch.no_grad():
            output = model(latents, condition, output_frames=13)
        self.assertEqual(tuple(output.shape), (1, 13, 3, 32, 32))

    def test_condition_changes_prediction(self):
        torch.manual_seed(2)
        model = make_model().eval()
        latents = torch.zeros(1, 2, 6, 2, 2)
        condition_a = torch.zeros(1, 5, 3, 32, 32)
        condition_b = torch.ones(1, 5, 3, 32, 32)
        with torch.no_grad():
            output_a = model(latents, condition_a, output_frames=5)
            output_b = model(latents, condition_b, output_frames=5)
        self.assertFalse(torch.allclose(output_a, output_b))

    def test_streaming_first_and_middle_frame_counts(self):
        torch.manual_seed(3)
        model = make_model().eval()
        stream = StreamingTinyConditionalDecoder(model)
        with torch.no_grad():
            first = stream.decode_chunk(
                torch.randn(1, 2, 6, 2, 2),
                torch.randn(1, 8, 3, 32, 32),
            )
            middle = stream.decode_chunk(
                torch.randn(1, 1, 6, 2, 2),
                torch.randn(1, 4, 3, 32, 32),
            )
        self.assertEqual(tuple(first.shape), (1, 5, 3, 32, 32))
        self.assertEqual(tuple(middle.shape), (1, 4, 3, 32, 32))

    def test_stream_reset_restores_first_chunk_semantics(self):
        torch.manual_seed(4)
        model = make_model().eval()
        stream = StreamingTinyConditionalDecoder(model)
        latents = torch.randn(1, 2, 6, 2, 2)
        condition = torch.randn(1, 8, 3, 32, 32)
        with torch.no_grad():
            first = stream.decode_chunk(latents, condition)
            stream.reset()
            repeated = stream.decode_chunk(latents, condition)
        self.assertTrue(torch.allclose(first, repeated, atol=0, rtol=0))

    def test_save_reload_is_exact(self):
        torch.manual_seed(5)
        model = make_model().eval()
        condition = torch.randn(1, 5, 3, 32, 32)
        latents = torch.randn(1, 2, 6, 2, 2)
        with torch.no_grad():
            reference = model(latents, condition, output_frames=5)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = TinyConditionalDecoder.from_pretrained(directory).eval()
            with torch.no_grad():
                actual = loaded(latents, condition, output_frames=5)
        self.assertTrue(torch.equal(reference, actual))

    def test_dual_objective_coefficients(self):
        prediction = torch.full((1, 2, 3, 4, 4), 0.25, requires_grad=True)
        gt = torch.zeros_like(prediction)
        teacher = torch.full_like(prediction, 0.5)
        perceptual = FakePerceptual()
        result = tiny_decoder_objective(
            prediction,
            gt,
            teacher,
            perceptual=perceptual,
            gt_l2_weight=1.0,
            teacher_l2_weight=1.0,
            lpips_weight=2.0,
        )
        # Both MSE terms are 0.25^2; both fake perceptual terms are 0.25.
        expected = 0.0625 + 0.0625 + 2.0 * (0.25 + 0.25)
        self.assertAlmostEqual(float(result["loss"].item()), expected, places=6)
        result["loss"].backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
