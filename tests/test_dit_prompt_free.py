"""CPU tests for the prompt-free streaming DiT wrapper.

Run from the repository root with:

    python -m unittest tests.test_dit_prompt_free
"""

import unittest

import torch

from swiftvr.models.transformer_prompt_free import WanTransformer3DModelPromptFree
from swiftvr.streaming.chunk import ChunkSpec, ChunkType
from swiftvr.streaming.dit import INFERENCE_TIMESTEP
from swiftvr.streaming.dit_prompt_free import StreamingDiTPromptFree


class PromptFreeStreamingDiTTest(unittest.TestCase):
    def test_first_chunk_matches_direct_forward_without_overlap(self):
        torch.manual_seed(0)
        model = self._build_tiny_model().eval()
        stream = StreamingDiTPromptFree(model, overlap=0)
        latent = torch.randn(1, 8, 2, 8, 8)
        timestep = torch.full((1,), INFERENCE_TIMESTEP)

        with torch.no_grad():
            direct = latent - model(latent, timestep).sample
            streamed = stream.denoise(latent.clone())

        torch.testing.assert_close(streamed, direct, rtol=1e-5, atol=1e-5)
        self.assertEqual(stream._g_off, latent.shape[2])

    def test_overlap_preserves_new_chunk_shape_and_reset_state(self):
        torch.manual_seed(1)
        model = self._build_tiny_model().eval()
        stream = StreamingDiTPromptFree(model, overlap=1)
        first = torch.randn(1, 8, 2, 8, 8)
        second = torch.randn(1, 8, 3, 8, 8)

        with torch.no_grad():
            first_out = stream.denoise(first)
            second_out = stream.denoise(second)

        self.assertEqual(first_out.shape, first.shape)
        self.assertEqual(second_out.shape, second.shape)
        self.assertEqual(stream._g_off, 5)
        self.assertEqual(stream._prev_lq.shape[2], 1)
        self.assertEqual(stream._prev_out.shape[2], 1)

        stream.reset()
        self.assertEqual(stream._g_off, 0)
        self.assertIsNone(stream._prev_lq)
        self.assertIsNone(stream._prev_out)

    def test_last_chunk_returns_only_new_latents(self):
        torch.manual_seed(2)
        model = self._build_tiny_model().eval()
        stream = StreamingDiTPromptFree(model, overlap=0)
        spec = ChunkSpec(
            ctype=ChunkType.LAST,
            frame_start=0,
            frame_count=5,
            b=1,
            clip_idx=0,
            is_first_decode=True,
        )
        z_new_ntchw = torch.randn(1, 2, 8, 8, 8)

        with torch.no_grad():
            output = stream.denoise_last_chunk(
                z_new_ntchw,
                spec,
                prev_dit_out_cpu=None,
                n_lat=3,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        self.assertEqual(output.shape, z_new_ntchw.shape)
        self.assertEqual(stream._g_off, 2)

    @staticmethod
    def _build_tiny_model():
        return WanTransformer3DModelPromptFree(
            patch_size=(1, 2, 2),
            num_attention_heads=4,
            attention_head_dim=8,
            in_channels=8,
            out_channels=8,
            freq_dim=16,
            ffn_dim=64,
            num_layers=2,
            rope_max_seq_len=64,
            self_attn_window_hw=(4, 4),
            adapter_dim=8,
        )


if __name__ == "__main__":
    unittest.main()
