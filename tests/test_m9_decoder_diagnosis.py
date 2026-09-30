from __future__ import annotations

import unittest

import torch

from swiftvr.models.reae_slim_decoder import (
    M8_DECODER76_CHANNELS,
    SlimReAEDecoder,
)
from swiftvr.streaming import StreamingTAE
from swiftvr.streaming.chunk import ChunkSpec, ChunkType
from swiftvr.training.forward import decode_reae_clip
from tools.diagnose_m9_decoder_streaming import _normalize_total_frames, _rows
from tools.profile_reae_slim_macs import estimate_reae_decoder_macs


class M9D0Test(unittest.TestCase):
    def test_frame_limit_keeps_4k_plus_1(self):
        self.assertEqual(_normalize_total_frames(100, 81), 81)
        self.assertEqual(_normalize_total_frames(80, 81), 77)

    def test_whole_and_streaming_decoder_paths_align(self):
        torch.manual_seed(0)
        model = SlimReAEDecoder(
            channels=(8, 8, 8, 8),
            latent_channels=48,
            patch_size=2,
            frames_to_trim=3,
        ).eval()
        z = torch.randn(1, 5, 48, 2, 2)
        whole = decode_reae_clip(model, z, clamp=True)

        specs = (
            ChunkSpec(ChunkType.FIRST, 0, 8, 0, 0, True),
            ChunkSpec(ChunkType.MIDDLE, 8, 8, 0, 1, False),
            ChunkSpec(ChunkType.LAST, 16, 1, 0, 2, False),
        )
        stream = StreamingTAE(model)
        chunks = [
            stream.decode_chunk_fixed(z[:, 0:2], specs[0]),
            stream.decode_chunk_fixed(z[:, 2:4], specs[1]),
            stream.decode_chunk_fixed(z[:, 4:5], specs[2]),
        ]
        streamed = torch.cat(chunks, dim=1)
        self.assertEqual(tuple(streamed.shape), tuple(whole.shape))
        self.assertTrue(torch.allclose(streamed, whole, atol=1e-6, rtol=1e-5))

        rows = _rows(
            whole,
            streamed,
            start=0,
            clip_idx=0,
            chunk_type="test",
        )
        self.assertLess(max(float(row["max_abs"]) for row in rows), 1e-5)


class M9P0Test(unittest.TestCase):
    def test_decoder76_analytical_budget_and_groups(self):
        profile = estimate_reae_decoder_macs(M8_DECODER76_CHANNELS)
        self.assertAlmostEqual(profile["total_gmac"], 76.45175808, places=6)
        self.assertAlmostEqual(
            sum(profile["groups_gmac"].values()),
            profile["total_gmac"],
            places=6,
        )
        self.assertGreater(profile["groups_percent"]["stage2_memblocks"], 35.0)
        self.assertGreater(profile["groups_percent"]["transition23"], 30.0)


if __name__ == "__main__":
    unittest.main()
