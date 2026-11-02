from __future__ import annotations

import unittest

from tools.profile_short_clip_sweep import (
    aligned_size,
    estimate_pipeline_compute,
    latent_frames,
    latent_spatial_shape,
    parse_lengths,
)


class ShortClipSweepTest(unittest.TestCase):
    def test_default_temporal_mapping(self) -> None:
        expected = {9: 3, 17: 5, 33: 9, 49: 13, 81: 21}
        for rgb, latent in expected.items():
            self.assertEqual(latent_frames(rgb), latent)

    def test_invalid_length_rejected(self) -> None:
        for value in (0, 4, 8, 10, 16):
            with self.assertRaises(ValueError):
                latent_frames(value)
        with self.assertRaises(Exception):
            parse_lengths("9,16,81")

    def test_resolution_alignment_and_latent_grid(self) -> None:
        self.assertEqual(aligned_size(1920, 32), 1920)
        self.assertEqual(aligned_size(1080, 32), 1088)
        self.assertEqual(latent_spatial_shape(1920, 1088), (68, 120))

    def test_pipeline_estimate_charges_short_clip_padding(self) -> None:
        # Hold DiT GMAC per *latent* frame fixed only to isolate the 3-frame
        # ReAE padding effect.  Short RGB clips should then pay a larger ReAE
        # cost per emitted frame.
        encoder = 42.278
        decoder = 343.10750208
        short = estimate_pipeline_compute(
            rgb_frames=9,
            dit_clip_gmac=300.0,
            encoder_gmac_per_padded_frame=encoder,
            decoder_gmac_per_padded_frame=decoder,
        )
        long = estimate_pipeline_compute(
            rgb_frames=81,
            dit_clip_gmac=2700.0,
            encoder_gmac_per_padded_frame=encoder,
            decoder_gmac_per_padded_frame=decoder,
        )
        self.assertAlmostEqual(short["padding_overhead_ratio"], 12 / 9)
        self.assertAlmostEqual(long["padding_overhead_ratio"], 84 / 81)
        self.assertGreater(
            short["encoder_gmac_per_output_frame_estimate"],
            long["encoder_gmac_per_output_frame_estimate"],
        )
        self.assertGreater(
            short["decoder_gmac_per_output_frame_estimate"],
            long["decoder_gmac_per_output_frame_estimate"],
        )

    def test_parse_lengths_deduplicates_without_reordering(self) -> None:
        self.assertEqual(parse_lengths("9,17,9,33,81"), (9, 17, 33, 81))


if __name__ == "__main__":
    unittest.main()
