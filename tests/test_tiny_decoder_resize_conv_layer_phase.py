from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from swiftvr.models.tiny_conditional_decoder_resize_conv import (
    ResizeConvTinyConditionalDecoder,
)
from tools import diagnose_tiny_decoder_resize_conv_layer_phase as diag


class ResizeConvLayerPhaseTests(unittest.TestCase):
    def _model(self) -> ResizeConvTinyConditionalDecoder:
        return ResizeConvTinyConditionalDecoder(
            latent_channels=48,
            condition_channels=32,
            channels=(16, 12, 8, 4),
            blocks_per_stage=(2, 2, 2, 1),
            temporal_factor=4,
            spatial_factor=16,
            patch_size=2,
            frames_to_trim=3,
            block_mode="compact",
            block_internal_channels=(8, 6, 4, 2),
            resize_mode="nearest",
        )

    def test_formal_resizeconv_semantic_layout(self):
        names = diag._semantic_layer_names(self._model())
        self.assertEqual(names[14], "decoder_s2_blocks_end")
        self.assertEqual(names[15], "transition23_nearest")
        self.assertEqual(names[16], "transition23_tgrow")
        self.assertEqual(names[17], "transition23_conv")
        self.assertEqual(names[18], "decoder_s3_blocks_end")
        self.assertEqual(names[19], "decoder_output_relu")

    def test_trace_contains_resize_and_rgb_head_boundaries(self):
        model = self._model()
        trace = diag._ResizeConvLayerTrace(model)
        try:
            self.assertIn("decoder_s2_blocks_end", trace.features)
            self.assertIn("transition23_conv", trace.features)
            self.assertIn("decoder_s3_blocks_end", trace.features)
            self.assertIn("resize_x2_pre_head", trace.features)
            self.assertIn("output_head_rgb", trace.features)
        finally:
            trace.close()
        self.assertEqual(trace.handles, [])

    def test_nonformal_block_layout_is_rejected(self):
        model = ResizeConvTinyConditionalDecoder(
            latent_channels=48,
            condition_channels=32,
            channels=(16, 12, 8, 4),
            blocks_per_stage=(1, 1, 1, 1),
            temporal_factor=4,
            spatial_factor=16,
            patch_size=2,
            frames_to_trim=3,
            block_mode="compact",
            block_internal_channels=(8, 6, 4, 2),
            resize_mode="nearest",
        )
        with self.assertRaisesRegex(ValueError, "blocks_per_stage"):
            diag._semantic_layer_names(model)

    def test_parser_defaults_to_full_val_audit(self):
        parser = diag.build_parser()
        args = parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--decoder", "R4=r4",
                "--val-cache", "cache",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
            ]
        )
        self.assertIsNone(args.sample_indices)
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.batch_size, 1)

    def test_pair_comparison_reports_same_layer_ratio(self):
        def payload(checker: float):
            return {
                "normalized_rms": {
                    "checkerboard": checker,
                    "horizontal": checker * 2,
                    "vertical": checker * 3,
                }
            }

        base = {
            "name": "R4",
            "layers": {
                "decoder_s2_blocks_end": payload(0.2),
                "decoder_s3_blocks_end": payload(0.1),
            },
        }
        other = {
            "name": "TailE9",
            "layers": {
                "decoder_s2_blocks_end": payload(0.1),
                "decoder_s3_blocks_end": payload(0.05),
            },
        }
        result = diag._comparison(base, other)
        self.assertAlmostEqual(
            result["layers"]["decoder_s2_blocks_end"]["checkerboard"]["ratio_other_over_base"],
            0.5,
        )
        self.assertAlmostEqual(
            result["layers"]["decoder_s3_blocks_end"]["horizontal"]["ratio_other_over_base"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
