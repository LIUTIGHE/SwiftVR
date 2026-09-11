from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from swiftvr.models import ReAE
from swiftvr.models.reae_slim_decoder import (
    AGGRESSIVE_CHANNELS,
    SLIM100_CHANNELS,
    TEACHER_CHANNELS,
    SlimReAEDecoder,
    topk_stage_indices,
)
from tools.profile_reae_slim_macs import estimate_reae_decoder_macs


class ReAESlimDecoderTest(unittest.TestCase):
    def test_topk_stage_indices_have_requested_widths_and_teacher_order(self) -> None:
        scores = [torch.arange(width, dtype=torch.float32) for width in TEACHER_CHANNELS]
        selected = topk_stage_indices(scores, AGGRESSIVE_CHANNELS)
        for values, target, teacher_width in zip(selected, AGGRESSIVE_CHANNELS, TEACHER_CHANNELS):
            self.assertEqual(len(values), target)
            self.assertEqual(tuple(sorted(values)), values)
            self.assertEqual(values[-1], teacher_width - 1)

    def test_profile_reproduces_teacher_and_targets(self) -> None:
        teacher = estimate_reae_decoder_macs(TEACHER_CHANNELS)
        slim100 = estimate_reae_decoder_macs(SLIM100_CHANNELS)
        aggressive = estimate_reae_decoder_macs(AGGRESSIVE_CHANNELS)
        self.assertAlmostEqual(teacher["total_gmac"], 343.10750208, places=6)
        self.assertAlmostEqual(slim100["total_gmac"], 98.2228992, places=6)
        self.assertAlmostEqual(aggressive["total_gmac"], 86.79211008, places=6)
        self.assertLess(slim100["total_gmac"], 100.0)
        self.assertLess(aggressive["total_gmac"], slim100["total_gmac"])

    def test_teacher_subset_initialization_and_forward(self) -> None:
        torch.manual_seed(3)
        teacher = ReAE(checkpoint_path=None, width_mult=2).float().eval()
        student = SlimReAEDecoder(channels=AGGRESSIVE_CHANNELS).float().eval()
        indices = tuple(tuple(range(width)) for width in AGGRESSIVE_CHANNELS)
        report = student.initialize_from_reae(teacher, indices, score_method="test_prefix")
        self.assertEqual(report["student_channels"], list(AGGRESSIVE_CHANNELS))
        self.assertEqual(student.decoder[1].out_channels, 256)
        self.assertEqual(student.decoder[8].out_channels, 128)
        self.assertEqual(student.decoder[14].out_channels, 64)
        self.assertEqual(student.decoder[20].out_channels, 32)
        self.assertEqual(student.decoder[22].in_channels, 32)

        latents = torch.randn(1, 1, 48, 1, 1)
        with torch.no_grad():
            output = student(latents, output_frames=1, clamp=False)
        self.assertEqual(tuple(output.shape), (1, 1, 3, 16, 16))

    def test_save_load_roundtrip(self) -> None:
        teacher = ReAE(checkpoint_path=None, width_mult=2).float().eval()
        student = SlimReAEDecoder(channels=SLIM100_CHANNELS).float().eval()
        indices = tuple(tuple(range(width)) for width in SLIM100_CHANNELS)
        student.initialize_from_reae(teacher, indices, score_method="test_prefix")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "student"
            student.save_pretrained(root)
            loaded = SlimReAEDecoder.from_pretrained(root)
            self.assertEqual(loaded.channels, student.channels)
            self.assertEqual(loaded.pruning_metadata, student.pruning_metadata)
            for name, value in student.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
