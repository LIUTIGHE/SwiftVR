"""CPU-only contract tests for M11B-S1-D0 diagnostic measurements."""
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.m11b_s1_d0 import diagnose_frames, effective_tgrow, parse_roi, read_tgrow


class M11BS1D0Tests(unittest.TestCase):
    def test_parse_roi(self):
        self.assertEqual(parse_roi("2,4,20,30"), (2, 4, 20, 30))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_roi("0,-1,10,10")

    def test_effective_tgrow_two_subphases(self):
        weight = np.zeros((2, 2, 3, 1, 1), dtype=np.float32)
        weight[0, 0, 0, 0, 0] = 1.0
        weight[0, 0, 1, 0, 0] = 2.0
        weight[0, 0, 2, 0, 0] = 4.0
        result = effective_tgrow(weight)
        # For one repeated input: first=(middle+right)=6, second=(left+middle)=3.
        self.assertGreater(result["relative_transfer_frobenius_gap"], 0)
        self.assertAlmostEqual(result["mean_abs_dc_gain_gap"], 1.5)
        self.assertEqual(result["channels_out"], 2)

    def test_tgrow_weight_file_loading(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "model.safetensors"
            save_file({
                "decoder.7.conv3d.weight": torch.ones(2, 2, 3, 1, 1),
                "decoder.13.conv3d.weight": torch.ones(2, 2, 3, 1, 1),
                "decoder.19.conv1.weight": torch.zeros(2, 2, 3, 1, 1),
            }, str(weights))
            layers = read_tgrow(weights)
            self.assertEqual(len(layers), 2)
            for layer in layers:
                self.assertAlmostEqual(layer["relative_transfer_frobenius_gap"], 0.0)

    def test_png_phase_alignment_and_gt_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ("a1", "original", "gt"):
                (root / sub).mkdir()
            for index in range(13):  # canonical 4k+1
                p = (index + 3) % 4
                original = np.full((24, 32, 3), 100, dtype=np.uint8)
                a1 = np.full((24, 32, 3), 120 if p == 0 else 100, dtype=np.uint8)
                name = f"{index:08d}.png"
                Image.fromarray(original).save(root / "original" / name)
                Image.fromarray(original).save(root / "gt" / name)
                Image.fromarray(a1).save(root / "a1" / name)
            args = argparse.Namespace(
                a1=root / "a1", original=root / "original", gt=root / "gt",
                output_dir=root / "report", start=0, end=12, phase_offset=3,
                roi=(2, 2, 20, 20), low_sigma=2.0, high_sigma=1.5,
                snapshots=[5],
            )
            summary = diagnose_frames(args)
            self.assertEqual([summary["by_phase"][str(p)]["count"] for p in range(4)], [3, 3, 3, 4])
            self.assertGreater(summary["by_phase"]["0"]["luma_delta_a1_minus_original"], 0.07)
            self.assertLess(abs(summary["by_phase"]["3"]["luma_delta_a1_minus_original"]), 1e-6)
            self.assertAlmostEqual(summary["by_phase"]["0"]["lf_original_gt_mae"], 0.0, places=6)
            self.assertTrue((root / "report" / "frames.csv").is_file())
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "native_roi_000005_p0.png").is_file())

    def test_reject_unmatched_frame_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ("a1", "original"):
                (root / sub).mkdir()
            frame = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
            for i in range(12):
                frame.save(root / "a1" / f"{i:08d}.png")
                frame.save(root / "original" / f"{i+1:08d}.png")
            args = argparse.Namespace(
                a1=root / "a1", original=root / "original", gt=None,
                output_dir=root / "out", start=0, end=11, phase_offset=3,
                roi=None, low_sigma=8.0, high_sigma=1.5, snapshots=[],
            )
            with self.assertRaisesRegex(ValueError, "filenames differ"):
                diagnose_frames(args)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
