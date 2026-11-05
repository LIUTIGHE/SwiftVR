"""CPU regression for M12 native fourway composition; no GPU or checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.m12_slim100_fourway import compose, _contiguous_pngs
from tools.compare_720p3x_outputs import FrameSource


def frames(directory: Path, count: int, color: tuple[int, int, int]):
    directory.mkdir(parents=True)
    for i in range(count):
        img = Image.new("RGB", (16, 16), (color[0], color[1], (color[2] + i) % 256))
        img.save(directory / f"{i:05d}.png")


class M12SlimFourwayTest(unittest.TestCase):
    def test_native_fourway_timeline_and_reference_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, aisr = root/"original", root/"aisr"
            mo, slim, inp = root/"m11b_original", root/"slim", root/"input"
            frames(original, 8, (255, 0, 0))
            frames(aisr, 16, (0, 255, 0))
            frames(mo, 8, (0, 0, 255))
            frames(slim, 8, (255, 255, 0))
            frames(inp, 8, (100, 100, 100))
            entry = {
                "frames": 8,
                "original": str(original),
                "basic": {"path": str(aisr)},
                "input": {"path": str(inp)},
                "roi": [2, 3, 8, 8],
            }
            output = root/"out"
            compose("Wetland1", entry, mo, slim, output)
            report = json.loads((output/"summary.json").read_text())
            self.assertEqual(report["frames"], 8)
            self.assertEqual(report["order"], [
                "Original SwiftVR", "AISR / TinyCNN",
                "M11-B 30k + Original", "M11-B 30k + Slim100",
            ])
            self.assertEqual(report["aisr_source_indices"], [0, 2, 4, 6, 9, 11, 13, 15])
            self.assertTrue((output/"native_fourway.mkv").is_file())
            with Image.open(output/"native_fourway_000000_p3.png") as im:
                rgb = np.asarray(im.convert("RGB"))
                self.assertEqual(im.size, (16, 72))
                # Each quadrant contains exact unresized colors from its
                # corresponding synthetic source; labels are above the ROI.
                self.assertEqual(tuple(rgb[30, 2]), (255, 0, 0))
                self.assertEqual(tuple(rgb[30, 10]), (0, 255, 0))
                self.assertEqual(tuple(rgb[66, 2]), (0, 0, 255))
                self.assertEqual(tuple(rgb[66, 10]), (255, 255, 0))
            with self.assertRaises(FileExistsError):
                compose("Wetland1", entry, mo, slim, output)

    def test_incomplete_png_sequence_is_not_silently_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"frames"
            frames(folder, 3, (10, 20, 30))
            (folder/"00001.png").unlink()
            with self.assertRaisesRegex(ValueError, "Noncontiguous"):
                _contiguous_pngs(FrameSource(folder), 3)


if __name__ == "__main__":
    unittest.main()
