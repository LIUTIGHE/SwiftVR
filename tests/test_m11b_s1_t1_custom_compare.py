"""CPU-only synthetic controls for custom full-origin three-decoder comparison."""
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.m11b_s1_t1_custom_compare import compare


class M11BT1CustomCompareTest(unittest.TestCase):
    def test_synthetic_frame_phases_and_three_way_png(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for key in ("e99", "step250", "step500"):
                (root / key).mkdir()
            for i in range(13):
                stripes = ((np.arange(24)[None,:] % 2) * 2 - 1).astype(float)
                for key, amplitude in (("e99",35),("step250",20),("step500",10)):
                    im=np.full((24,24,3),105,dtype=float)
                    im+=stripes[:, :, None]*amplitude
                    Image.fromarray(np.uint8(im)).save(root/key/f"{i:05d}.png")
            args=argparse.Namespace(
                e99=root/"e99",step250=root/"step250",step500=root/"step500",
                output_dir=root/"out",roi=None,start=0,end=None,snapshots="1,12",
                low_sigma=4.0,high_sigma=1.2)
            summary=compare(args)
            self.assertEqual(summary["analyzed_inclusive"],[0,12])
            self.assertEqual([summary["by_phase"][str(j)]["count"] for j in range(4)],
                             [3,3,3,4])
            for j in range(4):
                self.assertLess(summary["phase_hf_energy_ratio_to_e99"]["step500"][str(j)],
                                summary["phase_hf_energy_ratio_to_e99"]["step250"][str(j)])
            self.assertTrue((root/"out"/"global_000001_p0_E99_250_500.png").is_file())
            self.assertTrue((root/"out"/"summary.json").is_file())
            self.assertTrue((root/"out"/"frames.csv").is_file())

    def test_refuses_misaligned_missing_filename(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for key in ("e99","step250","step500"):
                (root/key).mkdir()
                for i in range(8):
                    if key=="step500" and i==4: continue
                    Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save(
                        root/key/f"{i:05d}.png")
            args=argparse.Namespace(
                e99=root/"e99",step250=root/"step250",step500=root/"step500",
                output_dir=root/"out",roi=None,start=0,end=None,snapshots="",
                low_sigma=4.0,high_sigma=1.2)
            with self.assertRaisesRegex(ValueError,"filenames"):
                compare(args)


if __name__=="__main__":
    unittest.main()
