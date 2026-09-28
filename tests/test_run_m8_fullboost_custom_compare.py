"""CPU tests for tools/run_m8_fullboost_custom_compare.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "run_m8_fullboost_custom_compare.py"
_SPEC = importlib.util.spec_from_file_location("run_m8_fullboost_custom_compare", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class RunM8FullBoostCustomCompareTest(unittest.TestCase):
    def test_parser_only_requires_input(self):
        parser = _TOOL.build_parser()
        args = parser.parse_args(["--input", "demo.mp4"])
        self.assertEqual(str(args.input), "demo.mp4")
        self.assertEqual(args.cuda_visible_devices, "7")
        self.assertEqual(args.frame_indices, "0,8,16,24,32")
        self.assertEqual(args.panel_width, 960)

    def test_default_candidates_match_current_fullboost_race(self):
        self.assertTrue(str(_TOOL.DEFAULTS["m8a"]).endswith("step_00030000"))
        self.assertTrue(str(_TOOL.DEFAULTS["ta1500"]).endswith("step_00001500"))
        self.assertTrue(str(_TOOL.DEFAULTS["ta4000"]).endswith("step_00004000"))
        self.assertTrue(str(_TOOL.DEFAULTS["stagea3500"]).endswith("step_00003500"))
        self.assertIn("epoch_099_step_00024552", str(_TOOL.DEFAULTS["decoder"]))

    def test_run_dry_run_does_not_execute(self):
        _TOOL._run(
            [sys.executable, "-c", "raise SystemExit(99)"],
            dry_run=True,
        )


    def test_auto_detail_crops_are_target_coordinates_and_non_overlapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "frames"
            image_dir.mkdir()
            frame = np.zeros((120, 200, 3), dtype=np.uint8)
            # Two separated textured areas.
            yy, xx = np.indices((40, 40))
            texture = (((xx + yy) % 2) * 255).astype(np.uint8)
            frame[10:50, 10:50, :] = texture[..., None]
            frame[60:100, 140:180, :] = texture[..., None]
            Image.fromarray(frame).save(image_dir / "00000.png")

            crops = _TOOL._auto_detail_crops(
                image_dir,
                upscale=3,
                frame_indices=[0],
                count=2,
                target_crop_size=90,
                iou_threshold=0.10,
            )
            self.assertEqual(len(crops), 2)
            parsed = []
            for value in crops:
                _, raw = value.split(":", 1)
                x, y, w, h = map(int, raw.split(","))
                self.assertEqual((w, h), (90, 90))
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + w, 600)
                self.assertLessEqual(y + h, 360)
                parsed.append((x, y, w, h))
            self.assertLessEqual(_TOOL._box_iou(parsed[0], parsed[1]), 0.10)


if __name__ == "__main__":
    unittest.main()
