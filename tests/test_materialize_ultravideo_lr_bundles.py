"""CPU tests for UltraVideo cleanup and LR materialization helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str, filename: str):
    path = _REPO_ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MAT = _load_tool(
    "materialize_ultravideo_lr_bundles",
    "materialize_ultravideo_lr_bundles.py",
)


class UltraVideoLRMaterializationTest(unittest.TestCase):
    def test_degradation_parameters_are_deterministic_and_bounded(self):
        left = _MAT._degradation_parameters(20260924, "clip-a", 3)
        right = _MAT._degradation_parameters(20260924, "clip-a", 3)
        self.assertEqual(left, right)
        self.assertGreaterEqual(left["severity"], 0.0)
        self.assertLess(left["severity"], 1.0)
        self.assertGreaterEqual(left["blur_sigma"], 0.10)
        self.assertLessEqual(left["blur_sigma"], 2.10)
        self.assertGreaterEqual(left["resize_scale"], 0.50)
        self.assertLessEqual(left["resize_scale"], 1.00)
        self.assertGreaterEqual(left["jpeg_quality"], 56)
        self.assertLessEqual(left["jpeg_quality"], 95)

    def test_degradation_is_frame_seed_deterministic(self):
        image = Image.fromarray(
            np.tile(np.arange(128, dtype=np.uint8)[None, :, None], (128, 1, 3)),
            mode="RGB",
        )
        params = _MAT._degradation_parameters(20260924, "clip-a", 0)
        left = _MAT._degrade_hq(image, params, noise_seed=7)
        right = _MAT._degrade_hq(image, params, noise_seed=7)
        self.assertTrue(np.array_equal(np.asarray(left), np.asarray(right)))

    def test_clean_hq_crop_4k_geometry(self):
        raw = np.zeros((2160, 3840, 3), dtype=np.uint8)
        raw[300:684, 600:984] = 180
        row = {
            "canonical_hr_width": 3840,
            "canonical_hr_height": 2160,
            "scale": 3,
            "crop_box_hq": [100, 200, 128, 128],
        }
        hq = _MAT._clean_hq_crop(raw, row)
        self.assertEqual(hq.size, (128, 128))
        self.assertGreater(float(np.asarray(hq).mean()), 170.0)

    def test_clean_hq_crop_8k_geometry(self):
        raw = np.zeros((4320, 7680, 3), dtype=np.uint8)
        raw[600:1368, 1200:1968] = 200
        row = {
            "canonical_hr_width": 3840,
            "canonical_hr_height": 2160,
            "scale": 3,
            "crop_box_hq": [100, 200, 128, 128],
        }
        hq = _MAT._clean_hq_crop(raw, row)
        self.assertEqual(hq.size, (128, 128))
        self.assertGreater(float(np.asarray(hq).mean()), 190.0)


if __name__ == "__main__":
    unittest.main()
