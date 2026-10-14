from __future__ import annotations

import unittest

import numpy as np

from swiftvr.streaming.chunk import build_chunk_specs
from tools.compare_720p3x_outputs import _grid, _parse_crop
from tools.profile_streaming_engineering_sweep import _csv_ints


class StreamingEngineeringSweepTest(unittest.TestCase):
    def test_csv_ints(self):
        self.assertEqual(_csv_ints("8,12,24"), (8, 12, 24))
        self.assertEqual(_csv_ints("0,1,2", allow_zero=True), (0, 1, 2))
        with self.assertRaises(Exception):
            _csv_ints("0")

    def test_chunk_protocol_stays_fixed(self):
        specs = build_chunk_specs(81, 24)
        self.assertEqual([s.ctype.value for s in specs], ["first", "middle", "middle", "last"])
        self.assertEqual([s.frame_count for s in specs], [28, 24, 24, 5])
        specs8 = build_chunk_specs(81, 8)
        self.assertEqual(specs8[0].frame_count, 12)
        self.assertTrue(all(s.frame_count == 8 for s in specs8[1:-1]))


class ComparisonUtilityTest(unittest.TestCase):
    def test_crop_parser(self):
        self.assertEqual(_parse_crop("face:100,200,256,256"), ("face", 100, 200, 256, 256))
        self.assertEqual(_parse_crop("0,0,64,32"), ("crop", 0, 0, 64, 32))

    def test_grid_shape(self):
        panel = np.zeros((100, 200, 3), dtype=np.uint8)
        grid = _grid([panel, panel, panel, panel], columns=2)
        self.assertEqual(grid.shape, (200, 400, 3))


if __name__ == "__main__":
    unittest.main()
