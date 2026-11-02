"""CPU tests for the UltraVideo D1536 TA cache builder helpers."""

from __future__ import annotations

import unittest

from tools.build_ultravideo_ta_teacher_cache import _dataset_geometry


class _FakeDataset:
    def __init__(self):
        self.rows = [
            {
                "materialized_index": index,
                "frame_indices": list(range(13)),
                "crop_size": 128,
                "scale": 3,
                "target_height": 384,
                "target_width": 384,
                "variant": "ultravideo_degradation_v1",
            }
            for index in range(4)
        ]

    def __len__(self):
        return len(self.rows)


class UltraVideoTACacheBuilderTest(unittest.TestCase):
    def test_dataset_geometry_accepts_contiguous_merged_manifest(self):
        geometry = _dataset_geometry(_FakeDataset())
        self.assertEqual(
            geometry,
            {
                "clip_length": 13,
                "crop_size": 128,
                "scale": 3,
                "target_height": 384,
                "target_width": 384,
            },
        )

    def test_dataset_geometry_rejects_sparse_materialized_indices(self):
        dataset = _FakeDataset()
        dataset.rows[2]["materialized_index"] = 9
        with self.assertRaisesRegex(ValueError, "merged materialized manifest"):
            _dataset_geometry(dataset)


if __name__ == "__main__":
    unittest.main()
