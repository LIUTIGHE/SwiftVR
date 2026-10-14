"""CPU tests for pre-materialized UltraVideo LR distillation samples."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from swiftvr.data import UltraVideoMaterializedLRDataset
from swiftvr.training import distillation_sample_identity, prepare_training_batch


class UltraVideoMaterializedLRDatasetTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        bundle = root / "clip-a.safetensors"
        tensors = {
            "lr_00": torch.arange(13 * 3 * 8 * 8, dtype=torch.uint8).reshape(13, 3, 8, 8),
            "lr_01": torch.zeros(13, 3, 8, 8, dtype=torch.uint8),
        }
        save_file(tensors, str(bundle))
        manifest = root / "materialized_views.jsonl"
        rows = []
        for index in range(2):
            rows.append(
                {
                    "kind": "ultravideo_materialized_lr_view_v1",
                    "materialized_index": index,
                    "bundle_path": str(bundle),
                    "bundle_key": f"lr_{index:02d}",
                    "dataset": "UltraVideo",
                    "split": "train",
                    "sample_id": "clip-a",
                    "record_uid": "ultravideo:clip-a",
                    "variant": "ultravideo_degradation_v1",
                    "frame_indices": list(range(10 + index, 23 + index)),
                    "crop_top": 4 + index,
                    "crop_left": 7 + index,
                    "crop_size": 8,
                    "scale": 3,
                    "target_height": 24,
                    "target_width": 24,
                    "horizontal_flip": bool(index),
                    "vertical_flip": False,
                    "view_index": index,
                    "view_seed": 100 + index,
                    "selection_category": "detail",
                }
            )
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return manifest

    def test_dataset_returns_fixed_distillation_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._fixture(Path(tmp))
            dataset = UltraVideoMaterializedLRDataset(manifest, verify_paths=True)
            self.assertEqual(len(dataset), 2)
            sample = dataset[1]
            self.assertEqual(tuple(sample["lr"].shape), (13, 3, 8, 8))
            self.assertEqual(sample["lr"].dtype, torch.float32)
            self.assertGreaterEqual(float(sample["lr"].min()), 0.0)
            self.assertLessEqual(float(sample["lr"].max()), 1.0)
            self.assertEqual(sample["distillation_index"], 1)
            self.assertEqual(sample["distillation_view_index"], 1)
            self.assertEqual(sample["distillation_view_seed"], 101)
            self.assertEqual(sample["target_height"], 24)
            self.assertEqual(sample["target_width"], 24)

    def test_collated_sample_matches_existing_cache_identity_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._fixture(Path(tmp))
            dataset = UltraVideoMaterializedLRDataset(manifest)
            batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
            identity = distillation_sample_identity(batch, 1)
            self.assertEqual(identity["distillation_index"], 1)
            self.assertEqual(identity["record_uid"], "ultravideo:clip-a")
            self.assertEqual(identity["view_index"], 1)
            self.assertEqual(identity["view_seed"], 101)
            self.assertEqual(identity["frame_indices"], list(range(11, 24)))
            self.assertEqual(identity["crop_top"], 5)
            self.assertEqual(identity["crop_left"], 8)
            self.assertTrue(identity["horizontal_flip"])

    def test_lr_only_prepare_infers_target_geometry(self):
        batch = {
            "lr": torch.rand(2, 13, 3, 8, 8),
            "scale": torch.tensor([3, 3]),
            "target_height": torch.tensor([24, 24]),
            "target_width": torch.tensor([24, 24]),
        }
        prepared = prepare_training_batch(batch, allow_missing_target=True)
        self.assertEqual(tuple(prepared["lq_input"].shape), (2, 13, 3, 24, 24))
        self.assertIsNone(prepared["target"])
        self.assertIsNone(prepared["hq_reference"])

    def test_legacy_strict_prepare_still_requires_hr(self):
        batch = {
            "lr": torch.rand(1, 13, 3, 8, 8),
            "scale": torch.tensor([3]),
        }
        with self.assertRaisesRegex(KeyError, "Missing target tensor"):
            prepare_training_batch(batch)


if __name__ == "__main__":
    unittest.main()
