from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from torch.utils.data import Dataset

from swiftvr.training.distillation_generalization import (
    build_cache_backed_subset,
    cache_overlap_report,
    cache_selected_indices,
    select_distillation_indices,
    selected_indices_sha256,
    validate_resume_fingerprint,
)


class SelectionTests(unittest.TestCase):
    def test_random_selection_is_deterministic_and_unique(self) -> None:
        first = select_distillation_indices(
            100, max_samples=12, mode="random", seed=7
        )
        second = select_distillation_indices(
            100, max_samples=12, mode="random", seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len(set(first)), 12)

    def test_random_seed_changes_selection(self) -> None:
        first = select_distillation_indices(
            100, max_samples=12, mode="random", seed=7
        )
        second = select_distillation_indices(
            100, max_samples=12, mode="random", seed=8
        )
        self.assertNotEqual(first, second)

    def test_prefix_and_all_modes(self) -> None:
        self.assertEqual(
            select_distillation_indices(
                5, max_samples=3, mode="prefix", seed=0
            ),
            (0, 1, 2),
        )
        self.assertEqual(
            select_distillation_indices(
                5, max_samples=None, mode="all", seed=0
            ),
            (0, 1, 2, 3, 4),
        )
        with self.assertRaises(ValueError):
            select_distillation_indices(
                5, max_samples=3, mode="all", seed=0
            )

    def test_cache_backed_subset_preserves_selected_order(self) -> None:
        class DummyDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, index):
                return index

        class DummyCache:
            metadata = {
                "sample_count": 3,
                "selected_indices": [7, 2, 5],
                "selected_indices_sha256": selected_indices_sha256([7, 2, 5]),
            }
            samples_by_index = {7: {}, 2: {}, 5: {}}

        subset = build_cache_backed_subset(DummyDataset(), DummyCache())
        self.assertEqual([subset[index] for index in range(len(subset))], [7, 2, 5])

    def test_cache_indices_validates_hash(self) -> None:
        indices = [4, 1, 9]
        metadata = {
            "sample_count": 3,
            "selected_indices": indices,
            "selected_indices_sha256": selected_indices_sha256(indices),
        }
        self.assertEqual(cache_selected_indices(metadata), tuple(indices))
        metadata["selected_indices_sha256"] = "bad"
        with self.assertRaises(ValueError):
            cache_selected_indices(metadata)


class CacheRelationshipTests(unittest.TestCase):
    @staticmethod
    def _metadata(pairs: list[tuple[str, str]]) -> dict[str, object]:
        return {
            "reference_checkpoint": "/teacher",
            "prompt_embedding_sha256": "prompt",
            "reae_sha256": "reae",
            "timestep": 1000.0,
            "samples": [
                {"record_uid": record_uid, "source_uid": source_uid}
                for record_uid, source_uid in pairs
            ],
        }

    def test_name_collision_without_source_overlap_is_warning_only(self) -> None:
        report = cache_overlap_report(
            self._metadata([("plain:dup", "source-train")]),
            self._metadata([("plain:dup", "source-val")]),
        )
        self.assertEqual(report["record_uid_collisions"], 1)
        self.assertEqual(report["record_uid_collision_values"], ["plain:dup"])
        self.assertEqual(report["source_overlap_records"], 0)
        self.assertEqual(report["overlap_records"], 0)

    def test_same_source_with_different_names_is_true_overlap(self) -> None:
        report = cache_overlap_report(
            self._metadata([("plain:train-name", "same-source")]),
            self._metadata([("plain:val-name", "same-source")]),
        )
        self.assertEqual(report["record_uid_collisions"], 0)
        self.assertEqual(report["source_overlap_records"], 1)
        self.assertEqual(report["source_overlap_uids"], ["same-source"])
        self.assertEqual(report["overlap_records"], 1)
        self.assertTrue(report["timestep_match"])

    def test_legacy_cache_reconstructs_source_from_full_hr_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = root / "plain_train.jsonl"
            val_manifest = root / "plain_val.jsonl"

            def write_manifest(path: Path, split: str, suffix: str) -> None:
                indices = list(range(13))
                payload = {
                    "sample_id": "same-basename",
                    "split": split,
                    "variant": "plain",
                    "frame_indices": indices,
                    "hr_frames": [
                        f"dataset/{split}/video/frame_comp0_{index:06d}_{suffix}.png"
                        for index in indices
                    ],
                    "hq_frames": [
                        f"dataset/{split}/hq/frame_comp0_{index:06d}_{suffix}.png"
                        for index in indices
                    ],
                    "lr_frames": [
                        f"dataset/{split}/lr/frame_comp0_{index:06d}_{suffix}.png"
                        for index in indices
                    ],
                }
                path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            write_manifest(train_manifest, "train", "train-content")
            write_manifest(val_manifest, "val", "val-content")

            def metadata(manifest: Path, split: str) -> dict[str, object]:
                return {
                    "reference_checkpoint": "/teacher",
                    "prompt_embedding_sha256": "prompt",
                    "reae_sha256": "reae",
                    "timestep": 1000.0,
                    "manifests": [str(manifest)],
                    "split": split,
                    "clip_length": 13,
                    "views_per_record": 1,
                    "base_record_count": 1,
                    "samples": [
                        {
                            "record_uid": "plain:same-basename",
                            "sample_id": "same-basename",
                            "variant": "plain",
                            "distillation_index": 0,
                        }
                    ],
                }

            report = cache_overlap_report(
                metadata(train_manifest, "train"),
                metadata(val_manifest, "val"),
                train_path_root=root,
                val_path_root=root,
            )
            self.assertEqual(report["record_uid_collisions"], 1)
            self.assertEqual(report["source_overlap_records"], 0)

    def test_resume_fingerprint_detects_change(self) -> None:
        validate_resume_fingerprint({"a": 1}, {"a": 1})
        with self.assertRaises(ValueError):
            validate_resume_fingerprint({"a": 1}, {"a": 2})


if __name__ == "__main__":
    unittest.main()
