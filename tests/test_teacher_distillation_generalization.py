from __future__ import annotations

import unittest

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
    def _metadata(uids: list[str]) -> dict[str, object]:
        return {
            "reference_checkpoint": "/teacher",
            "prompt_embedding_sha256": "prompt",
            "reae_sha256": "reae",
            "timestep": 1000.0,
            "samples": [{"record_uid": uid} for uid in uids],
        }

    def test_overlap_report(self) -> None:
        report = cache_overlap_report(
            self._metadata(["a", "b"]),
            self._metadata(["b", "c"]),
        )
        self.assertEqual(report["overlap_records"], 1)
        self.assertEqual(report["overlap_record_uids"], ["b"])
        self.assertTrue(report["timestep_match"])

    def test_resume_fingerprint_detects_change(self) -> None:
        validate_resume_fingerprint({"a": 1}, {"a": 1})
        with self.assertRaises(ValueError):
            validate_resume_fingerprint({"a": 1}, {"a": 2})


if __name__ == "__main__":
    unittest.main()
