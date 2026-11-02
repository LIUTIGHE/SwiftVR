"""CPU tests for mixed legacy + UltraVideo distillation helpers."""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import ConcatDataset, Dataset

from swiftvr.training.mixed_distillation import (
    BalancedTwoDomainDistributedSampler,
    MixedTeacherVelocityCache,
    TaggedDataset,
)


class _DictDataset(Dataset):
    def __init__(self, length: int, prefix: str) -> None:
        self.length = int(length)
        self.prefix = prefix

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return {
            "value": f"{self.prefix}{index}",
            "lr": torch.zeros(3, 3, 8, 8, dtype=torch.float32),
            "frame_indices": torch.tensor([0, 1, 2], dtype=torch.int64),
            "record_uid": f"{self.prefix}:{index}",
            "sample_id": f"{self.prefix}{index}",
            "variant": self.prefix,
            "crop_top": 0,
            "crop_left": 0,
            "horizontal_flip": False,
            "vertical_flip": False,
            "distillation_index": int(index),
            "distillation_view_index": 0,
            "distillation_view_seed": 1,
            "scale": 3,
            "hr": torch.zeros(3, 3, 24, 24),
        }


class _FakeCache:
    def __init__(self, base: float) -> None:
        self.base = float(base)

    def load(self, identity, *, device, dtype):
        value = self.base + float(identity["distillation_index"])
        return torch.full((1, 1, 1, 1), value, device=device, dtype=dtype)


class MixedDistillationTest(unittest.TestCase):
    def test_tagged_dataset_adds_domain_without_mutating_source(self):
        source = _DictDataset(2, "a")
        tagged = TaggedDataset(source, "legacy")
        sample = tagged[1]
        self.assertEqual(sample["teacher_cache_domain"], "legacy")
        self.assertNotIn("teacher_cache_domain", source[1])

    def test_tagged_datasets_collate_to_identical_velocity_schema(self):
        from torch.utils.data import DataLoader

        legacy = TaggedDataset(_DictDataset(1, "a"), "legacy")
        ultra = TaggedDataset(_DictDataset(1, "b"), "ultravideo")
        dataset = ConcatDataset([legacy, ultra])
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        self.assertEqual(set(batch), {
            "lr",
            "sample_id",
            "record_uid",
            "variant",
            "frame_indices",
            "crop_top",
            "crop_left",
            "horizontal_flip",
            "vertical_flip",
            "distillation_index",
            "distillation_view_index",
            "distillation_view_seed",
            "scale",
            "target_height",
            "target_width",
            "teacher_cache_domain",
        })
        self.assertNotIn("hr", batch)
        self.assertEqual(batch["teacher_cache_domain"], ["legacy", "ultravideo"])
        self.assertTrue(torch.equal(batch["target_height"], torch.tensor([24, 24])))
        self.assertTrue(torch.equal(batch["target_width"], torch.tensor([24, 24])))

    def test_balanced_sampler_is_exact_global_50_50(self):
        left = TaggedDataset(_DictDataset(3, "a"), "legacy")
        right = TaggedDataset(_DictDataset(5, "b"), "ultravideo")
        dataset = ConcatDataset([left, right])
        sampler = BalancedTwoDomainDistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            seed=17,
        )
        indices = list(iter(sampler))
        self.assertEqual(len(indices), 10)
        left_count = sum(index < len(left) for index in indices)
        right_count = len(indices) - left_count
        self.assertEqual((left_count, right_count), (5, 5))

    def test_balanced_sampler_ddp_shards_without_overlap(self):
        left = TaggedDataset(_DictDataset(4, "a"), "legacy")
        right = TaggedDataset(_DictDataset(6, "b"), "ultravideo")
        dataset = ConcatDataset([left, right])
        rank0 = BalancedTwoDomainDistributedSampler(
            dataset, num_replicas=2, rank=0, seed=3
        )
        rank1 = BalancedTwoDomainDistributedSampler(
            dataset, num_replicas=2, rank=1, seed=3
        )
        a = list(rank0)
        b = list(rank1)
        self.assertEqual(len(a), len(b))
        self.assertEqual(len(a) + len(b), 12)
        self.assertEqual(sorted(a + b), sorted(list(BalancedTwoDomainDistributedSampler(
            dataset, num_replicas=1, rank=0, seed=3
        ))))

    def test_mixed_cache_routes_per_sample_domain(self):
        cache = MixedTeacherVelocityCache(
            {
                "legacy": _FakeCache(10.0),
                "ultravideo": _FakeCache(100.0),
            }
        )
        batch = {
            "teacher_cache_domain": ["legacy", "ultravideo"],
            "frame_indices": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int64),
            "record_uid": ["a:2", "b:3"],
            "sample_id": ["a2", "b3"],
            "variant": ["a", "b"],
            "crop_top": torch.tensor([0, 0]),
            "crop_left": torch.tensor([0, 0]),
            "horizontal_flip": torch.tensor([False, False]),
            "vertical_flip": torch.tensor([False, False]),
            "distillation_index": torch.tensor([2, 3]),
            "distillation_view_index": torch.tensor([0, 0]),
            "distillation_view_seed": torch.tensor([1, 1]),
        }
        values = cache.load_batch(batch, device="cpu", dtype=torch.float32)
        self.assertEqual(tuple(values.shape), (2, 1, 1, 1, 1))
        self.assertEqual(float(values[0].item()), 12.0)
        self.assertEqual(float(values[1].item()), 103.0)


if __name__ == "__main__":
    unittest.main()
