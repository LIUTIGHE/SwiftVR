"""Regression tests for fast mid-epoch DataLoader resume."""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from swiftvr.training import skip_batches


class _CountingDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = int(length)
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        value = int(index)
        self.accessed.append(value)
        return value


def _loader(dataset: Dataset, *, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )


class FastResumeSkipTest(unittest.TestCase):
    def test_dataloader_skip_preserves_order_without_fetching_skipped_items(self):
        full_dataset = _CountingDataset(12)
        full_batches = [batch.tolist() for batch in _loader(full_dataset, seed=17)]

        resumed_dataset = _CountingDataset(12)
        resumed_iterator = iter(_loader(resumed_dataset, seed=17))
        skip_batches(resumed_iterator, 3)

        self.assertEqual(resumed_dataset.accessed, [])
        resumed_batches = [batch.tolist() for batch in resumed_iterator]
        self.assertEqual(resumed_batches, full_batches[3:])
        self.assertEqual(len(resumed_dataset.accessed), 6)

    def test_generic_iterator_fallback_still_consumes_items(self):
        iterator = iter([0, 1, 2, 3])
        skip_batches(iterator, 2)
        self.assertEqual(list(iterator), [2, 3])

    def test_skip_fails_when_batch_cursor_exceeds_epoch(self):
        iterator = iter(_loader(_CountingDataset(4), seed=3))
        with self.assertRaisesRegex(RuntimeError, "ended after"):
            skip_batches(iterator, 3)


if __name__ == "__main__":
    unittest.main()
