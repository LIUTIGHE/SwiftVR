"""Regression tests for multi-worker exact-resume batch skipping."""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from swiftvr.training.input_pipeline import skip_prefetched_batches


class _RangeDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return int(index)


def _loader(*, workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(17)
    kwargs: dict[str, object] = {}
    if workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    return DataLoader(
        _RangeDataset(64),
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        generator=generator,
        **kwargs,
    )


class PrefetchedResumeSkipTest(unittest.TestCase):
    def test_multiworker_skip_preserves_batch_order(self):
        full = [batch.tolist() for batch in _loader(workers=2)]

        resumed_loader = _loader(workers=2)
        resumed_iterator = iter(resumed_loader)
        skip_prefetched_batches(resumed_iterator, 9)
        resumed = [batch.tolist() for batch in resumed_iterator]

        self.assertEqual(resumed, full[9:])

    def test_single_process_skip_preserves_batch_order(self):
        full = [batch.tolist() for batch in _loader(workers=0)]

        resumed_iterator = iter(_loader(workers=0))
        skip_prefetched_batches(resumed_iterator, 9)
        resumed = [batch.tolist() for batch in resumed_iterator]

        self.assertEqual(resumed, full[9:])

    def test_skip_beyond_epoch_raises(self):
        iterator = iter(_loader(workers=2))
        with self.assertRaisesRegex(RuntimeError, "iterator ended after"):
            skip_prefetched_batches(iterator, 100)


if __name__ == "__main__":
    unittest.main()
