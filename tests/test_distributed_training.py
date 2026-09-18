"""CPU tests for SwiftVR distributed-training helper logic."""

from __future__ import annotations

import unittest

from swiftvr.training.distributed import (
    DistributedEvalSampler,
    accumulation_for_global_batch,
    global_effective_batch_size,
)


class _SizedDataset:
    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length


class DistributedTrainingHelperTest(unittest.TestCase):
    def test_global_effective_batch(self):
        self.assertEqual(
            global_effective_batch_size(
                world_size=4,
                local_batch_size=1,
                gradient_accumulation_steps=1,
            ),
            4,
        )
        self.assertEqual(
            global_effective_batch_size(
                world_size=2,
                local_batch_size=1,
                gradient_accumulation_steps=2,
            ),
            4,
        )

    def test_accumulation_preserves_target_global_batch(self):
        self.assertEqual(
            accumulation_for_global_batch(
                target_global_batch=4,
                world_size=4,
                local_batch_size=1,
            ),
            1,
        )
        self.assertEqual(
            accumulation_for_global_batch(
                target_global_batch=4,
                world_size=2,
                local_batch_size=1,
            ),
            2,
        )
        with self.assertRaisesRegex(ValueError, "not divisible"):
            accumulation_for_global_batch(
                target_global_batch=4,
                world_size=3,
                local_batch_size=1,
            )

    def test_eval_sampler_has_complete_nonduplicated_coverage(self):
        dataset = _SizedDataset(10)
        shards = [
            list(DistributedEvalSampler(dataset, rank=rank, world_size=3))
            for rank in range(3)
        ]
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(10)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(shards[0], [0, 3, 6, 9])
        self.assertEqual(shards[1], [1, 4, 7])
        self.assertEqual(shards[2], [2, 5, 8])

    def test_eval_sampler_handles_more_ranks_than_samples(self):
        dataset = _SizedDataset(2)
        self.assertEqual(
            list(DistributedEvalSampler(dataset, rank=0, world_size=4)), [0]
        )
        self.assertEqual(
            list(DistributedEvalSampler(dataset, rank=1, world_size=4)), [1]
        )
        self.assertEqual(
            list(DistributedEvalSampler(dataset, rank=2, world_size=4)), []
        )
        self.assertEqual(
            len(DistributedEvalSampler(dataset, rank=2, world_size=4)), 0
        )


if __name__ == "__main__":
    unittest.main()
