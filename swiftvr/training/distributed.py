"""Small, dependency-free helpers for single-node SwiftVR DDP training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sized

from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def global_effective_batch_size(
    *,
    world_size: int,
    local_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return samples contributing to one synchronized optimizer step."""

    values = {
        "world_size": world_size,
        "local_batch_size": local_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for name, value in values.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    return int(world_size) * int(local_batch_size) * int(gradient_accumulation_steps)


def accumulation_for_global_batch(
    *,
    target_global_batch: int,
    world_size: int,
    local_batch_size: int = 1,
) -> int:
    """Compute an integer local accumulation count preserving a global batch."""

    if target_global_batch <= 0:
        raise ValueError("target_global_batch must be positive")
    denominator = int(world_size) * int(local_batch_size)
    if denominator <= 0:
        raise ValueError("world_size and local_batch_size must be positive")
    if target_global_batch % denominator:
        raise ValueError(
            f"target_global_batch={target_global_batch} is not divisible by "
            f"world_size*local_batch_size={denominator}"
        )
    return target_global_batch // denominator


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices without padding or duplicating samples."""

    def __init__(self, dataset: Sized, *, rank: int, world_size: int) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else math.ceil(remaining / self.world_size)
