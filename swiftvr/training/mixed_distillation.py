"""Helpers for mixed legacy + UltraVideo teacher-velocity distillation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from .distillation import TeacherVelocityCache, distillation_sample_identity
from .reference import sha256_file


class TaggedDataset(Dataset):
    """Attach a teacher-cache domain tag without changing the wrapped sample."""

    def __init__(self, dataset: Dataset, tag: str) -> None:
        self.dataset = dataset
        self.tag = str(tag)
        if not self.tag:
            raise ValueError("tag must be non-empty")
        if len(dataset) <= 0:
            raise ValueError("dataset must be non-empty")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[int(index)]
        if not isinstance(sample, Mapping):
            raise TypeError("TaggedDataset requires mapping samples")
        result = dict(sample)
        result["teacher_cache_domain"] = self.tag
        return result


class MixedTeacherVelocityCache:
    """Route each collated sample to its domain-specific immutable teacher cache."""

    def __init__(self, caches: Mapping[str, TeacherVelocityCache]) -> None:
        normalized = {str(key): value for key, value in caches.items()}
        if not normalized:
            raise ValueError("At least one teacher cache is required")
        if any(not key for key in normalized):
            raise ValueError("Teacher cache tags must be non-empty")
        self.caches = normalized

    def load_batch(
        self,
        batch: Mapping[str, object],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        domains = batch.get("teacher_cache_domain")
        if not isinstance(domains, Sequence) or isinstance(domains, (str, bytes)):
            raise TypeError(
                "Mixed teacher batch requires collated teacher_cache_domain sequence"
            )
        frame_indices = batch.get("frame_indices")
        if not isinstance(frame_indices, torch.Tensor) or frame_indices.ndim != 2:
            raise TypeError("Expected collated frame_indices tensor [B,T]")
        if len(domains) != int(frame_indices.shape[0]):
            raise ValueError("teacher_cache_domain batch length mismatch")

        targets: list[torch.Tensor] = []
        for index, domain in enumerate(domains):
            tag = str(domain)
            cache = self.caches.get(tag)
            if cache is None:
                raise KeyError(f"Unknown teacher cache domain {tag!r}")
            identity = distillation_sample_identity(batch, index)
            targets.append(
                cache.load(
                    identity,
                    device=device,
                    dtype=dtype,
                )
            )
        return torch.stack(targets, dim=0)


class BalancedTwoDomainDistributedSampler(Sampler[int]):
    """Exact 50:50 two-domain sampling with deterministic DDP sharding.

    The wrapped dataset must be ConcatDataset([domain_a, domain_b]). Each global
    epoch draws max(len(a), len(b)) samples from each domain. The smaller domain
    is reshuffled and cycled only as much as necessary. The combined list is
    shuffled once more and then sharded by rank with no padding.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        if len(dataset.datasets) != 2:
            raise ValueError("BalancedTwoDomainDistributedSampler requires two datasets")
        self.length_a = len(dataset.datasets[0])
        self.length_b = len(dataset.datasets[1])
        if self.length_a <= 0 or self.length_b <= 0:
            raise ValueError("Both domains must be non-empty")
        self.offset_b = self.length_a
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        requested_per_domain = max(self.length_a, self.length_b)
        if self.drop_last:
            self.per_domain_global = (
                requested_per_domain // self.num_replicas
            ) * self.num_replicas
            if self.per_domain_global <= 0:
                raise ValueError(
                    "Largest domain is smaller than num_replicas with drop_last=True"
                )
        else:
            self.per_domain_global = (
                (requested_per_domain + self.num_replicas - 1)
                // self.num_replicas
            ) * self.num_replicas
        self.total_size = 2 * self.per_domain_global
        self.num_samples = self.total_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _domain_indices(
        self,
        length: int,
        count: int,
        *,
        seed_offset: int,
        add_offset: int,
    ) -> list[int]:
        result: list[int] = []
        round_index = 0
        while len(result) < count:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                self.seed
                + self.epoch * 1_000_003
                + int(seed_offset)
                + round_index * 97_409
            )
            permutation = torch.randperm(length, generator=generator).tolist()
            needed = count - len(result)
            result.extend(
                add_offset + int(index)
                for index in permutation[:needed]
            )
            round_index += 1
        return result

    def __iter__(self) -> Iterator[int]:
        a = self._domain_indices(
            self.length_a,
            self.per_domain_global,
            seed_offset=11,
            add_offset=0,
        )
        b = self._domain_indices(
            self.length_b,
            self.per_domain_global,
            seed_offset=29,
            add_offset=self.offset_b,
        )
        combined = [value for pair in zip(a, b) for value in pair]

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch * 1_000_003 + 53)
        order = torch.randperm(len(combined), generator=generator).tolist()
        shuffled = [combined[index] for index in order]

        if len(shuffled) != self.total_size:
            raise RuntimeError(
                f"Balanced sampler built {len(shuffled)} global indices, "
                f"expected {self.total_size}"
            )

        rank_indices = shuffled[self.rank : self.total_size : self.num_replicas]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError(
                f"Sampler rank {self.rank} produced {len(rank_indices)} indices, "
                f"expected {self.num_samples}"
            )
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples


def validate_ultravideo_teacher_cache(
    cache: TeacherVelocityCache,
    *,
    materialized_manifest: str | Path,
    dataset_length: int,
) -> None:
    manifest = Path(materialized_manifest).expanduser().resolve()
    differences: list[str] = []
    if cache.metadata.get("dataset_kind") != "ultravideo_materialized_lr_v1":
        differences.append(
            "dataset_kind: "
            f"cache={cache.metadata.get('dataset_kind')!r}, "
            "expected='ultravideo_materialized_lr_v1'"
        )
    if cache.metadata.get("materialized_manifest") != str(manifest):
        differences.append(
            "materialized_manifest: "
            f"cache={cache.metadata.get('materialized_manifest')!r}, "
            f"current={str(manifest)!r}"
        )
    expected_hash = sha256_file(manifest)
    if cache.metadata.get("materialized_manifest_sha256") != expected_hash:
        differences.append("materialized_manifest_sha256 mismatch")
    if int(cache.metadata.get("full_dataset_length", -1)) != int(dataset_length):
        differences.append(
            "full_dataset_length: "
            f"cache={cache.metadata.get('full_dataset_length')!r}, "
            f"current={int(dataset_length)!r}"
        )
    sample_count = int(cache.metadata.get("sample_count", -1))
    if sample_count != int(dataset_length):
        differences.append(
            f"sample_count: cache={sample_count}, current={int(dataset_length)}"
        )
    expected_indices = set(range(int(dataset_length)))
    if set(cache.samples_by_index) != expected_indices:
        differences.append("teacher cache does not cover every UltraVideo view")
    if differences:
        raise ValueError(
            "UltraVideo teacher cache configuration differs:\n  "
            + "\n  ".join(differences)
        )


__all__ = [
    "BalancedTwoDomainDistributedSampler",
    "MixedTeacherVelocityCache",
    "TaggedDataset",
    "validate_ultravideo_teacher_cache",
]
