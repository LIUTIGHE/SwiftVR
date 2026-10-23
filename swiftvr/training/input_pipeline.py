"""Input-pipeline helpers shared by SwiftVR training entry points.

The helpers in this module are deliberately CPU-only and side-effect free so
DataLoader worker settings can be validated without importing model code.
"""

from __future__ import annotations

from collections.abc import Iterator


def dataloader_worker_kwargs(
    *,
    num_workers: int,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> dict[str, object]:
    """Return valid PyTorch DataLoader worker keyword arguments.

    ``prefetch_factor`` and ``persistent_workers`` are only meaningful when
    workers are enabled. Omitting them for ``num_workers=0`` avoids PyTorch's
    invalid-configuration errors and preserves the historical synchronous path.
    """

    workers = int(num_workers)
    prefetch = int(prefetch_factor)
    persistent = bool(persistent_workers)
    if workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {workers}")
    if prefetch <= 0:
        raise ValueError(f"prefetch_factor must be positive, got {prefetch}")
    if persistent and workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")

    result: dict[str, object] = {"num_workers": workers}
    if workers > 0:
        result["prefetch_factor"] = prefetch
        result["persistent_workers"] = persistent
    return result


def skip_prefetched_batches(iterator: Iterator[object], count: int) -> None:
    """Advance a DataLoader iterator safely during exact resume.

    The historical fast-resume path advances a DataLoader iterator's private
    ``_sampler_iter`` directly. That is safe for ``num_workers=0`` because the
    sampler has not advanced ahead of the consumer. With multiprocessing,
    however, creating the iterator immediately consumes roughly
    ``num_workers * prefetch_factor`` batch indices to fill worker queues. Directly
    advancing ``_sampler_iter`` after that point therefore skips *after* the
    prefetched batches and corrupts the resume position.

    For a multiprocessing DataLoader we instead consume the public iterator so
    queued batches are drained in the exact original order. This re-decodes the
    skipped samples once at resume startup, but normal training throughput is
    unchanged. Single-process DataLoaders retain the zero-decode fast path.
    """

    skip = int(count)
    if skip < 0:
        raise ValueError(f"count must be non-negative, got {skip}")

    workers = int(getattr(iterator, "_num_workers", 0) or 0)
    sampler_iterator = getattr(iterator, "_sampler_iter", None)
    target = sampler_iterator if workers == 0 and sampler_iterator is not None else iterator

    for index in range(skip):
        try:
            next(target)
        except StopIteration as exc:
            raise RuntimeError(
                f"Cannot skip {skip} batches; iterator ended after {index}"
            ) from exc
