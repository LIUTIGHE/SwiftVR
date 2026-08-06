"""Input-pipeline helpers shared by SwiftVR training entry points.

The helpers in this module are deliberately CPU-only and side-effect free so
DataLoader worker settings can be validated without importing model code.
"""

from __future__ import annotations


def dataloader_worker_kwargs(
    *,
    num_workers: int,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> dict[str, object]:
    """Return valid PyTorch DataLoader worker keyword arguments.

    ``prefetch_factor`` and ``persistent_workers`` are only meaningful when
    workers are enabled.  Omitting them for ``num_workers=0`` avoids PyTorch's
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
