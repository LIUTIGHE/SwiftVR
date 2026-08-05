"""Generalization-stage helpers for cached SwiftVR teacher distillation.

This module is intentionally independent from the training CLI so its selection,
cache-coverage, overlap, and resume-fingerprint rules can be unit tested on CPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset, Subset

SELECTION_MODES = frozenset({"all", "prefix", "random"})


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_distillation_indices(
    dataset_length: int,
    *,
    max_samples: int | None,
    mode: str = "all",
    seed: int = 0,
) -> tuple[int, ...]:
    """Select stable full-dataset indices for an offline teacher cache."""

    length = int(dataset_length)
    if length <= 0:
        raise ValueError("dataset_length must be positive")
    normalized_mode = str(mode).lower()
    if normalized_mode not in SELECTION_MODES:
        raise ValueError(
            f"Unsupported selection mode {mode!r}; expected {sorted(SELECTION_MODES)}"
        )

    if max_samples is None:
        count = length
    else:
        count = int(max_samples)
        if count <= 0:
            raise ValueError("max_samples must be positive when provided")
        count = min(count, length)

    if normalized_mode == "all":
        if count != length:
            raise ValueError("selection mode 'all' cannot be combined with a subset")
        return tuple(range(length))
    if normalized_mode == "prefix":
        return tuple(range(count))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected = torch.randperm(length, generator=generator)[:count].tolist()
    return tuple(int(index) for index in selected)


def selected_indices_sha256(indices: Sequence[int]) -> str:
    return _sha256_json([int(index) for index in indices])


def cache_selected_indices(metadata: Mapping[str, object]) -> tuple[int, ...]:
    """Read selected full-dataset indices from cache metadata.

    Legacy prefix caches do not store ``selected_indices``. They remain supported
    by reconstructing ``range(sample_count)``.
    """

    raw = metadata.get("selected_indices")
    if raw is None:
        count = int(metadata.get("sample_count", -1))
        if count <= 0:
            raise ValueError("Cache metadata has no valid sample_count")
        return tuple(range(count))
    if not isinstance(raw, list):
        raise TypeError("selected_indices must be a JSON list")
    indices = tuple(int(index) for index in raw)
    if len(indices) != int(metadata.get("sample_count", -1)):
        raise ValueError("selected_indices length does not match sample_count")
    if len(set(indices)) != len(indices):
        raise ValueError("selected_indices contains duplicates")
    expected_hash = metadata.get("selected_indices_sha256")
    if expected_hash is not None and expected_hash != selected_indices_sha256(indices):
        raise ValueError("selected_indices_sha256 does not match selected_indices")
    return indices


def build_cache_backed_subset(dataset: Dataset, cache) -> Subset:
    """Return dataset views in exactly the order represented by a teacher cache."""

    indices = cache_selected_indices(cache.metadata)
    if not indices:
        raise ValueError("Teacher cache selected no samples")
    length = len(dataset)
    out_of_range = [index for index in indices if index < 0 or index >= length]
    if out_of_range:
        raise ValueError(
            f"Teacher cache indices exceed dataset length {length}: {out_of_range[:8]}"
        )
    cached = set(int(index) for index in cache.samples_by_index)
    if cached != set(indices):
        missing = sorted(set(indices) - cached)
        extra = sorted(cached - set(indices))
        raise ValueError(
            "Teacher cache files do not match selected_indices: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    return Subset(dataset, list(indices))


def cache_record_uids(metadata: Mapping[str, object]) -> set[str]:
    samples = metadata.get("samples")
    if not isinstance(samples, list):
        raise TypeError("Cache metadata samples must be a list")
    result: set[str] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise TypeError("Cache sample entries must be mappings")
        uid = sample.get("record_uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError("Cache sample is missing record_uid")
        result.add(uid)
    return result


def cache_overlap_report(
    train_metadata: Mapping[str, object],
    val_metadata: Mapping[str, object],
) -> dict[str, object]:
    train_uids = cache_record_uids(train_metadata)
    val_uids = cache_record_uids(val_metadata)
    overlap = sorted(train_uids & val_uids)
    return {
        "train_records": len(train_uids),
        "val_records": len(val_uids),
        "overlap_records": len(overlap),
        "overlap_record_uids": overlap,
        "reference_checkpoint_match": (
            train_metadata.get("reference_checkpoint")
            == val_metadata.get("reference_checkpoint")
        ),
        "prompt_embedding_sha256_match": (
            train_metadata.get("prompt_embedding_sha256")
            == val_metadata.get("prompt_embedding_sha256")
        ),
        "reae_sha256_match": (
            train_metadata.get("reae_sha256") == val_metadata.get("reae_sha256")
        ),
        "timestep_match": train_metadata.get("timestep") == val_metadata.get("timestep"),
    }


def validate_resume_fingerprint(
    saved: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    if dict(saved) == dict(current):
        return
    differences = [
        f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]
    raise ValueError(
        "Teacher-distillation resume configuration differs:\n  "
        + "\n  ".join(differences[:64])
    )


def resolve_stored_checkpoint(run_dir: Path, value: str) -> Path:
    candidate = Path(value)
    resolved = candidate if candidate.is_absolute() else run_dir / candidate
    return resolved.expanduser().resolve()
