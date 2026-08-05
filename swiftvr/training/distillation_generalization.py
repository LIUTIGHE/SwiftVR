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

from swiftvr.data import TripletSequenceRecord, read_triplet_manifests

SELECTION_MODES = frozenset({"all", "prefix", "random"})
SOURCE_IDENTITY_METHOD = "resolved_hr_frame_paths_sha256_v1"


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


def record_source_uid(record: TripletSequenceRecord) -> str:
    """Hash the complete resolved HR sequence path list for leakage checks.

    HR paths identify the underlying clean source sequence. HQ/LR paths, crop,
    temporal view, degradation variant, and augmentation are intentionally omitted:
    distinct degradations or views of the same HR sequence must still count as the
    same source. Paths are already absolute and normalized by the manifest reader.
    """

    if not record.hr_paths:
        raise ValueError("Triplet record has no HR paths")
    return _sha256_json(
        {
            "method": SOURCE_IDENTITY_METHOD,
            "hr_paths": [
                str(Path(path).expanduser().resolve()) for path in record.hr_paths
            ],
        }
    )


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


def _resolved_records_for_cache(
    metadata: Mapping[str, object],
    *,
    path_root: str | Path | None,
) -> list[TripletSequenceRecord]:
    manifests = metadata.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError(
            "Legacy cache has no embedded source_uid and metadata lacks manifests"
        )
    root_value: str | Path
    if path_root is not None:
        root_value = path_root
    elif metadata.get("path_root") not in (None, ""):
        root_value = str(metadata["path_root"])
    else:
        root_value = Path.cwd()
    split_value = metadata.get("split")
    split = None if split_value in (None, "") else str(split_value)
    records = read_triplet_manifests(
        [Path(str(value)) for value in manifests],
        split=split,
        path_root=root_value,
        verify_paths=False,
    )
    clip_length = int(metadata.get("clip_length", 0))
    if clip_length <= 0:
        raise ValueError("Cache metadata has no valid clip_length")
    filtered = [record for record in records if record.frame_count >= clip_length]
    expected = metadata.get("base_record_count")
    if expected is not None and len(filtered) != int(expected):
        raise ValueError(
            "Cache source reconstruction record count differs: "
            f"metadata={expected}, reconstructed={len(filtered)}"
        )
    if not filtered:
        raise ValueError("No records remain while reconstructing cache sources")
    return filtered


def cache_source_entries(
    metadata: Mapping[str, object],
    *,
    path_root: str | Path | None = None,
) -> dict[str, dict[str, object]]:
    """Return unique source identities represented by a cache.

    New caches embed ``source_uid`` per sample. Existing caches are supported by
    resolving their recorded manifests and mapping ``distillation_index`` back to
    the original record index. Thus old velocity tensors do not need rebuilding.
    """

    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples:
        raise TypeError("Cache metadata samples must be a non-empty list")
    embedded = all(
        isinstance(sample, Mapping)
        and isinstance(sample.get("source_uid"), str)
        and bool(sample.get("source_uid"))
        for sample in samples
    )
    records: list[TripletSequenceRecord] | None = None
    views_per_record = int(metadata.get("views_per_record", 0))
    if not embedded:
        if views_per_record <= 0:
            raise ValueError("Cache metadata has no valid views_per_record")
        records = _resolved_records_for_cache(metadata, path_root=path_root)

    entries: dict[str, dict[str, object]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise TypeError("Cache sample entries must be mappings")
        record_uid = sample.get("record_uid")
        if not isinstance(record_uid, str) or not record_uid:
            raise ValueError("Cache sample is missing record_uid")

        if embedded:
            source_uid = str(sample["source_uid"])
            first_path = sample.get("source_hr_first")
            last_path = sample.get("source_hr_last")
        else:
            assert records is not None
            index = int(sample.get("distillation_index", -1))
            if index < 0:
                raise ValueError("Legacy cache sample lacks distillation_index")
            record_index = index // views_per_record
            if record_index >= len(records):
                raise ValueError(
                    f"Cache distillation_index={index} maps beyond {len(records)} records"
                )
            record = records[record_index]
            if sample.get("sample_id") != record.sample_id:
                raise ValueError(
                    "Cache source reconstruction sample_id mismatch: "
                    f"cache={sample.get('sample_id')!r}, manifest={record.sample_id!r}"
                )
            if sample.get("variant") != record.variant:
                raise ValueError(
                    "Cache source reconstruction variant mismatch: "
                    f"cache={sample.get('variant')!r}, manifest={record.variant!r}"
                )
            source_uid = record_source_uid(record)
            first_path = record.hr_paths[0]
            last_path = record.hr_paths[-1]

        entry = entries.setdefault(
            source_uid,
            {
                "record_uids": set(),
                "source_hr_first": first_path,
                "source_hr_last": last_path,
            },
        )
        record_uids = entry["record_uids"]
        if not isinstance(record_uids, set):
            raise TypeError("Internal source entry record_uids must be a set")
        record_uids.add(record_uid)
    return entries


def cache_overlap_report(
    train_metadata: Mapping[str, object],
    val_metadata: Mapping[str, object],
    *,
    train_path_root: str | Path | None = None,
    val_path_root: str | Path | None = None,
) -> dict[str, object]:
    """Separate harmless human-readable name collisions from true source overlap."""

    train_record_uids = cache_record_uids(train_metadata)
    val_record_uids = cache_record_uids(val_metadata)
    record_uid_collisions = sorted(train_record_uids & val_record_uids)

    train_sources = cache_source_entries(train_metadata, path_root=train_path_root)
    val_sources = cache_source_entries(val_metadata, path_root=val_path_root)
    source_overlap_uids = sorted(set(train_sources) & set(val_sources))
    source_overlap_examples = []
    for uid in source_overlap_uids[:16]:
        train_entry = train_sources[uid]
        val_entry = val_sources[uid]
        source_overlap_examples.append(
            {
                "source_uid": uid,
                "train_record_uids": sorted(train_entry["record_uids"]),
                "val_record_uids": sorted(val_entry["record_uids"]),
                "train_hr_first": train_entry.get("source_hr_first"),
                "val_hr_first": val_entry.get("source_hr_first"),
            }
        )

    source_overlap_count = len(source_overlap_uids)
    return {
        "source_identity_method": SOURCE_IDENTITY_METHOD,
        "train_records": len(train_record_uids),
        "val_records": len(val_record_uids),
        "record_uid_collisions": len(record_uid_collisions),
        "record_uid_collision_values": record_uid_collisions,
        "train_sources": len(train_sources),
        "val_sources": len(val_sources),
        "source_overlap_records": source_overlap_count,
        "source_overlap_uids": source_overlap_uids,
        "source_overlap_examples": source_overlap_examples,
        # Compatibility for the existing inspector/trainer. These now mean true
        # source overlap, not record_uid string collision.
        "overlap_records": source_overlap_count,
        "overlap_record_uids": source_overlap_uids,
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
        "timestep_match": train_metadata.get("timestep")
        == val_metadata.get("timestep"),
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
