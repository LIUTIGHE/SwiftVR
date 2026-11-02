"""Paired HR adapters for task post-training; legacy/cache identities stay intact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import Dataset, ConcatDataset

HR_FIELDS = {"hr_bundle_path", "hr_bundle_key", "lr_row_sha256"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected nonempty JSONL object rows: {path}")
    return rows


def lr_row_hash(row):
    value = {key: val for key, val in row.items() if key not in HR_FIELDS}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def clean_hr_crop(raw, row):
    """Reuse the canonical LR materializer's crop at scale-one output geometry.

    Scale every HQ crop coordinate by the original scale, then request scale=1.
    The final BOX resize in the canonical helper is now identity, exposing its
    pre-downsample HR crop (including the exact native-8K to canonical-4K rule).
    No independent crop implementation or altered LR pixels are introduced.
    """
    from tools.materialize_ultravideo_lr_bundles import _clean_hq_crop
    scale = int(row["scale"])
    hr_row = {**row, "scale": 1,
              "crop_box_hq": [int(value) * scale for value in row["crop_box_hq"]]}
    return _clean_hq_crop(raw, hr_row)


class TaskViewDataset(Dataset):
    """A common LR/HR schema; no augmentation after the canonical fixed view."""
    def __init__(self, dataset, domain, *, hr_rows=None):
        self.dataset, self.domain, self.hr_rows = dataset, str(domain), hr_rows
        if hr_rows is not None and len(hr_rows) != len(dataset):
            raise ValueError("HR manifest length differs from canonical LR dataset")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        lr = sample["lr"]
        if self.hr_rows is None:
            hr = sample["hr"]
            pair_id = f"{self.domain}:{sample['record_uid']}:{sample['distillation_index']}"
        else:
            row = self.hr_rows[index]
            with safe_open(row["hr_bundle_path"], framework="pt", device="cpu") as handle:
                hr = handle.get_tensor(row["hr_bundle_key"])
            if hr.dtype != torch.uint8:
                raise ValueError("HR targets must be lossless uint8 RGB")
            hr = hr.float().div_(255)
            pair_id = row["lr_row_sha256"]
        scale = int(sample["scale"])
        if hr.ndim != 4 or hr.shape[:2] != lr.shape[:2] or hr.shape[1] != 3:
            raise ValueError("HR/LR temporal or channel mismatch")
        if tuple(hr.shape[-2:]) != tuple(n * scale for n in lr.shape[-2:]):
            raise ValueError("HR must be the real 3x target, not HQ upsampling")
        return {"lr": lr, "hr": hr, "pair_id": pair_id, "domain": self.domain}


def build_ultra_pairs(manifests, *, allow_partial=False):
    from swiftvr.data.materialized_lr_dataset import UltraVideoMaterializedLRDataset
    datasets, indices, source_signatures, shard_indices, metas = [], [], set(), [], []
    for path in manifests:
        path = Path(path).resolve()
        rows = read_jsonl(path)
        meta = json.loads((path.parent / "hr_metadata.json").read_text())
        if not allow_partial and not meta.get("source_complete", False):
            raise ValueError("Formal training requires the full merged LR source manifest")
        if meta.get("kind") != "ultravideo_hr_targets_v1":
            raise ValueError("Not an HR target materialization")
        if sha256_file(path) != meta["paired_manifest_sha256"]:
            raise ValueError("Paired manifest hash changed")
        source_signatures.add((meta["source_manifest_sha256"], meta["source_view_count"], meta["shard_count"]))
        shard_indices.append(meta["shard_index"])
        metas.append(meta)
        for row in rows:
            if not HR_FIELDS.issubset(row) or row["lr_row_sha256"] != lr_row_hash(row):
                raise ValueError("HR target is not bound to this exact LR view")
            if not Path(row["hr_bundle_path"]).is_file():
                raise FileNotFoundError(row["hr_bundle_path"])
            indices.append(int(row["materialized_index"]))
        canonical = UltraVideoMaterializedLRDataset(path, verify_paths=True)
        datasets.append(TaskViewDataset(canonical, "ultravideo", hr_rows=rows))
    if len(source_signatures) != 1 or len(indices) != len(set(indices)):
        raise ValueError("Mixed source manifests or duplicate materialized indices")
    _, expected, shards = next(iter(source_signatures))
    if not allow_partial and (set(indices) != set(range(expected)) or set(shard_indices) != set(range(shards))):
        raise ValueError("Formal training requires all HR shards/views; partial caches are smoke-only")
    return ConcatDataset(datasets), metas


def build_legacy_pairs(cache_root, *, path_root, split):
    # Teacher cache metadata determines historical views; NO velocity targets are loaded.
    from swiftvr.training.distillation import TeacherVelocityCache
    from tools.train_teacher_distillation_ddp import build_cached_dataset
    cache = TeacherVelocityCache(cache_root)
    m = cache.metadata
    if m["split"] != split:
        raise ValueError(f"Expected {split} view metadata")
    if int(m["sample_count"]) != int(m["full_dataset_length"]):
        raise ValueError("Use the complete historical view metadata, not train_short")
    views = build_cached_dataset(
        [Path(v) for v in m["manifests"]], cache, split=split, path_root=Path(path_root),
        clip_length=int(m["clip_length"]), crop_size=int(m["crop_size"]), scale=int(m["scale"]),
        views_per_record=int(m["views_per_record"]), view_seed=int(m["view_seed"]),
        hflip=float(m["horizontal_flip_probability"]), vflip=float(m["vertical_flip_probability"]),
        verify_paths=False, load_hq=False,
    )
    return TaskViewDataset(views, "legacy"), m
