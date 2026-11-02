"""Immutable paired GT and M11B-S1-D1 latent reader for decoder-only post-training."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from swiftvr.data.task_pairs import build_legacy_pairs, build_ultra_pairs
from tools.m11b_s1_d1_cache import sha256


def verified_cache_metadata(root: Path, *, model_root: Path, reae_root: Path,
                            require_formal: bool = True) -> dict:
    root = Path(root).resolve()
    path = root / "metadata.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("kind") != "M11B-S1-D1-restored-latent-v1" or m.get("phase") != "M11B-S1-D1":
        raise ValueError(f"Wrong latent-cache kind: {path}")
    if not m.get("complete") or int(m.get("sample_count", -1)) != len(m.get("samples", [])):
        raise ValueError(f"Incomplete cache: {path}")
    if (m.get("pilot") or m.get("partial_ultra") or
        int(m.get("sample_count", -1)) != int(m.get("dataset_count", -2)) or
        int(m.get("selected_count", -1)) != int(m.get("dataset_count", -2))):
        if require_formal:
            raise ValueError(f"Formal training refuses pilot or partial cache: {path}")
    if Path(m["source_checkpoint"]).resolve() != Path(model_root).resolve():
        raise ValueError(f"M11B checkpoint mismatch: {path}")
    if Path(m["base_checkpoint"]).resolve() != Path(reae_root).resolve():
        raise ValueError(f"ReAE checkpoint mismatch: {path}")
    expected = m.get("source_files", {})
    if not expected:
        raise ValueError(f"Missing source-hash provenance: {path}")
    for name, fingerprint in expected.items():
        source = Path(name)
        if not source.is_file() or sha256(source) != fingerprint:
            raise ValueError(f"Source file hash changed since D1: {source}")
    if not (root / "plan.json").is_file():
        raise FileNotFoundError(root / "plan.json")
    # Completed metadata must incorporate the same immutable plan as the cache build.
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("kind") != "M11B-S1-D1-restored-latent-plan-v1":
        raise ValueError(f"Invalid immutable D1 plan kind: {path}")
    for k, v in plan.items():
        # Cache finalization intentionally advances the plan kind to
        # 'M11B-S1-D1-restored-latent-v1' in completed metadata.
        if k == "kind":
            continue
        if m.get(k) != v:
            raise ValueError(f"Immutable D1 plan mismatch on {k}: {path}")
    return m


class M11BPairedLatentDataset(Dataset):
    """Load paired source frames + immutable z_sr, checking identity per sample."""

    def __init__(self, cache_root, *, model_root, reae_root, path_root=Path("."),
                 require_formal=True):
        self.root = Path(cache_root).expanduser().resolve()
        self.meta = verified_cache_metadata(
            self.root, model_root=model_root, reae_root=reae_root,
            require_formal=require_formal)
        self.rows = self.meta["samples"]
        data = self.meta["data"]
        if self.meta["domain"] == "legacy":
            dataset, _ = build_legacy_pairs(
                Path(data["legacy_view_cache"]), path_root=Path(path_root),
                split=self.meta["split"])
        elif self.meta["domain"] == "ultra":
            manifests = [Path(p) for p in data["ultra_pairs"]]
            dataset, _ = build_ultra_pairs(manifests, allow_partial=not require_formal)
        else:
            raise ValueError("Unknown D1 domain")
        if len(dataset) != len(self.rows):
            raise ValueError("D1 sample count does not match reconstructed paired view dataset")
        self.dataset = dataset
        self._check_index_manifest()

    def _check_index_manifest(self):
        seen = set()
        for i, row in enumerate(self.rows):
            if int(row["dataset_index"]) != i or int(row["cache_index"]) != i:
                raise ValueError(f"D1 index/order mismatch: {i}")
            pair = row["pair_id"]
            if pair in seen:
                raise ValueError(f"Duplicate D1 paired view identity: {pair}")
            seen.add(pair)
            file = (self.root / row["file"]).resolve()
            if self.root not in file.parents or not file.is_file():
                raise ValueError(f"Missing or escaped D1 latent file: {file}")
            if row["phase_formula"] != "(output_index+3)%4":
                raise ValueError(f"Unexpected phase convention: {row['phase_formula']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        i = int(index)
        source = self.dataset[i]
        meta = self.rows[i]
        if source["pair_id"] != meta["pair_id"] or source["domain"] not in (
            meta["domain"], "ultravideo" if meta["domain"] == "ultra" else meta["domain"]
        ):
            raise ValueError(f"LR/GT pair identity mismatch at index {i}")
        if list(source["lr"].shape) != meta["lr_shape"] or list(source["hr"].shape) != meta["gt_shape"]:
            raise ValueError(f"LR/GT geometry changed at index {i}")
        path = self.root / meta["file"]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"z_sr"}:
                raise ValueError(f"Unexpected D1 latent keys: {path}")
            latent = handle.get_tensor("z_sr")
        if latent.dtype != torch.float16 or list(latent.shape) != meta["z_sr_shape"]:
            raise ValueError(f"Latent shape/dtype changed: {path}")
        if not torch.isfinite(latent).all():
            raise ValueError(f"Non-finite cached latent: {path}")
        if (int(source["hr"].shape[0]) != 13 or
            tuple(latent.shape) != (4, 48, source["hr"].shape[-2]//16, source["hr"].shape[-1]//16)):
            raise ValueError(f"Latent-to-GT temporal/spatial mismatch at {i}")
        return {"z_sr": latent, "hr": source["hr"], "pair_id": meta["pair_id"],
                "domain": self.meta["domain"], "cache_index": i}
