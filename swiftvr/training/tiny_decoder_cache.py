"""Immutable Stage-B1 caches of long-run Stage-A SR latents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file

from .distillation import distillation_sample_identity
from .reference import sha256_file


TINY_DECODER_CACHE_FORMAT_VERSION = 1
TINY_DECODER_CACHE_METADATA_FILENAME = "metadata.json"


class TinyDecoderLatentCache:
    """Load deterministic ``z_SR`` targets produced by a frozen Stage-A model."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        metadata_path = self.root / TINY_DECODER_CACHE_METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Tiny-decoder cache metadata must be a JSON object")
        if int(metadata.get("format_version", -1)) != TINY_DECODER_CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported tiny-decoder cache format: {metadata.get('format_version')}"
            )
        if metadata.get("kind") != "swiftvr_stage_b1_sr_latent":
            raise ValueError(f"Unexpected cache kind: {metadata.get('kind')!r}")
        samples = metadata.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("Tiny-decoder cache must contain sample metadata")
        self.metadata = metadata
        self.samples_by_index: dict[int, Mapping[str, object]] = {}
        for item in samples:
            if not isinstance(item, Mapping):
                raise TypeError("Tiny-decoder cache sample entries must be mappings")
            index = int(item.get("distillation_index", -1))
            if index < 0 or index in self.samples_by_index:
                raise ValueError(f"Invalid/duplicate distillation index {index}")
            self.samples_by_index[index] = item
        if len(self.samples_by_index) != int(metadata.get("sample_count", -1)):
            raise ValueError("Tiny-decoder cache sample_count does not match metadata")

    def validate_dataset(
        self,
        *,
        manifests: Sequence[str | Path],
        split: str,
        clip_length: int,
        crop_size: int,
        scale: int,
        views_per_record: int,
        view_seed: int,
        horizontal_flip_probability: float,
        vertical_flip_probability: float,
        dataset_length: int,
    ) -> None:
        expected = {
            "split": str(split),
            "clip_length": int(clip_length),
            "crop_size": int(crop_size),
            "scale": int(scale),
            "views_per_record": int(views_per_record),
            "view_seed": int(view_seed),
            "horizontal_flip_probability": float(horizontal_flip_probability),
            "vertical_flip_probability": float(vertical_flip_probability),
            "full_dataset_length": int(dataset_length),
        }
        differences = [
            f"{key}: cache={self.metadata.get(key)!r}, current={value!r}"
            for key, value in expected.items()
            if self.metadata.get(key) != value
        ]
        hashes = self.metadata.get("manifest_sha256")
        if not isinstance(hashes, Mapping):
            differences.append("cache is missing manifest_sha256")
        else:
            for manifest in manifests:
                path = Path(manifest).expanduser().resolve()
                actual = sha256_file(path)
                if hashes.get(str(path)) != actual:
                    differences.append(f"manifest hash mismatch: {path}")
        if differences:
            raise ValueError(
                "Tiny-decoder latent cache configuration differs:\n  "
                + "\n  ".join(differences)
            )

    def selected_indices(self) -> tuple[int, ...]:
        raw = self.metadata.get("selected_indices")
        if not isinstance(raw, list) or not raw:
            raise ValueError("Tiny-decoder cache is missing selected_indices")
        indices = tuple(int(value) for value in raw)
        if len(indices) != int(self.metadata.get("sample_count", -1)):
            raise ValueError("selected_indices length does not match sample_count")
        if len(set(indices)) != len(indices):
            raise ValueError("selected_indices contains duplicates")
        return indices

    def load(
        self,
        identity: Mapping[str, object],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        index = int(identity["distillation_index"])
        item = self.samples_by_index.get(index)
        if item is None:
            raise KeyError(f"Tiny-decoder cache has no sample index {index}")
        for field in (
            "key",
            "record_uid",
            "frame_indices",
            "crop_top",
            "crop_left",
            "horizontal_flip",
            "vertical_flip",
            "view_index",
            "view_seed",
        ):
            if item.get(field) != identity.get(field):
                raise ValueError(
                    f"Tiny-decoder cache identity mismatch index={index}, field={field}: "
                    f"cache={item.get(field)!r}, current={identity.get(field)!r}"
                )
        filename = item.get("file")
        if not isinstance(filename, str):
            raise TypeError(f"Tiny-decoder cache file missing for index {index}")
        payload = load_file(str(self.root / filename), device="cpu")
        if "z_sr" not in payload:
            raise KeyError(f"Cache sample {filename} lacks 'z_sr'")
        tensor = payload["z_sr"].to(device=device)
        return tensor if dtype is None else tensor.to(dtype=dtype)

    def load_batch(
        self,
        batch: Mapping[str, object],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        frame_indices = batch.get("frame_indices")
        if not isinstance(frame_indices, torch.Tensor) or frame_indices.ndim != 2:
            raise TypeError("Expected collated frame_indices tensor [B,T]")
        values = [
            self.load(
                distillation_sample_identity(batch, index),
                device=device,
                dtype=dtype,
            )
            for index in range(int(frame_indices.shape[0]))
        ]
        return torch.stack(values, dim=0)
