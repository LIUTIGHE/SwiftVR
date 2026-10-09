"""Dataset for pre-materialized UltraVideo LR distillation views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from torch.utils.data import Dataset


class UltraVideoMaterializedLRDataset(Dataset):
    """Read fixed LR-only UltraVideo views from compact per-clip safetensors bundles.

    Each manifest row is already one complete deterministic distillation sample.
    No temporal/spatial/flip augmentation is applied here.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        verify_paths: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)

        rows: list[dict[str, object]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(
                        f"{self.manifest_path}:{line_number}: expected JSON object"
                    )
                required = {
                    "bundle_path",
                    "bundle_key",
                    "sample_id",
                    "record_uid",
                    "variant",
                    "frame_indices",
                    "crop_top",
                    "crop_left",
                    "scale",
                    "target_height",
                    "target_width",
                    "horizontal_flip",
                    "vertical_flip",
                    "view_index",
                    "view_seed",
                }
                missing = sorted(required - set(value))
                if missing:
                    raise ValueError(
                        f"{self.manifest_path}:{line_number}: missing {missing}"
                    )
                frame_indices = value["frame_indices"]
                if not isinstance(frame_indices, list) or not frame_indices:
                    raise ValueError(
                        f"{self.manifest_path}:{line_number}: frame_indices must be non-empty list"
                    )
                path = Path(str(value["bundle_path"])).expanduser().resolve()
                value["bundle_path"] = str(path)
                if verify_paths and not path.is_file():
                    raise FileNotFoundError(path)
                rows.append(value)

        if not rows:
            raise RuntimeError(f"No materialized views in {self.manifest_path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        normalized = int(index)
        if normalized < 0:
            normalized += len(self.rows)
        if not 0 <= normalized < len(self.rows):
            raise IndexError(index)
        row = self.rows[normalized]

        bundle_path = str(row["bundle_path"])
        key = str(row["bundle_key"])
        with safe_open(bundle_path, framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                raise KeyError(f"{bundle_path}: missing tensor {key!r}")
            lr_uint8 = handle.get_tensor(key)

        if lr_uint8.dtype != torch.uint8 or lr_uint8.ndim != 4:
            raise ValueError(
                f"{bundle_path}:{key}: expected uint8 [T,C,H,W], "
                f"got dtype={lr_uint8.dtype} shape={tuple(lr_uint8.shape)}"
            )
        if int(lr_uint8.shape[1]) != 3:
            raise ValueError(f"{bundle_path}:{key}: expected RGB channels")

        target_height = int(row["target_height"])
        target_width = int(row["target_width"])
        scale = int(row["scale"])
        if target_height <= 0 or target_width <= 0 or scale <= 0:
            raise ValueError(f"{bundle_path}:{key}: invalid target geometry")
        if target_height != int(lr_uint8.shape[-2]) * scale:
            raise ValueError(f"{bundle_path}:{key}: target_height/scale mismatch")
        if target_width != int(lr_uint8.shape[-1]) * scale:
            raise ValueError(f"{bundle_path}:{key}: target_width/scale mismatch")

        frame_indices = [int(value) for value in row["frame_indices"]]
        if len(frame_indices) != int(lr_uint8.shape[0]):
            raise ValueError(f"{bundle_path}:{key}: temporal length mismatch")

        return {
            "lr": lr_uint8.float().div_(255.0),
            "sample_id": str(row["sample_id"]),
            "record_uid": str(row["record_uid"]),
            "variant": str(row["variant"]),
            "split": str(row.get("split", "train")),
            "source_manifest": str(self.manifest_path),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
            "temporal_start": int(frame_indices[0]),
            "crop_top": int(row["crop_top"]),
            "crop_left": int(row["crop_left"]),
            "scale": scale,
            "target_height": target_height,
            "target_width": target_width,
            "horizontal_flip": bool(row["horizontal_flip"]),
            "vertical_flip": bool(row["vertical_flip"]),
            "distillation_index": normalized,
            "distillation_record_index": normalized,
            "distillation_view_index": int(row["view_index"]),
            "distillation_view_seed": int(row["view_seed"]),
            "materialized_index": int(row.get("materialized_index", normalized)),
            "selection_category": str(row.get("selection_category", "")),
        }


__all__ = ["UltraVideoMaterializedLRDataset"]
