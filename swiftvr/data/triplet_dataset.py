"""Aligned HR/HQ/LR frame-sequence Dataset for SwiftVR training.

The dataset consumes JSONL manifests produced by
``tools/build_triplet_manifest.py``. Both compact frame-pattern records and
explicit ``*_frames`` records are supported. Image decoding is lazy: only the
sampled temporal clip and spatial crop are loaded for each item.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from swiftvr.training.input_pipeline import dataloader_worker_kwargs


MEDIA_FIELDS = ("hr", "hq", "lr")


@dataclass(frozen=True)
class TripletSequenceRecord:
    """Resolved manifest record containing aligned image paths."""

    sample_id: str
    split: str
    variant: str
    source_manifest: str
    frame_indices: tuple[int, ...]
    hr_paths: tuple[str, ...]
    hq_paths: tuple[str, ...]
    lr_paths: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)


def _as_path_list(
    manifests: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    values = (manifests,) if isinstance(manifests, (str, Path)) else tuple(manifests)
    if not values:
        raise ValueError("At least one manifest path is required")
    return tuple(Path(value).expanduser().resolve() for value in values)


def _coerce_int_sequence(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a JSON array")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain integers") from exc
    if not result:
        raise ValueError(f"{field} must not be empty")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"{field} must be strictly increasing")
    return result


def _record_indices(record: Mapping[str, object]) -> tuple[int, ...]:
    if "frame_indices" in record:
        indices = _coerce_int_sequence(record["frame_indices"], field="frame_indices")
    else:
        try:
            start = int(record["frame_start"])
            end = int(record["frame_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Frame record requires frame_indices or valid frame_start/frame_end"
            ) from exc
        if end < start:
            raise ValueError(f"frame_end={end} is smaller than frame_start={start}")
        indices = tuple(range(start, end + 1))

    frame_count = record.get("frame_count")
    if frame_count is not None and int(frame_count) != len(indices):
        raise ValueError(
            f"frame_count={frame_count} does not match {len(indices)} frame indices"
        )
    return indices


def _resolve_path(value: object, path_root: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = path_root / path
    return str(path.resolve())


def _coerce_path_sequence(
    value: object,
    *,
    field: str,
    path_root: Path,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a JSON array")
    result = tuple(_resolve_path(item, path_root) for item in value)
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _infer_variant(manifest: Path, record: Mapping[str, object]) -> str:
    if record.get("variant") not in (None, ""):
        return str(record["variant"])
    stem = manifest.stem.lower()
    if "text" in stem:
        return "text"
    if "plain" in stem:
        return "plain"
    return manifest.stem


def _resolve_manifest_record(
    record: Mapping[str, object],
    *,
    manifest: Path,
    path_root: Path,
    line_number: int,
) -> TripletSequenceRecord:
    sample_id = str(record.get("sample_id", "")).strip()
    if not sample_id:
        raise ValueError(f"{manifest}:{line_number}: missing sample_id")

    media_type = record.get("media_type")
    if media_type not in (None, "frames"):
        raise ValueError(
            f"{manifest}:{line_number}: training Dataset supports frame manifests, "
            f"got media_type={media_type!r}"
        )

    indices = _record_indices(record)
    explicit_fields = tuple(f"{name}_frames" for name in MEDIA_FIELDS)
    explicit_presence = [field in record for field in explicit_fields]
    if any(explicit_presence) and not all(explicit_presence):
        missing = [
            field
            for field, present in zip(explicit_fields, explicit_presence)
            if not present
        ]
        raise ValueError(
            f"{manifest}:{line_number}: explicit frame record missing {missing}"
        )

    resolved: dict[str, tuple[str, ...]] = {}
    if all(explicit_presence):
        for name in MEDIA_FIELDS:
            paths = _coerce_path_sequence(
                record[f"{name}_frames"],
                field=f"{name}_frames",
                path_root=path_root,
            )
            if len(paths) != len(indices):
                raise ValueError(
                    f"{manifest}:{line_number}: {name}_frames length {len(paths)} "
                    f"does not match {len(indices)} frame indices"
                )
            resolved[name] = paths
    else:
        missing = [name for name in MEDIA_FIELDS if name not in record]
        if missing:
            raise ValueError(
                f"{manifest}:{line_number}: pattern frame record missing {missing}"
            )
        for name in MEDIA_FIELDS:
            pattern = str(record[name])
            try:
                resolved[name] = tuple(
                    _resolve_path(pattern.format(frame=index), path_root)
                    for index in indices
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"{manifest}:{line_number}: invalid {name} frame pattern "
                    f"{pattern!r}"
                ) from exc

    return TripletSequenceRecord(
        sample_id=sample_id,
        split=str(record.get("split", "")),
        variant=_infer_variant(manifest, record),
        source_manifest=str(manifest),
        frame_indices=indices,
        hr_paths=resolved["hr"],
        hq_paths=resolved["hq"],
        lr_paths=resolved["lr"],
    )


def read_triplet_manifests(
    manifests: str | Path | Sequence[str | Path],
    *,
    split: str | None = None,
    path_root: str | Path | None = None,
    verify_paths: bool = False,
) -> list[TripletSequenceRecord]:
    """Read and resolve one or more frame-sequence manifests.

    Relative media paths are resolved against ``path_root``. When omitted,
    ``Path.cwd()`` is used, matching the repository-root execution convention of
    the manifest builder and audit tools.
    """

    manifest_paths = _as_path_list(manifests)
    root = (
        Path.cwd().resolve()
        if path_root is None
        else Path(path_root).expanduser().resolve()
    )
    records: list[TripletSequenceRecord] = []

    for manifest in manifest_paths:
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {manifest}")
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"{manifest}:{line_number}: manifest row is not a JSON object"
                    )
                if split is not None and payload.get("split") != split:
                    continue
                resolved = _resolve_manifest_record(
                    payload,
                    manifest=manifest,
                    path_root=root,
                    line_number=line_number,
                )
                if verify_paths:
                    for field_name, paths in (
                        ("hr_frames", resolved.hr_paths),
                        ("hq_frames", resolved.hq_paths),
                        ("lr_frames", resolved.lr_paths),
                    ):
                        missing = next(
                            (path for path in paths if not Path(path).is_file()),
                            None,
                        )
                        if missing is not None:
                            raise FileNotFoundError(
                                f"{manifest}:{line_number}: missing {field_name} "
                                f"file: {missing}"
                            )
                records.append(resolved)

    if not records:
        suffix = f" for split={split!r}" if split is not None else ""
        raise RuntimeError(f"No manifest records were loaded{suffix}")
    return records


def _normalize_crop_size(
    crop_size: int | Sequence[int] | None,
) -> tuple[int, int] | None:
    if crop_size is None:
        return None
    if isinstance(crop_size, int):
        height = width = int(crop_size)
    else:
        values = tuple(int(value) for value in crop_size)
        if len(values) != 2:
            raise ValueError("crop_size must be an int or a (height, width) pair")
        height, width = values
    if height <= 0 or width <= 0:
        raise ValueError(f"crop_size must be positive, got {(height, width)}")
    return height, width


def _image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def _load_rgb_crop(
    path: str,
    *,
    expected_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> torch.Tensor:
    top, left, height, width = crop_box
    with Image.open(path) as image:
        image = image.convert("RGB")
        actual_size = (image.height, image.width)
        if actual_size != expected_size:
            raise ValueError(
                f"Inconsistent frame size for {path}: expected {expected_size}, "
                f"got {actual_size}"
            )
        image = image.crop((left, top, left + width, top + height))
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_clip(
    paths: Sequence[str],
    *,
    positions: Sequence[int],
    expected_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> torch.Tensor:
    frames = [
        _load_rgb_crop(
            paths[position],
            expected_size=expected_size,
            crop_box=crop_box,
        )
        for position in positions
    ]
    return torch.stack(frames, dim=0).to(dtype=torch.float32).div_(255.0)


def _randint_inclusive(high: int) -> int:
    if high <= 0:
        return 0
    return int(torch.randint(0, high + 1, (1,)).item())


class TripletVideoDataset(Dataset):
    """Sample aligned SwiftVR clips from HR/HQ/LR frame manifests.

    Returned image tensors use ``[T, C, H, W]`` layout and float32 values in
    ``[0, 1]``. A DataLoader therefore produces ``[B, T, C, H, W]``.

    ``clip_length`` defaults to 17 and, by default, must satisfy ``T = 4k + 1``
    so it is compatible with SwiftVR's causal temporal pooling protocol.
    Spatial crop coordinates are sampled in HQ/LR space and multiplied by
    ``scale`` for HR. ``load_hq=False`` omits HQ decoding and the returned
    ``hq`` tensor while preserving identical LR/HR view sampling.
    """

    def __init__(
        self,
        manifests: str | Path | Sequence[str | Path],
        *,
        split: str | None = None,
        training: bool = True,
        clip_length: int = 17,
        crop_size: int | Sequence[int] | None = None,
        scale: int = 3,
        load_hq: bool = True,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.0,
        require_4k_plus_1: bool = True,
        drop_short_sequences: bool = False,
        path_root: str | Path | None = None,
        verify_paths: bool = False,
    ) -> None:
        super().__init__()
        self.training = bool(training)
        self.clip_length = int(clip_length)
        self.crop_size = _normalize_crop_size(crop_size)
        self.scale = int(scale)
        self.load_hq = bool(load_hq)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.vertical_flip_probability = float(vertical_flip_probability)
        self.require_4k_plus_1 = bool(require_4k_plus_1)

        if self.clip_length <= 0:
            raise ValueError(f"clip_length must be positive, got {self.clip_length}")
        if self.require_4k_plus_1 and self.clip_length % 4 != 1:
            raise ValueError(
                f"SwiftVR clip_length must satisfy T=4k+1, got T={self.clip_length}"
            )
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        for name, probability in (
            ("horizontal_flip_probability", self.horizontal_flip_probability),
            ("vertical_flip_probability", self.vertical_flip_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {probability}")

        loaded = read_triplet_manifests(
            manifests,
            split=split,
            path_root=path_root,
            verify_paths=verify_paths,
        )
        short = [record for record in loaded if record.frame_count < self.clip_length]
        if short and not drop_short_sequences:
            examples = ", ".join(
                f"{record.sample_id}({record.frame_count})" for record in short[:5]
            )
            raise ValueError(
                f"{len(short)} sequences are shorter than "
                f"clip_length={self.clip_length}; examples: {examples}. "
                "Set drop_short_sequences=True to filter them."
            )
        self.records = [
            record for record in loaded if record.frame_count >= self.clip_length
        ]
        self.dropped_short_count = len(loaded) - len(self.records)
        if not self.records:
            raise RuntimeError("No sequences remain after clip-length filtering")

    def __len__(self) -> int:
        return len(self.records)

    def _temporal_start(self, frame_count: int) -> int:
        maximum = frame_count - self.clip_length
        return _randint_inclusive(maximum) if self.training else maximum // 2

    def _spatial_crop(
        self,
        hq_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        hq_height, hq_width = hq_size
        if self.crop_size is None:
            return 0, 0, hq_height, hq_width
        crop_height, crop_width = self.crop_size
        if crop_height > hq_height or crop_width > hq_width:
            raise ValueError(
                f"crop_size={self.crop_size} exceeds HQ/LR size={hq_size}"
            )
        if self.training:
            top = _randint_inclusive(hq_height - crop_height)
            left = _randint_inclusive(hq_width - crop_width)
        else:
            top = (hq_height - crop_height) // 2
            left = (hq_width - crop_width) // 2
        return top, left, crop_height, crop_width

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[int(index)]
        temporal_start = self._temporal_start(record.frame_count)
        positions = tuple(
            range(temporal_start, temporal_start + self.clip_length)
        )
        selected_indices = tuple(
            record.frame_indices[position] for position in positions
        )

        lr_size = _image_size(record.lr_paths[positions[0]])
        hr_size = _image_size(record.hr_paths[positions[0]])
        hq_size = lr_size
        if self.load_hq:
            hq_size = _image_size(record.hq_paths[positions[0]])
            if hq_size != lr_size:
                raise ValueError(
                    f"{record.sample_id}: HQ/LR size mismatch: {hq_size} vs {lr_size}"
                )
        expected_hr_size = (lr_size[0] * self.scale, lr_size[1] * self.scale)
        if hr_size != expected_hr_size:
            raise ValueError(
                f"{record.sample_id}: expected HR size {expected_hr_size} for "
                f"scale={self.scale}, got {hr_size}"
            )

        hq_box = self._spatial_crop(lr_size)
        top, left, crop_height, crop_width = hq_box
        hr_box = (
            top * self.scale,
            left * self.scale,
            crop_height * self.scale,
            crop_width * self.scale,
        )

        lr = _load_clip(
            record.lr_paths,
            positions=positions,
            expected_size=lr_size,
            crop_box=hq_box,
        )
        hr = _load_clip(
            record.hr_paths,
            positions=positions,
            expected_size=hr_size,
            crop_box=hr_box,
        )
        hq = None
        if self.load_hq:
            hq = _load_clip(
                record.hq_paths,
                positions=positions,
                expected_size=hq_size,
                crop_box=hq_box,
            )

        horizontal_flip = (
            self.training
            and self.horizontal_flip_probability > 0
            and float(torch.rand(()).item()) < self.horizontal_flip_probability
        )
        vertical_flip = (
            self.training
            and self.vertical_flip_probability > 0
            and float(torch.rand(()).item()) < self.vertical_flip_probability
        )
        if horizontal_flip:
            lr = torch.flip(lr, dims=(-1,))
            hr = torch.flip(hr, dims=(-1,))
            if hq is not None:
                hq = torch.flip(hq, dims=(-1,))
        if vertical_flip:
            lr = torch.flip(lr, dims=(-2,))
            hr = torch.flip(hr, dims=(-2,))
            if hq is not None:
                hq = torch.flip(hq, dims=(-2,))

        result: dict[str, object] = {
            "lr": lr,
            "hr": hr,
            "sample_id": record.sample_id,
            "record_uid": f"{record.variant}:{record.sample_id}",
            "variant": record.variant,
            "split": record.split,
            "source_manifest": record.source_manifest,
            "frame_indices": torch.tensor(selected_indices, dtype=torch.int64),
            "temporal_start": temporal_start,
            "crop_top": top,
            "crop_left": left,
            "scale": self.scale,
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip,
        }
        if hq is not None:
            result["hq"] = hq
        return result


def build_triplet_dataloader(
    manifests: str | Path | Sequence[str | Path],
    *,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    seed: int = 0,
    **dataset_kwargs: object,
) -> tuple[TripletVideoDataset, DataLoader]:
    """Construct a Dataset and reproducibly seeded PyTorch DataLoader."""

    dataset = TripletVideoDataset(manifests, **dataset_kwargs)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if shuffle is None:
        shuffle = dataset.training
    if drop_last is None:
        drop_last = dataset.training

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    worker_kwargs = dataloader_worker_kwargs(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers and int(num_workers) > 0,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(drop_last),
        pin_memory=bool(pin_memory),
        generator=generator,
        **worker_kwargs,
    )
    return dataset, loader
