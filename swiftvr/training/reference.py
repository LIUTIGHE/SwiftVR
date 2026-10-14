"""Conditional-reference caching and three-way Stage-3 evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .stage3 import VideoMetricAccumulator, temporal_difference_mse

CACHE_FORMAT_VERSION = 1
CACHE_METADATA_FILENAME = "metadata.json"


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_sample_key(
    record_uid: str,
    frame_indices: Sequence[int],
    *,
    crop_top: int = 0,
    crop_left: int = 0,
) -> str:
    payload = (
        record_uid
        + ":"
        + ",".join(str(int(value)) for value in frame_indices)
        + f":crop={int(crop_top)},{int(crop_left)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def batch_sample_identity(batch: Mapping[str, object], index: int) -> dict[str, object]:
    record_uids = batch.get("record_uid")
    sample_ids = batch.get("sample_id")
    variants = batch.get("variant")
    frame_indices = batch.get("frame_indices")
    if not isinstance(record_uids, Sequence) or isinstance(record_uids, (str, bytes)):
        raise TypeError("batch record_uid must be a sequence")
    if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)):
        raise TypeError("batch sample_id must be a sequence")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
        raise TypeError("batch variant must be a sequence")
    if not isinstance(frame_indices, torch.Tensor) or frame_indices.ndim != 2:
        raise TypeError("batch frame_indices must be a [B,T] tensor")
    indices = [int(value) for value in frame_indices[index].tolist()]
    record_uid = str(record_uids[index])

    def batch_int(name: str) -> int:
        value = batch.get(name)
        if isinstance(value, torch.Tensor):
            return int(value[index].item())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return int(value[index])
        raise TypeError(f"batch {name} must be a tensor or sequence")

    crop_top = batch_int("crop_top")
    crop_left = batch_int("crop_left")
    return {
        "record_uid": record_uid,
        "sample_id": str(sample_ids[index]),
        "variant": str(variants[index]),
        "frame_indices": indices,
        "crop_top": crop_top,
        "crop_left": crop_left,
        "key": cache_sample_key(
            record_uid, indices, crop_top=crop_top, crop_left=crop_left
        ),
    }


def extract_transformer_sample(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    if isinstance(output, Sequence) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Transformer output does not contain a tensor sample")


def expand_prompt_embedding(prompt_embedding: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize a cached empty-prompt embedding to [B,S,D]."""

    if prompt_embedding.ndim == 2:
        return prompt_embedding.unsqueeze(0).expand(batch_size, -1, -1)
    if prompt_embedding.ndim == 3:
        if prompt_embedding.shape[0] == batch_size:
            return prompt_embedding
        if prompt_embedding.shape[0] == 1:
            return prompt_embedding.expand(batch_size, -1, -1)
    raise ValueError(
        "prompt embedding must have shape [S,D], [1,S,D], or [B,S,D], got "
        f"{tuple(prompt_embedding.shape)}"
    )


@torch.no_grad()
def conditional_reference_forward(
    *,
    reae,
    transformer,
    prompt_embedding: torch.Tensor,
    lq_input: torch.Tensor,
    output_frames: int,
    timestep: float = 1000.0,
) -> dict[str, torch.Tensor]:
    """Run the original fixed-condition SwiftVR endpoint on one aligned clip."""

    from .forward import decode_reae_clip, encode_reae_clip

    z_lq_ntchw = encode_reae_clip(reae, lq_input, require_4k_plus_1=True)
    z_lq = z_lq_ntchw.permute(0, 2, 1, 3, 4).contiguous()
    batch = int(z_lq.shape[0])
    prompt = expand_prompt_embedding(prompt_embedding, batch).to(
        device=z_lq.device,
        dtype=z_lq.dtype,
    )
    timesteps = torch.full(
        (batch,),
        float(timestep),
        device=z_lq.device,
        dtype=torch.float32,
    )
    velocity = extract_transformer_sample(
        transformer(
            z_lq,
            timesteps,
            prompt,
            return_dict=True,
        )
    )
    if velocity.shape != z_lq.shape:
        raise ValueError(
            f"reference velocity shape {tuple(velocity.shape)} does not match "
            f"latent shape {tuple(z_lq.shape)}"
        )
    z_prediction = z_lq - velocity
    prediction = decode_reae_clip(
        reae,
        z_prediction.permute(0, 2, 1, 3, 4).contiguous(),
        output_frames=output_frames,
        clamp=False,
    )
    return {
        "prediction": prediction,
        "prediction_clamped": prediction.clamp(0.0, 1.0),
        "velocity": velocity,
        "z_lq": z_lq,
        "z_prediction": z_prediction,
    }


class ConditionalReferenceCache:
    """Read an immutable reference cache and enforce sample alignment."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        metadata_path = self.root / CACHE_METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Reference cache metadata must be a JSON object")
        if int(value.get("format_version", -1)) != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported reference cache format: {value.get('format_version')}"
            )
        samples = value.get("samples")
        if not isinstance(samples, list):
            raise TypeError("Reference cache metadata samples must be a list")
        self.metadata = value
        self.samples_by_key: dict[str, Mapping[str, object]] = {}
        for item in samples:
            if not isinstance(item, Mapping) or not isinstance(item.get("key"), str):
                raise ValueError("Invalid reference cache sample metadata")
            key = str(item["key"])
            if key in self.samples_by_key:
                raise ValueError(f"Duplicate reference cache key: {key}")
            self.samples_by_key[key] = item

    def load(
        self,
        identity: Mapping[str, object],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        key = str(identity["key"])
        item = self.samples_by_key.get(key)
        if item is None:
            raise KeyError(
                f"Reference cache has no entry for {identity.get('record_uid')} "
                f"frames={identity.get('frame_indices')}"
            )
        if str(item.get("record_uid")) != str(identity.get("record_uid")):
            raise ValueError(f"Reference record_uid mismatch for key {key}")
        expected_indices = [int(value) for value in identity.get("frame_indices", [])]
        saved_indices = [int(value) for value in item.get("frame_indices", [])]
        if saved_indices != expected_indices:
            raise ValueError(f"Reference frame_indices mismatch for key {key}")
        for field in ("crop_top", "crop_left"):
            if int(item.get(field, -1)) != int(identity.get(field, -2)):
                raise ValueError(f"Reference {field} mismatch for key {key}")
        filename = item.get("file")
        if not isinstance(filename, str):
            raise TypeError(f"Reference cache file is missing for key {key}")
        path = self.root / filename
        tensors = load_file(str(path), device="cpu")
        result: dict[str, torch.Tensor] = {}
        for name in ("prediction", "velocity"):
            if name not in tensors:
                raise KeyError(f"{path} is missing tensor {name!r}")
            tensor = tensors[name].to(device=device)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            result[name] = tensor
        return result


@dataclass
class VelocityMetricAccumulator:
    sum_squared: float = 0.0
    elements: int = 0
    student_squared: float = 0.0
    reference_squared: float = 0.0
    dot: float = 0.0

    @torch.no_grad()
    def update(self, student: torch.Tensor, reference: torch.Tensor) -> None:
        if student.shape != reference.shape:
            raise ValueError(
                f"velocity shape mismatch: {tuple(student.shape)} vs "
                f"{tuple(reference.shape)}"
            )
        student = student.float()
        reference = reference.float()
        difference = student - reference
        self.sum_squared += float(difference.square().sum().item())
        self.elements += int(difference.numel())
        self.student_squared += float(student.square().sum().item())
        self.reference_squared += float(reference.square().sum().item())
        self.dot += float((student * reference).sum().item())

    def compute(self) -> dict[str, float]:
        if self.elements <= 0:
            raise RuntimeError("No velocity samples were accumulated")
        mse = self.sum_squared / self.elements
        reference_norm = math.sqrt(self.reference_squared)
        student_norm = math.sqrt(self.student_squared)
        denominator = max(student_norm * reference_norm, 1e-12)
        return {
            "velocity_mse": mse,
            "velocity_rmse": math.sqrt(mse),
            "velocity_relative_l2": math.sqrt(self.sum_squared)
            / max(reference_norm, 1e-12),
            "velocity_cosine": self.dot / denominator,
        }


@torch.no_grad()
def pairwise_video_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    accumulator = VideoMetricAccumulator()
    accumulator.update(prediction, target, clamp=True)
    result = accumulator.compute()
    result["pixel_l1"] = float(
        F.l1_loss(prediction.float(), target.float()).item()
    )
    result["temporal_mse"] = float(
        temporal_difference_mse(prediction, target).item()
    )
    return result
