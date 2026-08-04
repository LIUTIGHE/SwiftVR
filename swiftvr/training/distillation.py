"""Teacher-behaviour distillation helpers for prompt-free SwiftVR students.

The first distillation stage keeps the released ReAE and 5B DiT topology fixed,
but removes runtime text/prompt/timestep inputs.  A frozen conditional teacher is
run offline at the deployment endpoint (empty prompt, timestep 1000) and its
velocity predictions are cached for deterministic video views.  Training then
matches the cached velocity without loading a second 5B model on every rank.
"""

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
from torch.utils.data import Dataset

from .forward import decode_reae_clip, encode_reae_clip
from .reference import expand_prompt_embedding, extract_transformer_sample, sha256_file
from .stage3 import temporal_difference_mse


TEACHER_CACHE_FORMAT_VERSION = 1
TEACHER_CACHE_METADATA_FILENAME = "metadata.json"


def _stable_view_seed(base_seed: int, record_index: int, view_index: int) -> int:
    payload = f"{int(base_seed)}:{int(record_index)}:{int(view_index)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value % (2**63 - 1)


class DeterministicTripletViewDataset(Dataset):
    """Expand each triplet record into reproducible random-looking views.

    ``TripletVideoDataset(training=True)`` normally draws temporal starts, crops,
    and flips from global torch RNG state.  This wrapper isolates that RNG inside
    ``fork_rng`` and seeds every ``(record, view)`` independently.  The resulting
    sample is therefore invariant to DataLoader ordering, rank sharding, resume,
    and worker scheduling, which is required for an offline teacher cache.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        views_per_record: int = 1,
        view_seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.views_per_record = int(views_per_record)
        self.view_seed = int(view_seed)
        if self.views_per_record <= 0:
            raise ValueError("views_per_record must be positive")
        if len(dataset) <= 0:
            raise ValueError("The wrapped dataset must be non-empty")

    def __len__(self) -> int:
        return len(self.dataset) * self.views_per_record

    def decode_index(self, index: int) -> tuple[int, int, int]:
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if not 0 <= normalized < len(self):
            raise IndexError(index)
        record_index, view_index = divmod(normalized, self.views_per_record)
        seed = _stable_view_seed(self.view_seed, record_index, view_index)
        return record_index, view_index, seed

    def __getitem__(self, index: int) -> dict[str, object]:
        record_index, view_index, seed = self.decode_index(index)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            sample = self.dataset[record_index]
        if not isinstance(sample, Mapping):
            raise TypeError("The wrapped dataset must return a mapping")
        result = dict(sample)
        result["distillation_index"] = record_index * self.views_per_record + view_index
        result["distillation_record_index"] = record_index
        result["distillation_view_index"] = view_index
        result["distillation_view_seed"] = seed
        return result


def _batch_value(batch: Mapping[str, object], name: str, index: int) -> object:
    value = batch.get(name)
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.item() if item.ndim == 0 else item
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[index]
    raise TypeError(f"batch {name!r} must be a tensor or sequence")


def distillation_sample_identity(
    batch: Mapping[str, object],
    index: int,
) -> dict[str, object]:
    frame_indices = _batch_value(batch, "frame_indices", index)
    if not isinstance(frame_indices, torch.Tensor) or frame_indices.ndim != 1:
        raise TypeError("batch frame_indices must collate to a [B,T] tensor")
    identity = {
        "distillation_index": int(_batch_value(batch, "distillation_index", index)),
        "record_uid": str(_batch_value(batch, "record_uid", index)),
        "sample_id": str(_batch_value(batch, "sample_id", index)),
        "variant": str(_batch_value(batch, "variant", index)),
        "frame_indices": [int(value) for value in frame_indices.tolist()],
        "crop_top": int(_batch_value(batch, "crop_top", index)),
        "crop_left": int(_batch_value(batch, "crop_left", index)),
        "horizontal_flip": bool(_batch_value(batch, "horizontal_flip", index)),
        "vertical_flip": bool(_batch_value(batch, "vertical_flip", index)),
        "view_index": int(_batch_value(batch, "distillation_view_index", index)),
        "view_seed": int(_batch_value(batch, "distillation_view_seed", index)),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["key"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return identity


@torch.inference_mode()
def conditional_teacher_velocity(
    *,
    reae,
    transformer,
    prompt_embedding: torch.Tensor,
    lq_input: torch.Tensor,
    timestep: float = 1000.0,
) -> dict[str, torch.Tensor]:
    """Run only the conditional teacher encoder and DiT endpoint."""

    z_lq_ntchw = encode_reae_clip(reae, lq_input, require_4k_plus_1=True)
    z_lq = z_lq_ntchw.permute(0, 2, 1, 3, 4).contiguous()
    batch_size = int(z_lq.shape[0])
    prompt = expand_prompt_embedding(prompt_embedding, batch_size).to(
        device=z_lq.device,
        dtype=z_lq.dtype,
    )
    timesteps = torch.full(
        (batch_size,),
        float(timestep),
        device=z_lq.device,
        dtype=torch.float32,
    )
    velocity = extract_transformer_sample(
        transformer(z_lq, timesteps, prompt, return_dict=True)
    )
    if velocity.shape != z_lq.shape:
        raise ValueError(
            f"Teacher velocity shape {tuple(velocity.shape)} does not match "
            f"input latent shape {tuple(z_lq.shape)}"
        )
    return {"velocity": velocity, "z_lq": z_lq}


class TeacherVelocityCache:
    """Immutable offline teacher targets with strict deterministic-view checks."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        metadata_path = self.root / TEACHER_CACHE_METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Teacher cache metadata must be a JSON object")
        if int(metadata.get("format_version", -1)) != TEACHER_CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported teacher cache format: {metadata.get('format_version')}"
            )
        samples = metadata.get("samples")
        if not isinstance(samples, list):
            raise TypeError("Teacher cache metadata samples must be a list")
        self.metadata = metadata
        self.samples_by_index: dict[int, Mapping[str, object]] = {}
        for item in samples:
            if not isinstance(item, Mapping):
                raise TypeError("Teacher cache sample entries must be mappings")
            sample_index = int(item.get("distillation_index", -1))
            if sample_index < 0 or sample_index in self.samples_by_index:
                raise ValueError(f"Invalid/duplicate distillation index {sample_index}")
            self.samples_by_index[sample_index] = item
        if len(self.samples_by_index) != int(metadata.get("sample_count", -1)):
            raise ValueError("Teacher cache sample_count does not match metadata entries")

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
        saved_hashes = self.metadata.get("manifest_sha256")
        if not isinstance(saved_hashes, Mapping):
            differences.append("cache is missing manifest_sha256")
        else:
            for manifest in manifests:
                path = Path(manifest).expanduser().resolve()
                if saved_hashes.get(str(path)) != sha256_file(path):
                    differences.append(f"manifest hash mismatch: {path}")
        if differences:
            raise ValueError(
                "Teacher cache configuration differs:\n  " + "\n  ".join(differences)
            )

    def load(
        self,
        identity: Mapping[str, object],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        sample_index = int(identity["distillation_index"])
        item = self.samples_by_index.get(sample_index)
        if item is None:
            raise KeyError(f"Teacher cache has no sample index {sample_index}")
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
                    f"Teacher cache identity mismatch for index={sample_index}, "
                    f"field={field}: cache={item.get(field)!r}, "
                    f"current={identity.get(field)!r}"
                )
        filename = item.get("file")
        if not isinstance(filename, str):
            raise TypeError(f"Teacher cache file missing for index {sample_index}")
        tensors = load_file(str(self.root / filename), device="cpu")
        if "velocity" not in tensors:
            raise KeyError(f"Teacher cache sample {filename} lacks 'velocity'")
        velocity = tensors["velocity"].to(device=device)
        return velocity if dtype is None else velocity.to(dtype=dtype)

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
        targets = [
            self.load(
                distillation_sample_identity(batch, index),
                device=device,
                dtype=dtype,
            )
            for index in range(int(frame_indices.shape[0]))
        ]
        return torch.stack(targets, dim=0)


@dataclass
class DistillationMetricAccumulator:
    sum_squared_error: float = 0.0
    sum_teacher_squared: float = 0.0
    sum_student_squared: float = 0.0
    sum_dot: float = 0.0
    elements: int = 0
    samples: int = 0

    @torch.no_grad()
    def update(self, student: torch.Tensor, teacher: torch.Tensor) -> None:
        if student.shape != teacher.shape:
            raise ValueError(
                f"Velocity shape mismatch: {tuple(student.shape)} vs {tuple(teacher.shape)}"
            )
        student_f = student.float()
        teacher_f = teacher.float()
        difference = student_f - teacher_f
        self.sum_squared_error += float(difference.square().sum().item())
        self.sum_teacher_squared += float(teacher_f.square().sum().item())
        self.sum_student_squared += float(student_f.square().sum().item())
        self.sum_dot += float((student_f * teacher_f).sum().item())
        self.elements += int(student.numel())
        self.samples += int(student.shape[0])

    def compute(self) -> dict[str, float | int]:
        if self.elements <= 0:
            raise RuntimeError("No teacher-distillation samples were accumulated")
        mse = self.sum_squared_error / self.elements
        teacher_power = self.sum_teacher_squared / self.elements
        denominator = max(
            math.sqrt(self.sum_teacher_squared * self.sum_student_squared), 1e-12
        )
        return {
            "velocity_mse": mse,
            "velocity_rmse": math.sqrt(mse),
            "velocity_normalized_mse": mse / max(teacher_power, 1e-12),
            "velocity_relative_l2": math.sqrt(self.sum_squared_error)
            / max(math.sqrt(self.sum_teacher_squared), 1e-12),
            "velocity_cosine": self.sum_dot / denominator,
            "velocity_elements": self.elements,
            "samples": self.samples,
        }


def velocity_distillation_objective(
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    *,
    student_prediction: torch.Tensor | None = None,
    teacher_prediction: torch.Tensor | None = None,
    velocity_mse_weight: float = 1.0,
    velocity_cosine_weight: float = 1.0,
    output_l1_weight: float = 0.0,
    output_temporal_weight: float = 0.0,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Composite endpoint teacher-matching objective.

    The velocity MSE is normalized by detached teacher power, so its scale is
    comparable across clips.  Cosine similarity is computed per sample over the
    complete latent velocity.  Optional RGB losses match the teacher output, not
    GT, and therefore preserve a concrete teacher-generated solution rather than
    encouraging a conditional mean.
    """

    if student_velocity.shape != teacher_velocity.shape:
        raise ValueError(
            f"Velocity shape mismatch: {tuple(student_velocity.shape)} vs "
            f"{tuple(teacher_velocity.shape)}"
        )
    student_f = student_velocity.float()
    teacher_f = teacher_velocity.detach().float()
    raw_mse = F.mse_loss(student_f, teacher_f)
    teacher_power = teacher_f.square().mean().detach()
    normalized_mse = raw_mse / teacher_power.clamp_min(float(epsilon))
    student_flat = student_f.flatten(1)
    teacher_flat = teacher_f.flatten(1)
    cosine = F.cosine_similarity(
        student_flat,
        teacher_flat,
        dim=1,
        eps=float(epsilon),
    ).mean()
    cosine_loss = 1.0 - cosine

    zero = raw_mse.new_zeros(())
    output_l1 = zero
    output_temporal = zero
    if output_l1_weight != 0.0 or output_temporal_weight != 0.0:
        if student_prediction is None or teacher_prediction is None:
            raise ValueError("Output teacher losses require both RGB predictions")
        if student_prediction.shape != teacher_prediction.shape:
            raise ValueError(
                f"Output shape mismatch: {tuple(student_prediction.shape)} vs "
                f"{tuple(teacher_prediction.shape)}"
            )
        output_l1 = F.l1_loss(student_prediction.float(), teacher_prediction.float())
        output_temporal = temporal_difference_mse(
            student_prediction,
            teacher_prediction,
        )

    total = (
        float(velocity_mse_weight) * normalized_mse
        + float(velocity_cosine_weight) * cosine_loss
        + float(output_l1_weight) * output_l1
        + float(output_temporal_weight) * output_temporal
    )
    return {
        "loss": total,
        "velocity_mse": raw_mse,
        "velocity_normalized_mse": normalized_mse,
        "velocity_cosine": cosine,
        "velocity_cosine_loss": cosine_loss,
        "teacher_velocity_power": teacher_power,
        "output_l1": output_l1,
        "output_temporal_mse": output_temporal,
    }


@torch.no_grad()
def decode_teacher_prediction(
    *,
    reae,
    z_lq: torch.Tensor,
    teacher_velocity: torch.Tensor,
    output_frames: int,
) -> torch.Tensor:
    if z_lq.shape != teacher_velocity.shape:
        raise ValueError(
            f"Teacher decode shape mismatch: {tuple(z_lq.shape)} vs "
            f"{tuple(teacher_velocity.shape)}"
        )
    z_teacher = z_lq - teacher_velocity
    return decode_reae_clip(
        reae,
        z_teacher.permute(0, 2, 1, 3, 4).contiguous(),
        output_frames=int(output_frames),
        clamp=False,
    )


class SwiftVRVelocityDistillationForward(torch.nn.Module):
    """Encode LQ and predict the prompt-free endpoint velocity without decoding."""

    def __init__(
        self,
        reae,
        transformer,
        *,
        attention_backend: str = "sdpa",
        prepare_transformer: bool = True,
    ) -> None:
        super().__init__()
        from .forward import prepare_prompt_free_no_time_transformer_for_training

        self.reae = reae
        self.transformer = transformer
        if prepare_transformer:
            prepare_prompt_free_no_time_transformer_for_training(
                self.transformer,
                attention_backend=attention_backend,
            )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        from .forward import forward_prompt_free_no_time_training, prepare_training_batch

        prepared = prepare_training_batch(batch)
        lq_input = prepared["lq_input"]
        target = prepared["target"]
        if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("Prepared batch is missing lq_input/target tensors")
        z_lq_ntchw = encode_reae_clip(self.reae, lq_input, require_4k_plus_1=True)
        z_lq = z_lq_ntchw.permute(0, 2, 1, 3, 4).contiguous()
        velocity = forward_prompt_free_no_time_training(self.transformer, z_lq)
        if velocity.shape != z_lq.shape:
            raise ValueError(
                f"Student velocity shape {tuple(velocity.shape)} does not match "
                f"input latent shape {tuple(z_lq.shape)}"
            )
        return {
            "velocity": velocity,
            "z_lq": z_lq,
            "target": target,
            "lq_input": lq_input,
        }


def decode_student_prediction(
    *,
    reae,
    z_lq: torch.Tensor,
    student_velocity: torch.Tensor,
    output_frames: int,
) -> torch.Tensor:
    if z_lq.shape != student_velocity.shape:
        raise ValueError(
            f"Student decode shape mismatch: {tuple(z_lq.shape)} vs "
            f"{tuple(student_velocity.shape)}"
        )
    z_student = z_lq - student_velocity
    return decode_reae_clip(
        reae,
        z_student.permute(0, 2, 1, 3, 4).contiguous(),
        output_frames=int(output_frames),
        clamp=False,
    )
