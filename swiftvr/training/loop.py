"""Single-process training-loop utilities for SwiftVR.

The helpers in this module make the first adapter-training stage reproducible and
resumable without depending on a distributed framework. Exact mid-epoch resume is
supported for ``num_workers=0`` by rebuilding the epoch permutation, skipping the
already-consumed batches, and then restoring Python/NumPy/PyTorch RNG state.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from .checkpoint import trainable_named_parameters

TRAINER_STATE_FILENAME = "trainer_state.pt"
LATEST_CHECKPOINT_FILENAME = "latest.json"
_CUDA_RNG_SCOPE_CURRENT_DEVICE = "current_device"


@dataclass(frozen=True)
class TrainingCursor:
    """Location of the next batch to process."""

    global_step: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    def advance(self, *, batches_per_epoch: int) -> "TrainingCursor":
        if batches_per_epoch <= 0:
            raise ValueError(
                f"batches_per_epoch must be positive, got {batches_per_epoch}"
            )
        next_batch = self.batch_in_epoch + 1
        next_epoch = self.epoch
        if next_batch >= batches_per_epoch:
            next_epoch += next_batch // batches_per_epoch
            next_batch %= batches_per_epoch
        return TrainingCursor(
            global_step=self.global_step + 1,
            epoch=next_epoch,
            batch_in_epoch=next_batch,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrainingCursor":
        return cls(
            global_step=int(value.get("global_step", 0)),
            epoch=int(value.get("epoch", 0)),
            batch_in_epoch=int(value.get("batch_in_epoch", 0)),
        )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU PyTorch, and all visible CUDA devices."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, object]:
    """Capture RNG state for the CPU and the current process-local CUDA device.

    DDP processes see the same visible CUDA device list but each process executes
    only on its ``LOCAL_RANK`` device. Saving every visible device therefore makes
    each rank checkpoint redundant and couples resume to ``CUDA_VISIBLE_DEVICES``.
    The current format saves exactly one CUDA generator state and restores it onto
    whichever logical CUDA device the process selects at resume time.
    """

    cuda_states: list[torch.Tensor] = []
    cuda_device: int | None = None
    cuda_scope = "none"
    if torch.cuda.is_available():
        cuda_device = int(torch.cuda.current_device())
        cuda_states = [torch.cuda.get_rng_state(cuda_device).cpu()]
        cuda_scope = _CUDA_RNG_SCOPE_CURRENT_DEVICE

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": cuda_states,
        "torch_cuda_scope": cuda_scope,
        "torch_cuda_device": cuda_device,
    }


def _as_cuda_rng_tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("CUDA RNG state entries must be tensors")
    return value.to(device="cpu", dtype=torch.uint8)


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore RNG state, including legacy all-visible-device checkpoints.

    New checkpoints contain one CUDA state with ``torch_cuda_scope`` set to
    ``current_device``. Legacy checkpoints contain one state per device that was
    visible when they were saved. For legacy data, the entry matching the current
    logical device is selected when available; a single legacy entry is also
    accepted and mapped to the current device. This removes the old requirement
    that checkpoint and runtime expose the same number of GPUs.
    """

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise KeyError(f"RNG state is missing keys: {missing}")

    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]

    torch_cpu = state["torch_cpu"]
    if not isinstance(torch_cpu, torch.Tensor):
        raise TypeError("torch_cpu RNG state must be a tensor")
    torch.set_rng_state(torch_cpu.to(device="cpu", dtype=torch.uint8))

    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, (list, tuple)):
        raise TypeError("torch_cuda RNG state must be a sequence")
    if not torch.cuda.is_available():
        return
    if not cuda_states:
        raise ValueError("Checkpoint does not contain a CUDA RNG state")

    current_device = int(torch.cuda.current_device())
    scope = state.get("torch_cuda_scope")
    if scope == _CUDA_RNG_SCOPE_CURRENT_DEVICE:
        if len(cuda_states) != 1:
            raise ValueError(
                "Current-device CUDA RNG checkpoints must contain exactly one state, "
                f"got {len(cuda_states)}"
            )
        selected = cuda_states[0]
    elif len(cuda_states) == 1:
        # Compatibility with manually migrated or early single-device states.
        selected = cuda_states[0]
    elif current_device < len(cuda_states):
        # Legacy format from torch.cuda.get_rng_state_all(). The logical device
        # index identifies the generator used by this process before migration.
        selected = cuda_states[current_device]
    else:
        raise ValueError(
            "Legacy CUDA RNG checkpoint has no state for the current logical device: "
            f"checkpoint_states={len(cuda_states)}, current_device={current_device}"
        )

    torch.cuda.set_rng_state(
        _as_cuda_rng_tensor(selected),
        device=current_device,
    )


def _load_torch_file(path: Path, *, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_trainer_state(
    checkpoint_dir: str | Path,
    *,
    cursor: TrainingCursor,
    config: Mapping[str, object],
    rng_state: Optional[Mapping[str, object]] = None,
) -> Path:
    """Atomically save dataloader cursor, run fingerprint, and RNG state."""

    root = Path(checkpoint_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / TRAINER_STATE_FILENAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "cursor": asdict(cursor),
        "config": dict(config),
        "rng_state": dict(rng_state or capture_rng_state()),
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_trainer_state(
    checkpoint_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Load and validate the trainer-side state file."""

    root = Path(checkpoint_dir).expanduser().resolve()
    path = root / TRAINER_STATE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing trainer state: {path}")
    payload = _load_torch_file(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid trainer state payload: {path}")
    for key in ("cursor", "config", "rng_state"):
        if key not in payload:
            raise ValueError(f"Trainer state is missing {key!r}: {path}")
    if not isinstance(payload["cursor"], Mapping):
        raise TypeError("Trainer cursor must be a mapping")
    payload = dict(payload)
    payload["cursor"] = TrainingCursor.from_mapping(payload["cursor"])
    return payload


def write_latest_checkpoint(run_dir: str | Path, checkpoint_dir: str | Path) -> Path:
    """Atomically update the run-local latest-checkpoint pointer."""

    run_root = Path(run_dir).expanduser().resolve()
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        stored_path = checkpoint.relative_to(run_root).as_posix()
    except ValueError:
        stored_path = str(checkpoint)
    pointer = run_root / LATEST_CHECKPOINT_FILENAME
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"checkpoint": stored_path}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(pointer)
    return pointer


def resolve_resume_checkpoint(
    resume: str | Path,
    *,
    run_dir: str | Path,
) -> Path:
    """Resolve an explicit checkpoint directory or the ``latest`` pointer."""

    run_root = Path(run_dir).expanduser().resolve()
    if str(resume) != "latest":
        checkpoint = Path(resume).expanduser().resolve()
    else:
        pointer = run_root / LATEST_CHECKPOINT_FILENAME
        if not pointer.is_file():
            raise FileNotFoundError(f"Missing latest-checkpoint pointer: {pointer}")
        value = json.loads(pointer.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("checkpoint"), str):
            raise ValueError(f"Invalid latest-checkpoint pointer: {pointer}")
        candidate = Path(value["checkpoint"])
        checkpoint = candidate if candidate.is_absolute() else (run_root / candidate)
        checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint is not a directory: {checkpoint}")
    return checkpoint


def append_jsonl(path: str | Path, record: Mapping[str, object]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def skip_batches(iterator: Iterator[object], count: int) -> None:
    """Skip batches without loading Dataset items when the iterator permits it.

    PyTorch DataLoader iterators expose their batch-sampler iterator internally as
    ``_sampler_iter``. Advancing that iterator preserves the epoch permutation but
    avoids calling ``Dataset.__getitem__`` for already-consumed batches. This is
    critical for exact mid-epoch resume of video training, where replaying hundreds
    of batches would otherwise decode thousands of frames before the first new
    optimizer step. Generic iterators fall back to consuming their items normally.
    """

    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    sampler_iterator = getattr(iterator, "_sampler_iter", None)
    target = sampler_iterator if sampler_iterator is not None else iterator
    for index in range(count):
        try:
            next(target)
        except StopIteration as exc:
            raise RuntimeError(
                f"Cannot skip {count} batches; iterator ended after {index}"
            ) from exc


def build_fp32_adamw(
    module: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Build AdamW and require FP32 trainable/master parameters."""

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    parameters = [parameter for _, parameter in trainable_named_parameters(module)]
    bad = sorted(
        {
            str(parameter.dtype)
            for parameter in parameters
            if parameter.dtype != torch.float32
        }
    )
    if bad:
        raise RuntimeError(
            "AdamW trainable parameters must be FP32; call "
            f"cast_trainable_parameters first. Found {bad}"
        )
    return torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        eps=float(eps),
        foreach=False,
    )


def build_grad_scaler(device: torch.device, runtime_dtype: torch.dtype):
    enabled = device.type == "cuda" and runtime_dtype == torch.float16
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def optimizer_state_summary(
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    dtype_counts: dict[str, int] = {}
    tensor_count = 0
    nonfinite = 0
    for state in optimizer.state.values():
        for value in state.values():
            if not isinstance(value, torch.Tensor):
                continue
            tensor_count += 1
            dtype_name = str(value.dtype).removeprefix("torch.")
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
            if value.is_floating_point():
                nonfinite += int((~torch.isfinite(value.detach())).sum().item())
    return {
        "tensor_count": tensor_count,
        "dtype_counts": dtype_counts,
        "nonfinite_elements": nonfinite,
    }
