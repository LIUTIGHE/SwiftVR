"""Lightweight resumable checkpoints for SwiftVR training.

The released folded SwiftVR checkpoint is treated as an immutable base model.
Training checkpoints therefore store only parameters whose ``requires_grad`` flag
is enabled, plus optimizer and optional AMP GradScaler state. This keeps
adapter/LoRA checkpoints small while still supporting an exact mixed-precision
resume.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


FORMAT_VERSION = 1
TRAINABLE_WEIGHTS_FILENAME = "trainable.safetensors"
OPTIMIZER_FILENAME = "optimizer.pt"
METADATA_FILENAME = "metadata.json"


def trainable_named_parameters(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return trainable parameters in stable ``named_parameters`` order."""
    named = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if not named:
        raise RuntimeError("The module has no trainable parameters")
    return named


def cast_trainable_parameters(
    module: nn.Module,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    """Cast only trainable parameters to a stable optimizer dtype.

    Frozen SwiftVR weights may remain FP16/BF16 for memory-efficient forward
    computation, while adapter/LoRA weights and their optimizer moments stay in
    FP32. This avoids AdamW state/epsilon underflow when updating FP16 parameters.
    Call this before constructing the optimizer and before backward.
    """
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"Trainable parameter dtype must be floating point, got {dtype}")

    named = trainable_named_parameters(module)
    source_dtypes = sorted(
        {str(parameter.dtype).removeprefix("torch.") for _, parameter in named}
    )
    with torch.no_grad():
        for _, parameter in named:
            if parameter.grad is not None:
                raise RuntimeError(
                    "cast_trainable_parameters must be called before backward or "
                    "after gradients are cleared"
                )
            parameter.data = parameter.detach().to(
                device=parameter.device,
                dtype=dtype,
            ).contiguous()

    return {
        "parameter_tensors": len(named),
        "parameter_elements": sum(parameter.numel() for _, parameter in named),
        "source_dtypes": source_dtypes,
        "target_dtype": str(dtype).removeprefix("torch."),
    }


def capture_trainable_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone the current trainable parameter values to contiguous CPU tensors."""
    return {
        name: parameter.detach().to(device="cpu").contiguous().clone()
        for name, parameter in trainable_named_parameters(module)
    }


def parameter_update_summary(
    before: Mapping[str, torch.Tensor],
    module: nn.Module,
) -> dict[str, object]:
    """Summarize the finite parameter delta relative to ``before``."""
    current = dict(trainable_named_parameters(module))
    expected_names = tuple(before)
    current_names = tuple(current)
    if expected_names != current_names:
        missing = sorted(set(expected_names) - set(current_names))
        unexpected = sorted(set(current_names) - set(expected_names))
        raise ValueError(
            "Trainable parameter set changed while measuring an update: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    squared_norm = 0.0
    maximum = 0.0
    tensors_changed = 0
    elements_changed = 0
    nonfinite = 0

    for name in expected_names:
        previous = before[name]
        value = current[name].detach().to(device="cpu", dtype=previous.dtype)
        if tuple(value.shape) != tuple(previous.shape):
            raise ValueError(
                f"Shape changed for {name}: {tuple(previous.shape)} -> {tuple(value.shape)}"
            )
        delta = value.float() - previous.float()
        finite = torch.isfinite(delta)
        nonfinite += int((~finite).sum().item())
        changed_count = int((delta != 0).sum().item())
        if changed_count:
            tensors_changed += 1
            elements_changed += changed_count
        if finite.any():
            finite_delta = delta[finite]
            squared_norm += float(torch.sum(finite_delta * finite_delta).item())
            maximum = max(maximum, float(finite_delta.abs().max().item()))

    return {
        "parameter_tensors": len(expected_names),
        "parameter_elements": sum(tensor.numel() for tensor in before.values()),
        "changed_tensors": tensors_changed,
        "changed_elements": elements_changed,
        "global_l2": math.sqrt(squared_norm),
        "max_abs": maximum,
        "nonfinite_elements": nonfinite,
    }


def _load_torch_file(path: Path, *, map_location: str | torch.device = "cpu"):
    """Load a trusted local optimizer checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_delta_checkpoint(
    output_dir: str | Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    metadata: Optional[Mapping[str, object]] = None,
    grad_scaler=None,
) -> dict[str, object]:
    """Save trainable weights, optimizer/scaler state, and resume metadata."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    named = trainable_named_parameters(module)
    parameter_names = [name for name, _ in named]
    trainable_state = {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in named
    }

    weights_path = output / TRAINABLE_WEIGHTS_FILENAME
    optimizer_path = output / OPTIMIZER_FILENAME
    metadata_path = output / METADATA_FILENAME

    save_file(trainable_state, str(weights_path))
    scaler_state = None if grad_scaler is None else grad_scaler.state_dict()
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "parameter_names": parameter_names,
            "optimizer": optimizer.state_dict(),
            "grad_scaler": scaler_state,
        },
        optimizer_path,
    )

    user_metadata = dict(metadata or {})
    result: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "step": int(step),
        "trainable_parameter_names": parameter_names,
        "trainable_parameter_tensors": len(named),
        "trainable_parameter_elements": sum(
            parameter.numel() for _, parameter in named
        ),
        "weights_file": TRAINABLE_WEIGHTS_FILENAME,
        "optimizer_file": OPTIMIZER_FILENAME,
        "grad_scaler_saved": grad_scaler is not None,
        "metadata": user_metadata,
    }
    _write_json(metadata_path, result)
    return result


def load_delta_checkpoint(
    checkpoint_dir: str | Path,
    module: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
    grad_scaler=None,
) -> dict[str, object]:
    """Restore trainable parameters and optional optimizer/scaler state exactly."""
    root = Path(checkpoint_dir).expanduser().resolve()
    metadata_path = root / METADATA_FILENAME
    weights_path = root / TRAINABLE_WEIGHTS_FILENAME
    optimizer_path = root / OPTIMIZER_FILENAME

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing training metadata: {metadata_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing trainable weights: {weights_path}")
    if (optimizer is not None or grad_scaler is not None) and not optimizer_path.is_file():
        raise FileNotFoundError(f"Missing optimizer state: {optimizer_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected JSON object in {metadata_path}")
    if int(metadata.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version: {metadata.get('format_version')}"
        )

    current_named = trainable_named_parameters(module)
    current_names = [name for name, _ in current_named]
    saved_names = list(metadata.get("trainable_parameter_names", []))
    if strict and current_names != saved_names:
        missing = sorted(set(saved_names) - set(current_names))
        unexpected = sorted(set(current_names) - set(saved_names))
        raise ValueError(
            "Trainable parameter set does not match checkpoint: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    weights = load_file(str(weights_path), device="cpu")
    if strict and set(weights) != set(saved_names):
        missing = sorted(set(saved_names) - set(weights))
        unexpected = sorted(set(weights) - set(saved_names))
        raise ValueError(
            "Trainable weight file does not match metadata: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    parameter_by_name = dict(current_named)
    with torch.no_grad():
        for name in saved_names:
            if name not in parameter_by_name:
                if strict:
                    raise KeyError(f"Missing trainable parameter in model: {name}")
                continue
            if name not in weights:
                if strict:
                    raise KeyError(f"Missing trainable tensor in checkpoint: {name}")
                continue
            parameter = parameter_by_name[name]
            tensor = weights[name]
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: model={tuple(parameter.shape)}, "
                    f"checkpoint={tuple(tensor.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))

    if optimizer is not None or grad_scaler is not None:
        payload = _load_torch_file(optimizer_path, map_location=map_location)
        if not isinstance(payload, dict) or "optimizer" not in payload:
            raise ValueError(f"Invalid optimizer checkpoint: {optimizer_path}")
        optimizer_names = list(payload.get("parameter_names", []))
        if strict and optimizer_names != saved_names:
            raise ValueError(
                "Optimizer parameter order does not match checkpoint metadata"
            )
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if grad_scaler is not None:
            scaler_state = payload.get("grad_scaler")
            if scaler_state is None:
                if strict:
                    raise ValueError("Checkpoint does not contain AMP GradScaler state")
            else:
                grad_scaler.load_state_dict(scaler_state)

    return metadata
