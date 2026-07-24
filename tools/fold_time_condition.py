#!/usr/bin/env python3
"""Fold SwiftVR's fixed one-step timestep condition into model parameters.

Input must be a checkpoint produced by ``convert_prompt_free_checkpoint.py``.
The output retains ReAE and prompt-free adapters, removes every
``condition_embedder`` tensor, and adds the fixed timestep modulation to each
block and to the final output ``scale_shift_table``.

Example:

    python tools/fold_time_condition.py \
        --source checkpoints_prompt_free \
        --output checkpoints_prompt_free_no_time \
        --timestep 1000
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from swiftvr.models.transformer_prompt_free import WanTimeEmbedding


WEIGHT_FILENAME = "diffusion_pytorch_model.safetensors"
CONFIG_FILENAME = "config.json"
REPORT_FILENAME = "time_folding_report.json"
TIME_PREFIX = "condition_embedder."


def _model_dimensions(config: Mapping[str, object]) -> tuple[int, int, int]:
    required = ("num_layers", "num_attention_heads", "attention_head_dim", "freq_dim")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing required transformer config fields: {missing}")

    num_layers = int(config["num_layers"])
    inner_dim = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    freq_dim = int(config["freq_dim"])
    if num_layers <= 0 or inner_dim <= 0 or freq_dim <= 0:
        raise ValueError("num_layers, inner_dim, and freq_dim must be positive")
    return num_layers, inner_dim, freq_dim


def _extract_time_state(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    time_state = {
        key[len(TIME_PREFIX):]: tensor.float()
        for key, tensor in state_dict.items()
        if key.startswith(TIME_PREFIX)
    }
    required_prefixes = ("time_embedder.", "time_proj.")
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in time_state)
    ]
    if missing:
        raise ValueError(
            "Source checkpoint has no complete prompt-free time module; "
            f"missing prefixes: {missing}"
        )
    return time_state


def compute_fixed_time_condition(
    state_dict: Mapping[str, torch.Tensor],
    config: Mapping[str, object],
    *,
    timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``temb[C]`` and ``time_proj[6,C]`` in float32."""

    _, inner_dim, freq_dim = _model_dimensions(config)
    module = WanTimeEmbedding(
        dim=inner_dim,
        time_freq_dim=freq_dim,
        time_proj_dim=inner_dim * 6,
    ).float().eval()
    module.load_state_dict(_extract_time_state(state_dict), strict=True)

    with torch.no_grad():
        temb, timestep_proj = module(
            torch.tensor([float(timestep)], dtype=torch.float32),
            output_dtype=torch.float32,
        )
    return temb[0].contiguous(), timestep_proj[0].reshape(6, inner_dim).contiguous()


def fold_state_dict(
    source_state: Mapping[str, torch.Tensor],
    config: Mapping[str, object],
    *,
    timestep: float = 1000.0,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Fold the fixed timestep and return no-time tensors plus a report."""

    num_layers, inner_dim, _ = _model_dimensions(config)
    temb, block_modulation = compute_fixed_time_condition(
        source_state,
        config,
        timestep=timestep,
    )

    folded: dict[str, torch.Tensor] = {}
    dropped_keys: list[str] = []
    folded_keys: list[str] = []

    for key, tensor in source_state.items():
        if key.startswith(TIME_PREFIX):
            dropped_keys.append(key)
            continue
        folded[key] = tensor

    for layer_index in range(num_layers):
        key = f"blocks.{layer_index}.scale_shift_table"
        if key not in folded:
            raise KeyError(f"Missing block modulation tensor: {key}")
        source = folded[key]
        expected_shape = (1, 6, inner_dim)
        if tuple(source.shape) != expected_shape:
            raise ValueError(
                f"Unexpected shape for {key}: {tuple(source.shape)}, "
                f"expected {expected_shape}"
            )
        folded[key] = (
            source.float() + block_modulation.unsqueeze(0)
        ).to(dtype=source.dtype).contiguous()
        folded_keys.append(key)

    output_key = "scale_shift_table"
    if output_key not in folded:
        raise KeyError(f"Missing output modulation tensor: {output_key}")
    output_table = folded[output_key]
    expected_output_shape = (1, 2, inner_dim)
    if tuple(output_table.shape) != expected_output_shape:
        raise ValueError(
            f"Unexpected shape for {output_key}: {tuple(output_table.shape)}, "
            f"expected {expected_output_shape}"
        )
    folded[output_key] = (
        output_table.float() + temb.view(1, 1, inner_dim)
    ).to(dtype=output_table.dtype).contiguous()
    folded_keys.append(output_key)

    report: dict[str, object] = {
        "fixed_timestep": float(timestep),
        "num_layers": num_layers,
        "inner_dim": inner_dim,
        "source_tensor_count": len(source_state),
        "dropped_tensor_count": len(dropped_keys),
        "folded_tensor_count": len(folded_keys),
        "output_tensor_count": len(folded),
        "dropped_keys": sorted(dropped_keys),
        "folded_keys": sorted(folded_keys),
        "time_embedding_l2": float(torch.linalg.vector_norm(temb).item()),
        "block_modulation_l2": float(
            torch.linalg.vector_norm(block_modulation).item()
        ),
    }
    return folded, report


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _prepare_output_path(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Pass --overwrite to replace it."
            )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=False)


def fold_checkpoint(
    source_root: Path,
    output_root: Path,
    *,
    transformer_subfolder: str = "transformer",
    timestep: float = 1000.0,
    copy_reae: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root:
        raise ValueError("Source and output checkpoint directories must be different")

    source_transformer = source_root / transformer_subfolder
    config_path = source_transformer / CONFIG_FILENAME
    weight_path = source_transformer / WEIGHT_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Transformer config not found: {config_path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"Transformer weights not found: {weight_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("time_condition_folded", False):
        raise ValueError("Source checkpoint is already time-folded")
    source_state = load_file(str(weight_path), device="cpu")
    folded_state, report = fold_state_dict(
        source_state,
        config,
        timestep=timestep,
    )

    _prepare_output_path(output_root, overwrite=overwrite)
    output_transformer = output_root / transformer_subfolder
    output_transformer.mkdir(parents=True, exist_ok=False)

    output_config = dict(config)
    output_config["_class_name"] = "WanTransformer3DModelPromptFreeNoTime"
    output_config["time_condition_folded"] = True
    output_config["folded_timestep"] = float(timestep)
    (output_transformer / CONFIG_FILENAME).write_text(
        json.dumps(output_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = _read_safetensors_metadata(weight_path)
    metadata.update(
        {
            "swiftvr_variant": "prompt_free_no_time",
            "time_condition_folded": "true",
            "folded_timestep": str(float(timestep)),
        }
    )
    save_file(
        folded_state,
        str(output_transformer / WEIGHT_FILENAME),
        metadata=metadata,
    )

    copied_files: list[str] = []
    if copy_reae:
        reae_path = source_root / "reae.safetensors"
        if not reae_path.is_file():
            raise FileNotFoundError(f"ReAE checkpoint not found: {reae_path}")
        shutil.copy2(reae_path, output_root / reae_path.name)
        copied_files.append(reae_path.name)

    report.update(
        {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "transformer_subfolder": transformer_subfolder,
            "copied_files": copied_files,
        }
    )
    (output_root / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument(
        "--copy-reae",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = fold_checkpoint(
        args.source,
        args.output,
        transformer_subfolder=args.transformer_subfolder,
        timestep=args.timestep,
        copy_reae=args.copy_reae,
        overwrite=args.overwrite,
    )
    print(
        "Folded SwiftVR timestep condition: "
        f"dropped={report['dropped_tensor_count']} "
        f"folded={report['folded_tensor_count']} "
        f"output={report['output_root']}"
    )


if __name__ == "__main__":
    main()
