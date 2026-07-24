#!/usr/bin/env python3
"""Convert a released SwiftVR checkpoint to the prompt-free student layout.

The converter works directly on state-dict tensors and never instantiates the
5B transformer. It keeps the ReAE checkpoint, removes text/image conditioning
and cross-attention parameters, and adds one zero-output residual adapter per
transformer block.

Example:

    python tools/convert_prompt_free_checkpoint.py \
        --source checkpoints/ \
        --output checkpoints_prompt_free/ \
        --adapter-dim 128

Expected source layout:

    checkpoints/
    ├── reae.safetensors
    ├── prompt_embedding.safetensors
    └── transformer/
        ├── config.json
        └── diffusion_pytorch_model.safetensors

The output intentionally omits ``prompt_embedding.safetensors``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


WEIGHT_FILENAME = "diffusion_pytorch_model.safetensors"
CONFIG_FILENAME = "config.json"
REPORT_FILENAME = "conversion_report.json"

_DROP_PREFIXES = (
    "condition_embedder.text_embedder.",
    "condition_embedder.image_embedder.",
)
_DROP_INFIXES = (
    ".attn2.",
    ".norm2.",
    # These projections are runtime-only products of fuse_projections().
    ".to_qkv.",
    ".to_kv.",
    ".to_added_kv.",
)


def should_drop_source_key(key: str) -> bool:
    """Return whether a released teacher tensor is absent from the student."""

    return key.startswith(_DROP_PREFIXES) or any(part in key for part in _DROP_INFIXES)


def _adapter_state_for_layer(
    *,
    layer_index: int,
    inner_dim: int,
    adapter_dim: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Create the state of one PromptFreeResidualAdapter without a full model."""

    if inner_dim <= 0:
        raise ValueError(f"inner_dim must be positive, got {inner_dim}")
    if adapter_dim <= 0:
        raise ValueError(f"adapter_dim must be positive, got {adapter_dim}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + layer_index)

    down_weight = torch.empty(adapter_dim, inner_dim, dtype=torch.float32)
    torch.nn.init.kaiming_uniform_(
        down_weight,
        a=math.sqrt(5),
        generator=generator,
    )
    bound = 1.0 / math.sqrt(inner_dim)
    down_bias = torch.empty(adapter_dim, dtype=torch.float32)
    torch.nn.init.uniform_(
        down_bias,
        -bound,
        bound,
        generator=generator,
    )

    prefix = f"blocks.{layer_index}.prompt_free_adapter"
    return {
        f"{prefix}.norm.weight": torch.ones(inner_dim, dtype=torch.float32),
        f"{prefix}.norm.bias": torch.zeros(inner_dim, dtype=torch.float32),
        f"{prefix}.down.weight": down_weight,
        f"{prefix}.down.bias": down_bias,
        # Zero output makes the converted model exactly the hard-CA-removal
        # baseline before adapter training.
        f"{prefix}.up.weight": torch.zeros(inner_dim, adapter_dim, dtype=torch.float32),
        f"{prefix}.up.bias": torch.zeros(inner_dim, dtype=torch.float32),
    }


def _model_dimensions(config: Mapping[str, object]) -> tuple[int, int]:
    try:
        num_layers = int(config["num_layers"])
        num_heads = int(config["num_attention_heads"])
        head_dim = int(config["attention_head_dim"])
    except KeyError as exc:
        raise KeyError(f"Missing required transformer config field: {exc.args[0]}") from exc

    if num_layers <= 0 or num_heads <= 0 or head_dim <= 0:
        raise ValueError(
            "num_layers, num_attention_heads, and attention_head_dim must all be positive"
        )
    return num_layers, num_heads * head_dim


def validate_source_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    num_layers: int,
) -> None:
    """Catch wrong directories and incomplete source checkpoints early."""

    required_prefixes = (
        "patch_embedding.",
        "condition_embedder.time_embedder.",
        "condition_embedder.time_proj.",
        "blocks.0.attn1.",
        "blocks.0.ffn.",
        f"blocks.{num_layers - 1}.",
        "proj_out.",
    )
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in state_dict)
    ]
    if missing:
        raise ValueError(
            "Source state dict does not look like a complete SwiftVR transformer; "
            f"missing prefixes: {missing}"
        )


def convert_state_dict(
    source_state: Mapping[str, torch.Tensor],
    config: Mapping[str, object],
    *,
    adapter_dim: int,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Convert tensors and return an auditable conversion summary."""

    num_layers, inner_dim = _model_dimensions(config)
    validate_source_state_dict(source_state, num_layers=num_layers)

    converted: dict[str, torch.Tensor] = {}
    dropped_keys: list[str] = []

    for key, tensor in source_state.items():
        if should_drop_source_key(key):
            dropped_keys.append(key)
            continue
        converted[key] = tensor.contiguous()

    added_keys: list[str] = []
    for layer_index in range(num_layers):
        adapter_state = _adapter_state_for_layer(
            layer_index=layer_index,
            inner_dim=inner_dim,
            adapter_dim=adapter_dim,
            seed=seed,
        )
        collisions = sorted(set(adapter_state).intersection(converted))
        if collisions:
            raise ValueError(f"Adapter keys already exist in source checkpoint: {collisions}")
        converted.update(adapter_state)
        added_keys.extend(adapter_state)

    report: dict[str, object] = {
        "source_tensor_count": len(source_state),
        "kept_tensor_count": len(source_state) - len(dropped_keys),
        "dropped_tensor_count": len(dropped_keys),
        "added_tensor_count": len(added_keys),
        "output_tensor_count": len(converted),
        "num_layers": num_layers,
        "inner_dim": inner_dim,
        "adapter_dim": adapter_dim,
        "adapter_seed": seed,
        "dropped_keys": sorted(dropped_keys),
        "added_keys": sorted(added_keys),
    }
    return converted, report


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


def convert_checkpoint(
    source_root: Path,
    output_root: Path,
    *,
    transformer_subfolder: str = "transformer",
    adapter_dim: int = 128,
    seed: int = 0,
    copy_reae: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    """Convert one complete SwiftVR checkpoint directory."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root:
        raise ValueError("Source and output checkpoint directories must be different")

    source_transformer = source_root / transformer_subfolder
    source_config_path = source_transformer / CONFIG_FILENAME
    source_weight_path = source_transformer / WEIGHT_FILENAME

    if not source_config_path.is_file():
        raise FileNotFoundError(f"Transformer config not found: {source_config_path}")
    if not source_weight_path.is_file():
        raise FileNotFoundError(f"Transformer weights not found: {source_weight_path}")

    config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_state = load_file(str(source_weight_path), device="cpu")
    converted_state, report = convert_state_dict(
        source_state,
        config,
        adapter_dim=adapter_dim,
        seed=seed,
    )

    _prepare_output_path(output_root, overwrite=overwrite)
    output_transformer = output_root / transformer_subfolder
    output_transformer.mkdir(parents=True, exist_ok=False)

    output_config = dict(config)
    output_config["_class_name"] = "WanTransformer3DModelPromptFree"
    output_config["adapter_dim"] = int(adapter_dim)
    (output_transformer / CONFIG_FILENAME).write_text(
        json.dumps(output_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = _read_safetensors_metadata(source_weight_path)
    metadata.update(
        {
            "swiftvr_variant": "prompt_free",
            "adapter_dim": str(adapter_dim),
            "adapter_seed": str(seed),
        }
    )
    save_file(
        converted_state,
        str(output_transformer / WEIGHT_FILENAME),
        metadata=metadata,
    )

    copied_files: list[str] = []
    reae_path = source_root / "reae.safetensors"
    if copy_reae:
        if not reae_path.is_file():
            raise FileNotFoundError(
                f"--copy-reae was requested but the file was not found: {reae_path}"
            )
        shutil.copy2(reae_path, output_root / reae_path.name)
        copied_files.append(reae_path.name)

    report.update(
        {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "transformer_subfolder": transformer_subfolder,
            "copied_files": copied_files,
            "omitted_files": ["prompt_embedding.safetensors"],
        }
    )
    (output_root / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Released checkpoint root")
    parser.add_argument("--output", type=Path, required=True, help="Converted checkpoint root")
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--copy-reae",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy reae.safetensors to the converted checkpoint (default: true)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_checkpoint(
        args.source,
        args.output,
        transformer_subfolder=args.transformer_subfolder,
        adapter_dim=args.adapter_dim,
        seed=args.seed,
        copy_reae=args.copy_reae,
        overwrite=args.overwrite,
    )
    print(
        "Converted SwiftVR checkpoint: "
        f"kept={report['kept_tensor_count']} "
        f"dropped={report['dropped_tensor_count']} "
        f"added={report['added_tensor_count']} "
        f"output={report['output_root']}"
    )


if __name__ == "__main__":
    main()
