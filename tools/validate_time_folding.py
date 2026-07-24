#!/usr/bin/env python3
"""Numerically compare prompt-free and time-folded SwiftVR checkpoints.

The two multi-billion-parameter models are loaded sequentially so they do not
reside on the GPU at the same time. A deterministic small latent is used to
compare predicted degradation velocity before any ReAE decoding.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from swiftvr.models.transformer_prompt_free import WanTransformer3DModelPromptFree
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _load_config(root: Path, subfolder: str) -> dict[str, object]:
    path = root / subfolder / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _release(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def validate(
    source_root: Path,
    folded_root: Path,
    *,
    transformer_subfolder: str,
    device: str,
    dtype: torch.dtype,
    frames: int,
    height: int,
    width: int,
    seed: int,
) -> dict[str, object]:
    source_config = _load_config(source_root, transformer_subfolder)
    folded_config = _load_config(folded_root, transformer_subfolder)
    if not folded_config.get("time_condition_folded", False):
        raise ValueError("Folded checkpoint config has time_condition_folded != true")

    in_channels = int(source_config["in_channels"])
    patch_t, patch_h, patch_w = map(int, source_config["patch_size"])
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            "Latent dimensions must be divisible by patch_size: "
            f"shape={(frames, height, width)} patch={(patch_t, patch_h, patch_w)}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    latent_cpu = torch.randn(
        1,
        in_channels,
        frames,
        height,
        width,
        generator=generator,
        dtype=torch.float32,
    )
    device_obj = torch.device(device)
    latent = latent_cpu.to(device=device_obj, dtype=dtype)

    source = WanTransformer3DModelPromptFree.from_pretrained(
        str(source_root),
        subfolder=transformer_subfolder,
        torch_dtype=dtype,
    ).to(device_obj).eval()
    source_parameter_count = sum(parameter.numel() for parameter in source.parameters())
    timestep = torch.tensor([1000.0], device=device_obj, dtype=torch.float32)
    with torch.inference_mode():
        source_output = source(latent.clone(), timestep).sample.float().cpu()
    _release(source)

    folded = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(folded_root),
        subfolder=transformer_subfolder,
        torch_dtype=dtype,
    ).to(device_obj).eval()
    folded_parameter_count = sum(parameter.numel() for parameter in folded.parameters())
    folded_names = tuple(name for name, _ in folded.named_parameters())
    with torch.inference_mode():
        folded_output = folded(latent.clone()).sample.float().cpu()
    _release(folded)

    difference = folded_output - source_output
    abs_difference = difference.abs()
    mae = float(abs_difference.mean().item())
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    max_abs = float(abs_difference.max().item())
    signal_rms = float(torch.sqrt(torch.mean(source_output.square())).item())
    relative_rmse = rmse / max(signal_rms, 1e-12)
    psnr = float("inf") if rmse == 0 else 20.0 * math.log10(1.0 / rmse)

    return {
        "dtype": str(dtype).removeprefix("torch."),
        "latent_shape": list(latent_cpu.shape),
        "source_parameter_count": source_parameter_count,
        "folded_parameter_count": folded_parameter_count,
        "removed_parameter_count": source_parameter_count - folded_parameter_count,
        "folded_has_condition_embedder": any(
            "condition_embedder" in name for name in folded_names
        ),
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "signal_rms": signal_rms,
        "relative_rmse": relative_rmse,
        "unit_range_psnr": psnr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--folded", type=Path, required=True)
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=tuple(_DTYPES),
        default="float16",
    )
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(
        args.source.resolve(),
        args.folded.resolve(),
        transformer_subfolder=args.transformer_subfolder,
        device=args.device,
        dtype=_DTYPES[args.dtype],
        frames=args.frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
