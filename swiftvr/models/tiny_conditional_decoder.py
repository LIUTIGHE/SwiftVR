"""Tiny conditional decoder for Stage-B SwiftVR compression.

The design follows the central TC-decoder idea used by FlashVSR -- decode the SR
latent together with an aligned low-quality RGB condition -- but adapts it to the
SwiftVR ReAE latent contract:

* temporal compression: 4x;
* spatial compression: 16x;
* latent channels: 48 by default.

Packing RGB directly at 4x16x16 would create 3*4*16*16 = 3072 condition channels.
To keep the replacement genuinely lightweight, the packed condition is first
projected with a 1x1 convolution and only then concatenated with the SR latent.
The decoder itself keeps inexpensive causal MemBlock/TGrow temporal modeling so it
can later replace the ReAE decoder in the existing streaming protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .reae import Clamp, MemBlock, TGrow


CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
FORMAT_VERSION = 1


def _conv(in_channels: int, out_channels: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


def _validate_video(name: str, value: torch.Tensor, *, channels: int | None = None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 5:
        raise ValueError(f"{name} must be [B,T,C,H,W], got {tuple(value.shape)}")
    if channels is not None and int(value.shape[2]) != int(channels):
        raise ValueError(f"{name} must have C={channels}, got C={value.shape[2]}")


def pack_rgb_condition(
    condition: torch.Tensor,
    *,
    temporal_factor: int = 4,
    spatial_factor: int = 16,
) -> torch.Tensor:
    """Pack ``[B,T,3,H,W]`` RGB to the ReAE latent grid.

    Causal decoder alignment needs *prefix* padding.  For a 4k+1 clip, three
    copies of frame 0 are prepended so the first packed temporal group consists
    entirely of frame 0.  After the decoder expands time by four and removes its
    three causal warm-up frames, the remaining first output aligns with the true
    frame 0.  Middle streaming chunks are multiples of four and require no pad.
    """

    _validate_video("condition", condition, channels=3)
    temporal_factor = int(temporal_factor)
    spatial_factor = int(spatial_factor)
    if temporal_factor <= 0 or spatial_factor <= 0:
        raise ValueError("packing factors must be positive")

    batch, frames, channels, height, width = condition.shape
    if height % spatial_factor or width % spatial_factor:
        raise ValueError(
            f"Condition size {height}x{width} must be divisible by "
            f"spatial_factor={spatial_factor}"
        )

    prefix = (-int(frames)) % temporal_factor
    if prefix:
        condition = torch.cat(
            [
                condition[:, :1].expand(-1, prefix, -1, -1, -1),
                condition,
            ],
            dim=1,
        )
    padded_frames = int(condition.shape[1])

    flat = condition.reshape(batch * padded_frames, channels, height, width)
    packed = F.pixel_unshuffle(flat, spatial_factor)
    packed_channels = int(packed.shape[1])
    packed_h, packed_w = int(packed.shape[2]), int(packed.shape[3])
    packed = packed.reshape(
        batch,
        padded_frames // temporal_factor,
        temporal_factor,
        packed_channels,
        packed_h,
        packed_w,
    )
    return packed.reshape(
        batch,
        padded_frames // temporal_factor,
        temporal_factor * packed_channels,
        packed_h,
        packed_w,
    ).contiguous()


def _apply_video_stack(stack: nn.Sequential, video: torch.Tensor) -> torch.Tensor:
    """Differentiable whole-clip execution for MemBlock/TGrow decoder stacks."""

    _validate_video("decoder input", video)
    batch, frames, channels, height, width = video.shape
    hidden = video.reshape(batch * frames, channels, height, width)

    for index, layer in enumerate(stack):
        if isinstance(layer, MemBlock):
            _, channels, height, width = hidden.shape
            current = hidden.reshape(batch, frames, channels, height, width)
            past = torch.cat(
                [torch.zeros_like(current[:, :1]), current[:, :-1]],
                dim=1,
            )
            hidden = layer(
                hidden,
                past.reshape(batch * frames, channels, height, width),
            )
        elif isinstance(layer, TGrow):
            hidden = layer(hidden)
        else:
            hidden = layer(hidden)

        if hidden.shape[0] % batch:
            raise RuntimeError(
                f"Decoder layer {index} produced leading dimension {hidden.shape[0]} "
                f"for batch={batch}"
            )
        frames = int(hidden.shape[0] // batch)

    _, channels, height, width = hidden.shape
    return hidden.reshape(batch, frames, channels, height, width)


class TinyConditionalDecoder(nn.Module):
    """Causal RGB-condition + SR-latent decoder for the SwiftVR latent contract."""

    def __init__(
        self,
        *,
        latent_channels: int = 48,
        condition_channels: int = 32,
        channels: Sequence[int] = (192, 128, 64, 32),
        blocks_per_stage: Sequence[int] = (2, 2, 2, 1),
        temporal_factor: int = 4,
        spatial_factor: int = 16,
        patch_size: int = 2,
        frames_to_trim: int = 3,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.condition_channels = int(condition_channels)
        self.channels = tuple(int(value) for value in channels)
        self.blocks_per_stage = tuple(int(value) for value in blocks_per_stage)
        self.temporal_factor = int(temporal_factor)
        self.spatial_factor = int(spatial_factor)
        self.patch_size = int(patch_size)
        self.frames_to_trim = int(frames_to_trim)

        if self.latent_channels <= 0 or self.condition_channels <= 0:
            raise ValueError("latent/condition channels must be positive")
        if len(self.channels) != 4 or len(self.blocks_per_stage) != 4:
            raise ValueError("channels and blocks_per_stage must each contain four stages")
        if any(value <= 0 for value in self.channels):
            raise ValueError("all decoder stage widths must be positive")
        if any(value < 0 for value in self.blocks_per_stage):
            raise ValueError("blocks_per_stage values must be non-negative")
        if self.temporal_factor != 4:
            raise ValueError("The initial SwiftVR tiny decoder requires temporal_factor=4")
        if self.spatial_factor != 16:
            raise ValueError("The initial SwiftVR tiny decoder requires spatial_factor=16")
        if self.patch_size != 2:
            raise ValueError("The initial SwiftVR tiny decoder requires patch_size=2")
        if self.frames_to_trim < 0:
            raise ValueError("frames_to_trim must be non-negative")

        packed_rgb_channels = 3 * self.temporal_factor * self.spatial_factor**2
        self.condition_projection = nn.Conv2d(
            packed_rgb_channels,
            self.condition_channels,
            kernel_size=1,
            bias=True,
        )

        c0, c1, c2, c3 = self.channels
        layers: list[nn.Module] = [
            Clamp(),
            _conv(self.latent_channels + self.condition_channels, c0),
            nn.ReLU(inplace=True),
        ]
        layers.extend(MemBlock(c0, c0) for _ in range(self.blocks_per_stage[0]))
        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                TGrow(c0, 1),
                _conv(c0, c1, bias=False),
            ]
        )
        layers.extend(MemBlock(c1, c1) for _ in range(self.blocks_per_stage[1]))
        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                TGrow(c1, 2),
                _conv(c1, c2, bias=False),
            ]
        )
        layers.extend(MemBlock(c2, c2) for _ in range(self.blocks_per_stage[2]))
        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                TGrow(c2, 2),
                _conv(c2, c3, bias=False),
            ]
        )
        layers.extend(MemBlock(c3, c3) for _ in range(self.blocks_per_stage[3]))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                _conv(c3, 3 * self.patch_size**2),
            ]
        )
        self.decoder = nn.Sequential(*layers)

    @property
    def config_dict(self) -> dict[str, object]:
        return {
            "format_version": FORMAT_VERSION,
            "class_name": type(self).__name__,
            "latent_channels": self.latent_channels,
            "condition_channels": self.condition_channels,
            "channels": list(self.channels),
            "blocks_per_stage": list(self.blocks_per_stage),
            "temporal_factor": self.temporal_factor,
            "spatial_factor": self.spatial_factor,
            "patch_size": self.patch_size,
            "frames_to_trim": self.frames_to_trim,
        }

    def project_condition(self, condition: torch.Tensor) -> torch.Tensor:
        packed = pack_rgb_condition(
            condition,
            temporal_factor=self.temporal_factor,
            spatial_factor=self.spatial_factor,
        )
        batch, frames, channels, height, width = packed.shape
        flat = packed.reshape(batch * frames, channels, height, width)
        projected = self.condition_projection(flat)
        return projected.reshape(
            batch,
            frames,
            self.condition_channels,
            height,
            width,
        )

    def forward(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor,
        *,
        output_frames: int | None = None,
        clamp: bool = False,
    ) -> torch.Tensor:
        """Decode ``latents[B,F,C,h,w]`` conditioned on ``RGB[B,T,3,H,W]``."""

        _validate_video("latents", latents, channels=self.latent_channels)
        _validate_video("condition", condition, channels=3)
        if int(latents.shape[0]) != int(condition.shape[0]):
            raise ValueError("latents and condition must share batch size")

        condition_latent = self.project_condition(condition)
        if tuple(condition_latent.shape[:2]) != tuple(latents.shape[:2]) or tuple(
            condition_latent.shape[-2:]
        ) != tuple(latents.shape[-2:]):
            raise ValueError(
                "Packed condition does not match latent grid: "
                f"condition={tuple(condition_latent.shape)}, latent={tuple(latents.shape)}"
            )

        hidden = torch.cat([latents, condition_latent], dim=2)
        pixels = _apply_video_stack(self.decoder, hidden)

        batch, frames, channels, height, width = pixels.shape
        flat = F.pixel_shuffle(
            pixels.reshape(batch * frames, channels, height, width),
            self.patch_size,
        )
        pixels = flat.reshape(batch, frames, *flat.shape[1:])

        if self.frames_to_trim:
            pixels = pixels[:, self.frames_to_trim :]
        if output_frames is not None:
            output_frames = int(output_frames)
            if output_frames <= 0:
                raise ValueError("output_frames must be positive")
            if pixels.shape[1] < output_frames:
                raise RuntimeError(
                    f"Tiny decoder emitted {pixels.shape[1]} valid frames; "
                    f"requested {output_frames}"
                )
            pixels = pixels[:, :output_frames]
        return pixels.clamp(0.0, 1.0) if clamp else pixels

    def save_pretrained(self, output_dir: str | Path) -> Path:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / CONFIG_FILENAME).write_text(
            json.dumps(self.config_dict, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        state = {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in self.state_dict().items()
        }
        save_file(state, str(root / WEIGHTS_FILENAME))
        return root

    @classmethod
    def from_pretrained(
        cls,
        root: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "TinyConditionalDecoder":
        root = Path(root).expanduser().resolve()
        config_path = root / CONFIG_FILENAME
        weights_path = root / WEIGHTS_FILENAME
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(config.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(f"Unsupported tiny-decoder format: {config.get('format_version')}")
        kwargs = {
            key: config[key]
            for key in (
                "latent_channels",
                "condition_channels",
                "channels",
                "blocks_per_stage",
                "temporal_factor",
                "spatial_factor",
                "patch_size",
                "frames_to_trim",
            )
        }
        model = cls(**kwargs)
        weights = load_file(str(weights_path), device="cpu")
        model.load_state_dict(weights, strict=True)
        model.to(device=device, dtype=dtype)
        return model
