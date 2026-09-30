"""Isolated resize-convolution output-head variant for the Stage-B1 Tiny Decoder.

The trained TinyConditionalDecoder trunk is preserved exactly through its final
32-channel activation. Only the original 32->12 convolution + PixelShuffle(2)
output mapping is replaced by resize x2 + shared 32->3 RGB convolution. This
removes the four independent subpixel channel groups that were diagnosed as the
main source of the visible 2x2 checkerboard artifact.

This module does not modify the canonical TinyConditionalDecoder class or its
checkpoint format. Resize-conv checkpoints use their own class name/config and
must be loaded through ResizeConvTinyConditionalDecoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .tiny_conditional_decoder import (
    CONFIG_FILENAME,
    SUPPORTED_FORMAT_VERSIONS,
    WEIGHTS_FILENAME,
    TinyConditionalDecoder,
    _apply_video_stack,
    _validate_video,
)


RESIZE_MODES = frozenset({"nearest", "bilinear"})
SURGERY_SCHEME = "pixelshuffle_phase_average_to_resize_conv_v1"


class ResizeConvTinyConditionalDecoder(TinyConditionalDecoder):
    """TinyConditionalDecoder with a phase-shared resize-convolution RGB head."""

    def __init__(self, *, resize_mode: str = "nearest", **kwargs) -> None:
        super().__init__(**kwargs)
        resize_mode = str(resize_mode).lower()
        if resize_mode not in RESIZE_MODES:
            raise ValueError(
                f"Unsupported resize_mode={resize_mode!r}; expected {sorted(RESIZE_MODES)}"
            )
        self.resize_mode = resize_mode

        # Keep the canonical trunk untouched through its final out-of-place ReLU.
        # The last canonical layer is Conv(c3, 3 * patch_size**2), which is the
        # phase-specific subpixel head diagnosed as producing the visible p2 bias.
        original_layers = list(self.decoder.children())
        if len(original_layers) < 2 or not isinstance(original_layers[-1], nn.Conv2d):
            raise ValueError("Unexpected TinyConditionalDecoder output topology")
        old_head = original_layers[-1]
        expected_channels = 3 * self.patch_size**2
        if int(old_head.out_channels) != expected_channels:
            raise ValueError(
                "Unexpected canonical output channels: "
                f"expected {expected_channels}, got {old_head.out_channels}"
            )
        self.decoder = nn.Sequential(*original_layers[:-1])
        self.output_head = nn.Conv2d(
            self.channels[-1],
            3,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    @property
    def config_dict(self) -> dict[str, object]:
        config = dict(super().config_dict)
        config.update(
            {
                "class_name": type(self).__name__,
                "output_head": "resize_conv",
                "resize_mode": self.resize_mode,
                "surgery_scheme": SURGERY_SCHEME,
            }
        )
        return config

    def _resize(self, value: torch.Tensor) -> torch.Tensor:
        if self.resize_mode == "bilinear":
            return F.interpolate(
                value,
                scale_factor=self.patch_size,
                mode="bilinear",
                align_corners=False,
            )
        return F.interpolate(
            value,
            scale_factor=self.patch_size,
            mode="nearest",
        )

    def forward(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor,
        *,
        output_frames: int | None = None,
        clamp: bool = False,
    ) -> torch.Tensor:
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
        trunk = _apply_video_stack(self.decoder, hidden)

        batch, frames, channels, height, width = trunk.shape
        flat = trunk.reshape(batch * frames, channels, height, width)
        flat = self._resize(flat)
        flat = self.output_head(flat)
        pixels = flat.reshape(batch, frames, 3, *flat.shape[-2:])

        if self.frames_to_trim:
            pixels = pixels[:, self.frames_to_trim :]
        if output_frames is not None:
            output_frames = int(output_frames)
            if output_frames <= 0:
                raise ValueError("output_frames must be positive")
            if pixels.shape[1] < output_frames:
                raise RuntimeError(
                    f"Resize-conv Tiny decoder emitted {pixels.shape[1]} valid frames; "
                    f"requested {output_frames}"
                )
            pixels = pixels[:, :output_frames]
        return pixels.clamp(0.0, 1.0) if clamp else pixels

    def initialize_from_pixelshuffle_decoder(
        self,
        source: TinyConditionalDecoder,
    ) -> dict[str, object]:
        """Copy the canonical trunk and phase-average the old subpixel head.

        PixelShuffle(2) interprets canonical Conv output channels as
        [R00,R01,R10,R11,G00,...,B11]. The new shared RGB head is initialized by
        averaging the four phase-specific 3x3 kernels (and biases) for each RGB
        channel. This is intentionally only an initialization: because resize-conv
        applies its kernel after upsampling, exact function equivalence is neither
        possible nor desirable.
        """
        if not isinstance(source, TinyConditionalDecoder):
            raise TypeError("source must be a TinyConditionalDecoder")
        for attribute in (
            "latent_channels",
            "condition_channels",
            "channels",
            "blocks_per_stage",
            "temporal_factor",
            "spatial_factor",
            "patch_size",
            "frames_to_trim",
            "block_mode",
            "block_internal_channels",
        ):
            if getattr(self, attribute) != getattr(source, attribute):
                raise ValueError(
                    f"Source/resize-conv topology mismatch for {attribute}: "
                    f"source={getattr(source, attribute)!r}, target={getattr(self, attribute)!r}"
                )

        source_state = source.state_dict()
        target_state = self.state_dict()
        transferable: dict[str, torch.Tensor] = {}
        transferred_elements = 0
        for name, target in target_state.items():
            if name.startswith("output_head."):
                continue
            value = source_state.get(name)
            if value is None:
                raise KeyError(f"Source checkpoint is missing trunk tensor {name!r}")
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for trunk tensor {name}: "
                    f"source={tuple(value.shape)}, target={tuple(target.shape)}"
                )
            transferable[name] = value
            transferred_elements += int(value.numel())

        incompatible = self.load_state_dict(transferable, strict=False)
        expected_missing = {"output_head.weight", "output_head.bias"}
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected resize-conv trunk transfer result: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

        old_head = source.decoder[-1]
        if not isinstance(old_head, nn.Conv2d):
            raise TypeError("Canonical Tiny decoder must end with Conv2d before PixelShuffle")
        phases = self.patch_size**2
        if int(old_head.out_channels) != 3 * phases:
            raise ValueError(
                f"Expected canonical head out_channels={3 * phases}, got {old_head.out_channels}"
            )
        if tuple(old_head.weight.shape[1:]) != tuple(self.output_head.weight.shape[1:]):
            raise ValueError(
                "Canonical/new RGB head kernel shape mismatch: "
                f"old={tuple(old_head.weight.shape)}, new={tuple(self.output_head.weight.shape)}"
            )

        with torch.no_grad():
            averaged_weight = old_head.weight.detach().reshape(
                3, phases, *old_head.weight.shape[1:]
            ).mean(dim=1)
            self.output_head.weight.copy_(
                averaged_weight.to(
                    device=self.output_head.weight.device,
                    dtype=self.output_head.weight.dtype,
                )
            )
            if old_head.bias is None:
                self.output_head.bias.zero_()
            else:
                averaged_bias = old_head.bias.detach().reshape(3, phases).mean(dim=1)
                self.output_head.bias.copy_(
                    averaged_bias.to(
                        device=self.output_head.bias.device,
                        dtype=self.output_head.bias.dtype,
                    )
                )

        phase_weight = old_head.weight.detach().float().reshape(
            3, phases, *old_head.weight.shape[1:]
        )
        phase_weight_centered = phase_weight - phase_weight.mean(dim=1, keepdim=True)
        phase_weight_rms = float(phase_weight_centered.square().mean().sqrt().item())
        if old_head.bias is None:
            phase_bias_rms = 0.0
        else:
            phase_bias = old_head.bias.detach().float().reshape(3, phases)
            phase_bias_centered = phase_bias - phase_bias.mean(dim=1, keepdim=True)
            phase_bias_rms = float(phase_bias_centered.square().mean().sqrt().item())

        return {
            "scheme": SURGERY_SCHEME,
            "resize_mode": self.resize_mode,
            "transferred_trunk_tensors": len(transferable),
            "transferred_trunk_elements": transferred_elements,
            "source_head_phase_weight_rms": phase_weight_rms,
            "source_head_phase_bias_rms": phase_bias_rms,
            "new_head_parameters": sum(p.numel() for p in self.output_head.parameters()),
        }

    @classmethod
    def from_pixelshuffle_pretrained(
        cls,
        root: str | Path,
        *,
        resize_mode: str = "nearest",
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> tuple["ResizeConvTinyConditionalDecoder", dict[str, object]]:
        source = TinyConditionalDecoder.from_pretrained(root, device=device, dtype=dtype)
        model = cls(
            latent_channels=source.latent_channels,
            condition_channels=source.condition_channels,
            channels=source.channels,
            blocks_per_stage=source.blocks_per_stage,
            temporal_factor=source.temporal_factor,
            spatial_factor=source.spatial_factor,
            patch_size=source.patch_size,
            frames_to_trim=source.frames_to_trim,
            block_mode=source.block_mode,
            block_internal_channels=source.block_internal_channels,
            resize_mode=resize_mode,
        ).to(device=device, dtype=dtype)
        report = model.initialize_from_pixelshuffle_decoder(source)
        return model, report

    @classmethod
    def from_pretrained(
        cls,
        root: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "ResizeConvTinyConditionalDecoder":
        root = Path(root).expanduser().resolve()
        config = json.loads((root / CONFIG_FILENAME).read_text(encoding="utf-8"))
        version = int(config.get("format_version", -1))
        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(f"Unsupported tiny-decoder format: {config.get('format_version')}")
        if config.get("class_name") != cls.__name__:
            raise ValueError(
                f"Expected class_name={cls.__name__!r}, got {config.get('class_name')!r}"
            )
        if config.get("output_head") != "resize_conv":
            raise ValueError(f"Expected resize_conv output head, got {config.get('output_head')!r}")
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
        if version >= 2:
            kwargs["block_mode"] = config.get("block_mode", "dense")
            kwargs["block_internal_channels"] = config.get(
                "block_internal_channels", config["channels"]
            )
        else:
            kwargs["block_mode"] = "dense"
            kwargs["block_internal_channels"] = config["channels"]
        kwargs["resize_mode"] = config.get("resize_mode", "nearest")
        model = cls(**kwargs)
        weights = load_file(str(root / WEIGHTS_FILENAME), device="cpu")
        model.load_state_dict(weights, strict=True)
        model.to(device=device, dtype=dtype)
        return model
