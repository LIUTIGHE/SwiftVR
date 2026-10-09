"""Exact-init rich-condition bypass experiment for Stage-B1.

This isolated variant keeps the trained ResizeConv Tiny Decoder path intact and
adds a zero-initialized direct packed-RGB branch at the decoder input feature
level.  The source 3072->32 condition projection is intentionally preserved: it
forms an exact R4 baseline path, while the new 3072->C0 3x3 branch can learn to
use rich packed LQ information without being restricted by the 32-channel
bottleneck.

At initialization the bypass is identically zero, so the model is functionally
identical to the supplied ResizeConv checkpoint.  This makes the experiment a
clean test of whether *additional direct access* to the packed condition helps;
it does not conflate that question with a destructive reparameterization.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .tiny_conditional_decoder import _apply_video_stack, _validate_video, pack_rgb_condition
from .tiny_conditional_decoder_resize_conv import ResizeConvTinyConditionalDecoder


CONDITION_BYPASS_SCHEME = "packed_rgb_zero_init_bypass_v1"
CONDITION_BYPASS_MODE = "packed_rgb_direct_bypass"


class ConditionBypassResizeConvTinyConditionalDecoder(ResizeConvTinyConditionalDecoder):
    """ResizeConv Tiny decoder with an exact-init direct packed-RGB bypass."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.direct_condition_bypass = nn.Conv2d(
            self.packed_condition_channels,
            self.channels[0],
            kernel_size=3,
            padding=1,
            bias=False,
        )
        nn.init.zeros_(self.direct_condition_bypass.weight)

    @property
    def packed_condition_channels(self) -> int:
        return 3 * self.temporal_factor * self.spatial_factor**2

    @property
    def config_dict(self) -> dict[str, object]:
        config = dict(super().config_dict)
        config.update(
            {
                "class_name": type(self).__name__,
                "condition_injection": CONDITION_BYPASS_MODE,
                "condition_bypass_scheme": CONDITION_BYPASS_SCHEME,
                "packed_condition_channels": self.packed_condition_channels,
                "source_condition_projection_preserved": True,
                "direct_condition_bypass_bias": False,
            }
        )
        return config

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

        projected = self.project_condition(condition)
        packed = pack_rgb_condition(
            condition,
            temporal_factor=self.temporal_factor,
            spatial_factor=self.spatial_factor,
        )
        expected = (int(latents.shape[0]), int(latents.shape[1]), int(latents.shape[3]), int(latents.shape[4]))
        for name, value in (("projected condition", projected), ("packed condition", packed)):
            actual = (int(value.shape[0]), int(value.shape[1]), int(value.shape[3]), int(value.shape[4]))
            if actual != expected:
                raise ValueError(
                    f"{name} does not match latent grid: value={tuple(value.shape)} "
                    f"latent={tuple(latents.shape)}"
                )

        # Exact source path through Clamp + the trained R4 input convolution.
        base_input = torch.cat([latents, projected], dim=2)
        prefix = nn.Sequential(self.decoder[0], self.decoder[1])
        base = _apply_video_stack(prefix, base_input)

        # New rich-condition path.  Apply the same stateless Clamp used at the
        # source decoder input, then a zero-init 3x3 mapping directly to C0.
        batch, frames, channels, height, width = packed.shape
        packed_flat = packed.reshape(batch * frames, channels, height, width)
        packed_flat = self.decoder[0](packed_flat)
        direct_flat = self.direct_condition_bypass(packed_flat)
        direct = direct_flat.reshape(batch, frames, self.channels[0], height, width)

        fused = base + direct
        tail = nn.Sequential(*list(self.decoder.children())[2:])
        trunk = _apply_video_stack(tail, fused)

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
                    f"Condition-bypass decoder emitted {pixels.shape[1]} valid frames; "
                    f"requested {output_frames}"
                )
            pixels = pixels[:, :output_frames]
        return pixels.clamp(0.0, 1.0) if clamp else pixels

    def initialize_from_resizeconv_decoder(
        self,
        source: ResizeConvTinyConditionalDecoder,
    ) -> dict[str, object]:
        """Copy the complete source decoder and zero the new bypass."""
        if type(source) is not ResizeConvTinyConditionalDecoder:
            raise TypeError("source must be an exact ResizeConvTinyConditionalDecoder")
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
            "resize_mode",
        ):
            if getattr(self, attribute) != getattr(source, attribute):
                raise ValueError(
                    f"Source/bypass topology mismatch for {attribute}: "
                    f"source={getattr(source, attribute)!r}, target={getattr(self, attribute)!r}"
                )

        source_state = source.state_dict()
        target_state = self.state_dict()
        transferable: dict[str, torch.Tensor] = {}
        transferred_elements = 0
        for name, target in target_state.items():
            if name == "direct_condition_bypass.weight":
                continue
            value = source_state.get(name)
            if value is None:
                raise KeyError(f"Source checkpoint is missing transferable tensor {name!r}")
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: source={tuple(value.shape)} target={tuple(target.shape)}"
                )
            transferable[name] = value
            transferred_elements += int(value.numel())

        incompatible = self.load_state_dict(transferable, strict=False)
        if set(incompatible.missing_keys) != {"direct_condition_bypass.weight"} or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected condition-bypass transfer result: "
                f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )
        with torch.no_grad():
            self.direct_condition_bypass.weight.zero_()

        source_parameters = sum(p.numel() for p in source.parameters())
        target_parameters = sum(p.numel() for p in self.parameters())
        return {
            "scheme": CONDITION_BYPASS_SCHEME,
            "condition_injection": CONDITION_BYPASS_MODE,
            "packed_condition_channels": self.packed_condition_channels,
            "projected_condition_channels": self.condition_channels,
            "source_condition_projection_preserved": True,
            "source_function_exact_at_initialization": True,
            "bypass_initialization": "all_zero",
            "bypass_parameters": self.direct_condition_bypass.weight.numel(),
            "source_decoder_parameters": source_parameters,
            "new_decoder_parameters": target_parameters,
            "parameter_delta": target_parameters - source_parameters,
            "transferred_tensors": len(transferable),
            "transferred_elements": transferred_elements,
        }

    @classmethod
    def from_resizeconv_pretrained(
        cls,
        root: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> tuple["ConditionBypassResizeConvTinyConditionalDecoder", dict[str, object]]:
        source = ResizeConvTinyConditionalDecoder.from_pretrained(
            root, device=device, dtype=dtype
        )
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
            resize_mode=source.resize_mode,
        ).to(device=device, dtype=dtype)
        report = model.initialize_from_resizeconv_decoder(source)
        return model, report
