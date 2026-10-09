"""FlashVSR-style direct packed-LQ condition variant for Stage-B1.

This is an isolated experimental variant. It keeps the compact decoder trunk and
resize-convolution RGB head unchanged, but removes the learned
3072->32 condition bottleneck. Packed RGB condition channels are concatenated
with the SR latent directly before the decoder input fusion convolution, matching
the central TC-decoder conditioning pattern used by FlashVSR.

The source ResizeConv checkpoint is transferred everywhere except the input
fusion convolution. That convolution receives an approximate initialization
obtained by composing the old condition projection with the old condition slice
of the input convolution. The composition is not function-exact because Clamp
sits between projection and convolution in the source graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .tiny_conditional_decoder import (
    CONFIG_FILENAME,
    SUPPORTED_FORMAT_VERSIONS,
    WEIGHTS_FILENAME,
    pack_rgb_condition,
)
from .tiny_conditional_decoder_resize_conv import ResizeConvTinyConditionalDecoder


DIRECT_CONDITION_SCHEME = "packed_rgb_direct_concat_v1"
DIRECT_CONDITION_MODE = "packed_rgb_direct_concat"


class DirectConditionResizeConvTinyConditionalDecoder(
    ResizeConvTinyConditionalDecoder
):
    """Resize-conv Tiny decoder with direct packed-RGB condition fusion."""

    def __init__(
        self,
        *,
        condition_channels: int | None = None,
        **kwargs,
    ) -> None:
        temporal_factor = int(kwargs.get("temporal_factor", 4))
        spatial_factor = int(kwargs.get("spatial_factor", 16))
        packed_condition_channels = 3 * temporal_factor * spatial_factor**2
        if condition_channels is not None and int(condition_channels) != packed_condition_channels:
            raise ValueError(
                "Direct-condition decoder requires condition_channels equal to packed RGB "
                f"width {packed_condition_channels}, got {condition_channels}"
            )

        # Build the inherited topology with a one-channel placeholder so we do
        # not allocate a useless 3072->3072 projection. The placeholder
        # projection and inherited input convolution are replaced below.
        super().__init__(condition_channels=1, **kwargs)

        if not hasattr(self, "condition_projection"):
            raise RuntimeError("Inherited Tiny decoder is missing condition_projection")
        del self.condition_projection

        old_input = self.decoder[1]
        if not isinstance(old_input, nn.Conv2d):
            raise RuntimeError("Unexpected Tiny decoder input topology")
        self.condition_channels = packed_condition_channels
        self.decoder[1] = nn.Conv2d(
            self.latent_channels + self.condition_channels,
            self.channels[0],
            kernel_size=3,
            padding=1,
            bias=old_input.bias is not None,
        )

    @property
    def packed_condition_channels(self) -> int:
        return 3 * self.temporal_factor * self.spatial_factor**2

    @property
    def config_dict(self) -> dict[str, object]:
        config = dict(super().config_dict)
        config.update(
            {
                "class_name": type(self).__name__,
                "condition_channels": self.packed_condition_channels,
                "condition_injection": DIRECT_CONDITION_MODE,
                "condition_surgery_scheme": DIRECT_CONDITION_SCHEME,
                "source_condition_projection_removed": True,
            }
        )
        return config

    def project_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """Pack RGB directly to the latent grid without a learned bottleneck."""
        return pack_rgb_condition(
            condition,
            temporal_factor=self.temporal_factor,
            spatial_factor=self.spatial_factor,
        )

    def initialize_from_resizeconv_decoder(
        self,
        source: ResizeConvTinyConditionalDecoder,
    ) -> dict[str, object]:
        """Transfer an R4-style decoder and initialize direct condition fusion.

        All tensors after the input convolution, including the compact trunk and
        resize-conv RGB head, are copied exactly.

        For the new input convolution, latent-channel weights are copied exactly.
        Condition-channel weights are initialized by linearly composing the old
        3072->32 projection with the old 32-channel condition slice of the input
        3x3 convolution. Because the source graph applies ``Clamp`` after the
        projection while this direct variant applies it to packed RGB before the
        input convolution, this fold is only an initialization, not an exact
        functional reparameterization.
        """
        if not isinstance(source, ResizeConvTinyConditionalDecoder):
            raise TypeError("source must be a ResizeConvTinyConditionalDecoder")

        for attribute in (
            "latent_channels",
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
                    f"Source/direct-condition topology mismatch for {attribute}: "
                    f"source={getattr(source, attribute)!r}, "
                    f"target={getattr(self, attribute)!r}"
                )

        projection = source.condition_projection
        source_input = source.decoder[1]
        target_input = self.decoder[1]
        if not isinstance(projection, nn.Conv2d) or projection.kernel_size != (1, 1):
            raise TypeError("Source condition projection must be a 1x1 Conv2d")
        if not isinstance(source_input, nn.Conv2d) or not isinstance(target_input, nn.Conv2d):
            raise TypeError("Source/target decoder input layers must be Conv2d")
        if int(projection.in_channels) != self.packed_condition_channels:
            raise ValueError(
                "Source packed-condition width mismatch: "
                f"expected {self.packed_condition_channels}, got {projection.in_channels}"
            )
        if int(source_input.in_channels) != self.latent_channels + int(projection.out_channels):
            raise ValueError("Source input convolution does not match latent+projected condition")

        source_state = source.state_dict()
        target_state = self.state_dict()
        transferable: dict[str, torch.Tensor] = {}
        transferred_elements = 0
        for name, target in target_state.items():
            if name.startswith("decoder.1."):
                continue
            value = source_state.get(name)
            if value is None:
                raise KeyError(f"Source checkpoint is missing transferable tensor {name!r}")
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for transferable tensor {name}: "
                    f"source={tuple(value.shape)}, target={tuple(target.shape)}"
                )
            transferable[name] = value
            transferred_elements += int(value.numel())

        incompatible = self.load_state_dict(transferable, strict=False)
        expected_missing = {"decoder.1.weight"}
        if target_input.bias is not None:
            expected_missing.add("decoder.1.bias")
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected direct-condition transfer result: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

        latent_channels = self.latent_channels
        source_weight = source_input.weight.detach()
        latent_weight = source_weight[:, :latent_channels]
        projected_condition_weight = source_weight[:, latent_channels:]
        projection_weight = projection.weight.detach()[:, :, 0, 0]

        # [O, Cproj, Kh, Kw] x [Cproj, Cpacked]
        # -> [O, Cpacked, Kh, Kw]
        direct_condition_weight = torch.einsum(
            "ockl,ci->oikl",
            projected_condition_weight.float(),
            projection_weight.float(),
        )

        with torch.no_grad():
            target_input.weight.zero_()
            target_input.weight[:, :latent_channels].copy_(
                latent_weight.to(
                    device=target_input.weight.device,
                    dtype=target_input.weight.dtype,
                )
            )
            target_input.weight[:, latent_channels:].copy_(
                direct_condition_weight.to(
                    device=target_input.weight.device,
                    dtype=target_input.weight.dtype,
                )
            )

            if target_input.bias is not None:
                if source_input.bias is None:
                    target_input.bias.zero_()
                else:
                    folded_bias = source_input.bias.detach().float()
                    if projection.bias is not None:
                        # Interior-point linearized fold. Border pixels and the
                        # Clamp nonlinearity prevent exact global equivalence.
                        folded_bias = folded_bias + torch.einsum(
                            "ockl,c->o",
                            projected_condition_weight.float(),
                            projection.bias.detach().float(),
                        )
                    target_input.bias.copy_(
                        folded_bias.to(
                            device=target_input.bias.device,
                            dtype=target_input.bias.dtype,
                        )
                    )

        source_parameter_count = sum(p.numel() for p in source.parameters())
        target_parameter_count = sum(p.numel() for p in self.parameters())
        return {
            "scheme": DIRECT_CONDITION_SCHEME,
            "condition_injection": DIRECT_CONDITION_MODE,
            "packed_condition_channels": self.packed_condition_channels,
            "source_projected_condition_channels": int(projection.out_channels),
            "source_condition_projection_parameters": sum(
                p.numel() for p in projection.parameters()
            ),
            "source_input_conv_parameters": sum(
                p.numel() for p in source_input.parameters()
            ),
            "new_input_conv_parameters": sum(
                p.numel() for p in target_input.parameters()
            ),
            "source_decoder_parameters": source_parameter_count,
            "new_decoder_parameters": target_parameter_count,
            "parameter_delta": target_parameter_count - source_parameter_count,
            "transferred_nonfusion_tensors": len(transferable),
            "transferred_nonfusion_elements": transferred_elements,
            "latent_input_weights_exactly_copied": True,
            "condition_weight_initialization": "linear_composition_projection_into_input_conv",
            "condition_fold_function_exact": False,
            "non_exact_reason": (
                "source applies Clamp after the learned condition projection; "
                "direct variant applies Clamp to packed RGB before the input convolution"
            ),
        }

    @classmethod
    def from_resizeconv_pretrained(
        cls,
        root: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> tuple["DirectConditionResizeConvTinyConditionalDecoder", dict[str, object]]:
        source = ResizeConvTinyConditionalDecoder.from_pretrained(
            root,
            device=device,
            dtype=dtype,
        )
        model = cls(
            latent_channels=source.latent_channels,
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

    @classmethod
    def from_pretrained(
        cls,
        root: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "DirectConditionResizeConvTinyConditionalDecoder":
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
        if config.get("condition_injection") != DIRECT_CONDITION_MODE:
            raise ValueError(
                "Expected direct packed-RGB condition injection, got "
                f"{config.get('condition_injection')!r}"
            )

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
