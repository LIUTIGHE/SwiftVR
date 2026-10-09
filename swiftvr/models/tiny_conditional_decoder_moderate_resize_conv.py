"""Moderate-capacity ResizeConv conditional decoder for Stage-B1.

This variant deliberately backs away from the aggressive keep040 configuration
without reverting to the much more expensive ReAE / FlashVSR-width decoder.
The production v1 topology keeps the validated outer interfaces and streaming
contract unchanged while restoring two suspected bottlenecks:

* packed-RGB condition width: 32 -> 128;
* CompactMemBlock hidden widths: (80,48,24,16) -> (128,96,48,24).

Outer stage widths remain (192,128,64,32), block counts remain (2,2,2,1), and the
phase-shared nearest+Conv RGB head is retained.  At 1920x1088 this topology is
analytically 66.548 GMAC/output-frame under the same counting convention that
reproduces the 47.945-GMAC keep040 decoder and 343.108-GMAC ReAE decoder.

Initialization from an existing ResizeConv checkpoint is mathematically
function-preserving.  Existing channels are copied exactly.  Newly added feature
channels retain ordinary random feature-producing weights, but their downstream
connections are initialized to zero.  This avoids a permanently dead all-zero
branch while keeping the initial output equal to the source up to numerical
roundoff; gradients can open the new capacity during training.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .tiny_conditional_decoder_resize_conv import ResizeConvTinyConditionalDecoder
from .tiny_decoder_sparsity import CompactMemBlock


MODERATE_SCHEME = "moderate_tc_v1_function_preserving_widen"
MODERATE_CONDITION_CHANNELS = 128
MODERATE_CHANNELS = (192, 128, 64, 32)
MODERATE_BLOCKS_PER_STAGE = (2, 2, 2, 1)
MODERATE_INTERNAL_CHANNELS = (128, 96, 48, 24)
MODERATE_RESIZE_MODE = "nearest"


def _copy_tensor(dst: torch.Tensor, src: torch.Tensor, *, name: str) -> None:
    if tuple(dst.shape) != tuple(src.shape):
        raise ValueError(f"Shape mismatch for {name}: source={tuple(src.shape)} target={tuple(dst.shape)}")
    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def _widen_condition_path(
    source: ResizeConvTinyConditionalDecoder,
    target: "ModerateResizeConvTinyConditionalDecoder",
) -> dict[str, int]:
    src_proj = source.condition_projection
    dst_proj = target.condition_projection
    old_condition = int(source.condition_channels)
    new_condition = int(target.condition_channels)
    latent = int(source.latent_channels)

    if new_condition < old_condition:
        raise ValueError(
            f"Moderate condition width must not shrink source: {old_condition} -> {new_condition}"
        )
    if src_proj.in_channels != dst_proj.in_channels:
        raise ValueError("Packed condition input width changed unexpectedly")

    src_input = source.decoder[1]
    dst_input = target.decoder[1]
    if not isinstance(src_input, nn.Conv2d) or not isinstance(dst_input, nn.Conv2d):
        raise TypeError("Decoder input layer must be Conv2d")
    if src_input.out_channels != dst_input.out_channels:
        raise ValueError("Moderate v1 requires unchanged stage0 outer width")
    if src_input.in_channels != latent + old_condition:
        raise ValueError("Unexpected source input-conv channel contract")
    if dst_input.in_channels != latent + new_condition:
        raise ValueError("Unexpected target input-conv channel contract")

    with torch.no_grad():
        # Keep the target's ordinary random initialization in the *new* projection
        # rows so those features are nonzero.  The following input conv gates them
        # off initially, making the composed source path function-preserving.
        dst_proj.weight[:old_condition].copy_(
            src_proj.weight.to(device=dst_proj.weight.device, dtype=dst_proj.weight.dtype)
        )
        if src_proj.bias is None or dst_proj.bias is None:
            if src_proj.bias is not None or dst_proj.bias is not None:
                raise ValueError("Source/target condition projection bias mismatch")
        else:
            dst_proj.bias[:old_condition].copy_(
                src_proj.bias.to(device=dst_proj.bias.device, dtype=dst_proj.bias.dtype)
            )

        # Zero only the downstream connections from the new condition features.
        # Existing latent and condition columns are copied to their identical slots.
        dst_input.weight.zero_()
        dst_input.weight[:, :latent].copy_(
            src_input.weight[:, :latent].to(
                device=dst_input.weight.device, dtype=dst_input.weight.dtype
            )
        )
        dst_input.weight[:, latent : latent + old_condition].copy_(
            src_input.weight[:, latent : latent + old_condition].to(
                device=dst_input.weight.device, dtype=dst_input.weight.dtype
            )
        )
        if src_input.bias is None or dst_input.bias is None:
            if src_input.bias is not None or dst_input.bias is not None:
                raise ValueError("Source/target input-conv bias mismatch")
        else:
            _copy_tensor(dst_input.bias, src_input.bias, name="decoder input bias")

    return {
        "source_condition_channels": old_condition,
        "target_condition_channels": new_condition,
        "new_condition_channels": new_condition - old_condition,
    }


def _widen_compact_block(
    source: CompactMemBlock,
    target: CompactMemBlock,
) -> dict[str, int]:
    if int(source.interface_channels) != int(target.interface_channels):
        raise ValueError("CompactMemBlock interface width must remain unchanged")
    old_k = int(source.internal_channels)
    new_k = int(target.internal_channels)
    if new_k < old_k:
        raise ValueError(f"Moderate block must not shrink hidden width: {old_k} -> {new_k}")

    src0: nn.Conv2d = source.conv[0]
    src1: nn.Conv2d = source.conv[2]
    src2: nn.Conv2d = source.conv[4]
    dst0: nn.Conv2d = target.conv[0]
    dst1: nn.Conv2d = target.conv[2]
    dst2: nn.Conv2d = target.conv[4]

    with torch.no_grad():
        # conv0: copy the old hidden rows; retain normal random initialization for
        # the newly added rows so they immediately encode useful nonzero features.
        dst0.weight[:old_k].copy_(
            src0.weight.to(device=dst0.weight.device, dtype=dst0.weight.dtype)
        )
        if src0.bias is not None and dst0.bias is not None:
            dst0.bias[:old_k].copy_(
                src0.bias.to(device=dst0.bias.device, dtype=dst0.bias.dtype)
            )

        # conv1 old rows must ignore new conv0 features to preserve the old path.
        # New rows remain randomly initialized and can build new hidden features.
        dst1.weight[:old_k, :].zero_()
        dst1.weight[:old_k, :old_k].copy_(
            src1.weight.to(device=dst1.weight.device, dtype=dst1.weight.dtype)
        )
        if src1.bias is not None and dst1.bias is not None:
            dst1.bias[:old_k].copy_(
                src1.bias.to(device=dst1.bias.device, dtype=dst1.bias.dtype)
            )

        # conv2 is the output gate.  New hidden columns start at zero, so the
        # widened block initially produces the exact source residual.  These zero
        # columns receive gradients immediately because the new hidden activations
        # are nonzero; the widened path therefore becomes active after optimization.
        dst2.weight.zero_()
        dst2.weight[:, :old_k].copy_(
            src2.weight.to(device=dst2.weight.device, dtype=dst2.weight.dtype)
        )
        if src2.bias is not None and dst2.bias is not None:
            _copy_tensor(dst2.bias, src2.bias, name="compact block output bias")

    return {
        "interface_channels": int(source.interface_channels),
        "source_internal_channels": old_k,
        "target_internal_channels": new_k,
        "new_internal_channels": new_k - old_k,
    }


class ModerateResizeConvTinyConditionalDecoder(ResizeConvTinyConditionalDecoder):
    """Function-preserving moderate-width recovery of the compact Tiny decoder."""

    def __init__(
        self,
        *,
        latent_channels: int = 48,
        condition_channels: int = MODERATE_CONDITION_CHANNELS,
        channels=MODERATE_CHANNELS,
        blocks_per_stage=MODERATE_BLOCKS_PER_STAGE,
        temporal_factor: int = 4,
        spatial_factor: int = 16,
        patch_size: int = 2,
        frames_to_trim: int = 3,
        block_mode: str = "compact",
        block_internal_channels=MODERATE_INTERNAL_CHANNELS,
        resize_mode: str = MODERATE_RESIZE_MODE,
    ) -> None:
        super().__init__(
            latent_channels=latent_channels,
            condition_channels=condition_channels,
            channels=channels,
            blocks_per_stage=blocks_per_stage,
            temporal_factor=temporal_factor,
            spatial_factor=spatial_factor,
            patch_size=patch_size,
            frames_to_trim=frames_to_trim,
            block_mode=block_mode,
            block_internal_channels=block_internal_channels,
            resize_mode=resize_mode,
        )

    @property
    def config_dict(self) -> dict[str, object]:
        config = dict(super().config_dict)
        config.update(
            {
                "class_name": type(self).__name__,
                "moderate_scheme": MODERATE_SCHEME,
                "moderate_target": "sub100_gmac_v1",
            }
        )
        return config

    def initialize_from_resizeconv_decoder(
        self,
        source: ResizeConvTinyConditionalDecoder,
    ) -> dict[str, object]:
        """Widen a ResizeConv compact decoder while preserving its initial function."""
        if type(source) is not ResizeConvTinyConditionalDecoder:
            raise TypeError("source must be an exact ResizeConvTinyConditionalDecoder")
        if source.block_mode != "compact" or self.block_mode != "compact":
            raise ValueError("Moderate widening requires compact source and target")
        for attribute in (
            "latent_channels",
            "channels",
            "blocks_per_stage",
            "temporal_factor",
            "spatial_factor",
            "patch_size",
            "frames_to_trim",
            "resize_mode",
        ):
            if getattr(self, attribute) != getattr(source, attribute):
                raise ValueError(
                    f"Moderate v1 requires unchanged {attribute}: "
                    f"source={getattr(source, attribute)!r} target={getattr(self, attribute)!r}"
                )

        condition_report = _widen_condition_path(source, self)

        source_modules = dict(source.named_modules())
        target_modules = dict(self.named_modules())
        widened_blocks: list[dict[str, object]] = []
        widened_block_prefixes: set[str] = set()
        for name, src_module in source_modules.items():
            if not isinstance(src_module, CompactMemBlock):
                continue
            dst_module = target_modules.get(name)
            if not isinstance(dst_module, CompactMemBlock):
                raise TypeError(f"Target module {name!r} is not CompactMemBlock")
            report = _widen_compact_block(src_module, dst_module)
            widened_blocks.append({"module": name, **report})
            widened_block_prefixes.add(name + ".")

        # Copy every unchanged tensor exactly.  Widened condition/input and compact
        # block tensors were handled above because their shapes differ.
        source_state = source.state_dict()
        target_state = self.state_dict()
        handled_prefixes = {
            "condition_projection.",
            "decoder.1.",
            *widened_block_prefixes,
        }
        copied = 0
        copied_elements = 0
        with torch.no_grad():
            for name, src_value in source_state.items():
                if any(name.startswith(prefix) for prefix in handled_prefixes):
                    continue
                dst_value = target_state.get(name)
                if dst_value is None:
                    raise KeyError(f"Target is missing source tensor {name!r}")
                if tuple(dst_value.shape) != tuple(src_value.shape):
                    raise ValueError(
                        f"Unexpected non-widened shape mismatch for {name}: "
                        f"source={tuple(src_value.shape)} target={tuple(dst_value.shape)}"
                    )
                dst_value.copy_(src_value.to(device=dst_value.device, dtype=dst_value.dtype))
                copied += 1
                copied_elements += int(src_value.numel())

        source_parameters = sum(p.numel() for p in source.parameters())
        target_parameters = sum(p.numel() for p in self.parameters())
        return {
            "scheme": MODERATE_SCHEME,
            "source_function_preserved_at_initialization": True,
            "numerical_note": "Outputs should match the source up to floating-point roundoff.",
            "condition": condition_report,
            "widened_blocks": widened_blocks,
            "source_decoder_parameters": source_parameters,
            "target_decoder_parameters": target_parameters,
            "parameter_delta": target_parameters - source_parameters,
            "copied_unchanged_tensors": copied,
            "copied_unchanged_elements": copied_elements,
            "target_channels": list(self.channels),
            "target_internal_channels": list(self.block_internal_channels),
            "target_condition_channels": int(self.condition_channels),
        }

    @classmethod
    def from_resizeconv_pretrained(
        cls,
        root: str | Path,
        *,
        condition_channels: int = MODERATE_CONDITION_CHANNELS,
        block_internal_channels=MODERATE_INTERNAL_CHANNELS,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> tuple["ModerateResizeConvTinyConditionalDecoder", dict[str, object]]:
        source = ResizeConvTinyConditionalDecoder.from_pretrained(
            root, device=device, dtype=dtype
        )
        model = cls(
            latent_channels=source.latent_channels,
            condition_channels=condition_channels,
            channels=source.channels,
            blocks_per_stage=source.blocks_per_stage,
            temporal_factor=source.temporal_factor,
            spatial_factor=source.spatial_factor,
            patch_size=source.patch_size,
            frames_to_trim=source.frames_to_trim,
            block_mode="compact",
            block_internal_channels=block_internal_channels,
            resize_mode=source.resize_mode,
        ).to(device=device, dtype=dtype)
        report = model.initialize_from_resizeconv_decoder(source)
        return model, report
