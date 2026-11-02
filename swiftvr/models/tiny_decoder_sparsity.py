"""Structured channel sparsity for the SwiftVR Tiny Conditional Decoder.

The deployment goal is real dense-kernel reduction rather than unstructured zeros.
A dense MemBlock keeps its stage/interface width C while a learnable channel gate
scores the C internal channels.  After a short sparsity calibration, top-k channels
are materialized into a compact bottleneck block

    2C -> K -> K -> C,

so block count, residual width, causal past fusion, TGrow topology, and stage
interfaces stay unchanged while the expensive 3x3 convolution MACs shrink.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Iterable, Sequence

import torch
import torch.nn as nn

from .reae import MemBlock

if TYPE_CHECKING:
    from .tiny_conditional_decoder import TinyConditionalDecoder


def _conv(in_channels: int, out_channels: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


class StructuredSparseMemBlock(MemBlock):
    """Dense-shape MemBlock with one learnable structured channel gate.

    The same gate is applied after the first and second hidden activations.  With
    all gate values equal to one this is functionally identical to a same-width
    ReAE MemBlock, which makes an existing dense Tiny Decoder an exact
    initialization for sparsity calibration.
    """

    def __init__(self, channels: int) -> None:
        nn.Module.__init__(self)
        channels = int(channels)
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.interface_channels = channels
        self.internal_channels = channels
        self.conv = nn.Sequential(
            _conv(channels * 2, channels),
            nn.ReLU(inplace=False),
            _conv(channels, channels),
            nn.ReLU(inplace=False),
            _conv(channels, channels),
        )
        self.skip = nn.Identity()
        self.act = nn.ReLU(inplace=False)
        self.channel_gate = nn.Parameter(torch.ones(channels, dtype=torch.float32))

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        hidden = self.conv[0](torch.cat([x, past], dim=1))
        hidden = self.conv[1](hidden)
        gate = self.channel_gate.to(dtype=hidden.dtype).view(1, -1, 1, 1)
        hidden = hidden * gate
        hidden = self.conv[2](hidden)
        hidden = self.conv[3](hidden)
        hidden = hidden * gate
        hidden = self.conv[4](hidden)
        return self.act(hidden + x)


class CompactMemBlock(MemBlock):
    """Materialized structured-sparse MemBlock with a real narrow hidden width."""

    def __init__(self, channels: int, internal_channels: int) -> None:
        nn.Module.__init__(self)
        channels = int(channels)
        internal_channels = int(internal_channels)
        if channels <= 0 or internal_channels <= 0:
            raise ValueError("channels/internal_channels must be positive")
        if internal_channels > channels:
            raise ValueError(
                f"internal_channels={internal_channels} exceeds interface channels={channels}"
            )
        self.interface_channels = channels
        self.internal_channels = internal_channels
        self.conv = nn.Sequential(
            _conv(channels * 2, internal_channels),
            nn.ReLU(inplace=False),
            _conv(internal_channels, internal_channels),
            nn.ReLU(inplace=False),
            _conv(internal_channels, channels),
        )
        self.skip = nn.Identity()
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], dim=1)) + x)


def structured_sparsity_penalty(module: nn.Module) -> torch.Tensor:
    """Mean L1 magnitude of all structured gates, normalized across channels."""

    gates = [
        block.channel_gate
        for block in module.modules()
        if isinstance(block, StructuredSparseMemBlock)
    ]
    if not gates:
        raise ValueError("module contains no StructuredSparseMemBlock gates")
    return torch.cat([gate.reshape(-1) for gate in gates]).abs().mean()


def structured_gate_summary(module: nn.Module) -> list[dict[str, float | int]]:
    summary: list[dict[str, float | int]] = []
    block_index = 0
    for block in module.modules():
        if not isinstance(block, StructuredSparseMemBlock):
            continue
        values = block.channel_gate.detach().float().abs().cpu()
        summary.append(
            {
                "block": block_index,
                "channels": int(values.numel()),
                "min": float(values.min().item()),
                "mean": float(values.mean().item()),
                "max": float(values.max().item()),
                "std": float(values.std(unbiased=False).item()),
            }
        )
        block_index += 1
    return summary


def _rounded_internal_width(channels: int, keep_ratio: float, multiple: int) -> int:
    channels = int(channels)
    keep_ratio = float(keep_ratio)
    multiple = int(multiple)
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0,1], got {keep_ratio}")
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    raw = channels * keep_ratio
    rounded = int(round(raw / multiple)) * multiple
    rounded = max(min(multiple, channels), rounded)
    return min(channels, rounded)


def stage_internal_widths(
    channels: Sequence[int],
    *,
    keep_ratio: float,
    multiple: int = 8,
) -> tuple[int, ...]:
    return tuple(
        _rounded_internal_width(int(width), float(keep_ratio), int(multiple))
        for width in channels
    )


def convert_dense_decoder_to_sparse(
    dense: "TinyConditionalDecoder",
) -> "TinyConditionalDecoder":
    """Convert a dense Tiny Decoder to an exactly initialized sparse supernet."""

    from .tiny_conditional_decoder import TinyConditionalDecoder

    if dense.block_mode != "dense":
        raise ValueError(f"expected block_mode='dense', got {dense.block_mode!r}")
    sparse = TinyConditionalDecoder(
        latent_channels=dense.latent_channels,
        condition_channels=dense.condition_channels,
        channels=dense.channels,
        blocks_per_stage=dense.blocks_per_stage,
        temporal_factor=dense.temporal_factor,
        spatial_factor=dense.spatial_factor,
        patch_size=dense.patch_size,
        frames_to_trim=dense.frames_to_trim,
        block_mode="sparse",
    )
    incompatible = sparse.load_state_dict(dense.state_dict(), strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    expected_missing = sorted(
        name for name, _ in sparse.named_parameters() if name.endswith("channel_gate")
    )
    if missing != expected_missing or unexpected:
        raise RuntimeError(
            "Dense-to-sparse checkpoint mapping mismatch: "
            f"missing={missing}, expected_gate_missing={expected_missing}, "
            f"unexpected={unexpected}"
        )
    return sparse


def _copy_nonblock_state(source: nn.Module, target: nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    if set(source_state) != set(target_state):
        raise RuntimeError(
            f"Non-block state mismatch: source={sorted(source_state)}, "
            f"target={sorted(target_state)}"
        )
    target.load_state_dict(source_state, strict=True)


def _materialize_block(
    source: StructuredSparseMemBlock,
    target: CompactMemBlock,
) -> tuple[int, ...]:
    k = int(target.internal_channels)
    scores = source.channel_gate.detach().float().abs()
    if k > scores.numel():
        raise ValueError(f"requested k={k} from {scores.numel()} sparse channels")
    selected = torch.topk(scores, k=k, largest=True, sorted=True).indices
    selected = selected.sort().values
    selected_device = selected.to(device=source.conv[0].weight.device)
    gate = source.channel_gate.detach()[selected_device].to(
        device=source.conv[0].weight.device,
        dtype=source.conv[0].weight.dtype,
    )

    with torch.no_grad():
        src0: nn.Conv2d = source.conv[0]
        src1: nn.Conv2d = source.conv[2]
        src2: nn.Conv2d = source.conv[4]
        dst0: nn.Conv2d = target.conv[0]
        dst1: nn.Conv2d = target.conv[2]
        dst2: nn.Conv2d = target.conv[4]

        dst0.weight.copy_(src0.weight[selected_device])
        if dst0.bias is not None and src0.bias is not None:
            dst0.bias.copy_(src0.bias[selected_device])

        # First gate scales the input channels of conv2.
        sliced1 = src1.weight[selected_device][:, selected_device]
        sliced1 = sliced1 * gate.view(1, -1, 1, 1)
        dst1.weight.copy_(sliced1)
        if dst1.bias is not None and src1.bias is not None:
            dst1.bias.copy_(src1.bias[selected_device])

        # Second gate scales the input channels of conv3.
        sliced2 = src2.weight[:, selected_device]
        sliced2 = sliced2 * gate.view(1, -1, 1, 1)
        dst2.weight.copy_(sliced2)
        if dst2.bias is not None and src2.bias is not None:
            dst2.bias.copy_(src2.bias)

    return tuple(int(index) for index in selected.tolist())


def materialize_sparse_decoder(
    sparse: "TinyConditionalDecoder",
    *,
    keep_ratio: float,
    multiple: int = 8,
) -> tuple["TinyConditionalDecoder", list[dict[str, object]]]:
    """Export fixed top-k structured channels as an ordinary compact dense model."""

    from .tiny_conditional_decoder import TinyConditionalDecoder

    if sparse.block_mode != "sparse":
        raise ValueError(f"expected block_mode='sparse', got {sparse.block_mode!r}")
    internal = stage_internal_widths(
        sparse.channels,
        keep_ratio=keep_ratio,
        multiple=multiple,
    )
    compact = TinyConditionalDecoder(
        latent_channels=sparse.latent_channels,
        condition_channels=sparse.condition_channels,
        channels=sparse.channels,
        blocks_per_stage=sparse.blocks_per_stage,
        temporal_factor=sparse.temporal_factor,
        spatial_factor=sparse.spatial_factor,
        patch_size=sparse.patch_size,
        frames_to_trim=sparse.frames_to_trim,
        block_mode="compact",
        block_internal_channels=internal,
    )

    # Condition projection is outside decoder Sequential.
    _copy_nonblock_state(sparse.condition_projection, compact.condition_projection)

    manifest: list[dict[str, object]] = []
    stage = 0
    blocks_seen_in_stage = 0
    for layer_index, (source_layer, target_layer) in enumerate(
        zip(sparse.decoder, compact.decoder)
    ):
        if isinstance(source_layer, StructuredSparseMemBlock):
            if not isinstance(target_layer, CompactMemBlock):
                raise RuntimeError(f"layer {layer_index} compact block type mismatch")
            selected = _materialize_block(source_layer, target_layer)
            manifest.append(
                {
                    "layer_index": int(layer_index),
                    "stage": int(stage),
                    "interface_channels": int(source_layer.interface_channels),
                    "internal_channels": int(target_layer.internal_channels),
                    "selected_channels": list(selected),
                }
            )
            blocks_seen_in_stage += 1
            if blocks_seen_in_stage == sparse.blocks_per_stage[stage]:
                blocks_seen_in_stage = 0
                stage = min(stage + 1, len(sparse.channels) - 1)
            continue
        if isinstance(target_layer, CompactMemBlock):
            raise RuntimeError(f"layer {layer_index} source block type mismatch")
        _copy_nonblock_state(source_layer, target_layer)

    expected_blocks = sum(sparse.blocks_per_stage)
    if len(manifest) != expected_blocks:
        raise RuntimeError(
            f"materialized {len(manifest)} blocks, expected {expected_blocks}"
        )
    return compact, manifest
