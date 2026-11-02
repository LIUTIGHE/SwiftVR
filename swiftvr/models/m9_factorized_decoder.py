"""M9-A1 factorized ReAE decoder at the M8 Decoder76 compute budget.

The first two decoder stages and all upsampling transitions keep the validated
M8 Decoder76 widths (128,96,64,64).  Only stage2 is refactored:

* three 3-conv spatial residual blocks initialized from the corresponding
  teacher MemBlocks with current/past kernels collapsed into a spatial kernel;
* one extra 2-conv spatial residual block, zero-initialized as identity;
* one explicit causal temporal-only adapter implemented by (3,1,1) convolutions;
* one high-resolution depthwise 3x3 detail residual after conv23.

At 1920x1088 this is analytically 75.95040768 GMAC/output-frame, within 0.66%
of the 76.45175808 GMAC/frame M8 Decoder76 baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .reae import Clamp, MemBlock, ReAE, TGrow, conv
from .reae_slim_decoder import (
    FORMAT_VERSION,
    TEACHER_CHANNELS,
    _copy_conv2d_subset,
    _copy_memblock_subset,
    _copy_tgrow_subset,
    _validate_indices,
)

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
M9_A1_CHANNELS = (128, 96, 64, 64)
M9_A1_GMAC_1920X1088 = 75.95040768


class SpatialResBlock3(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = conv(channels, channels)
        self.conv2 = conv(channels, channels)
        self.conv3 = conv(channels, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = self.conv3(y)
        return self.act(x + y)


class SpatialResBlock2(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = conv(channels, channels)
        self.conv2 = conv(channels, channels)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.act(self.conv1(x))))


class CausalTemporalAdapter(nn.Module):
    """Two-layer temporal-only residual adapter over [B,T,C,H,W]."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=(3, 1, 1), padding=0, bias=True
        )
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size=(3, 1, 1), padding=0, bias=True
        )
        self.act = nn.ReLU(inplace=True)
        # Identity start: the explicit temporal branch is learned without
        # perturbing the mapped spatial initialization at step 0.
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    @staticmethod
    def _causal(conv: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
        return conv(F.pad(x, (0, 0, 0, 0, 2, 0)))

    def forward_clip(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"temporal adapter expects [B,T,C,H,W], got {video.shape}")
        x = video.permute(0, 2, 1, 3, 4).contiguous()
        y = self.act(self._causal(self.conv1, x))
        y = self._causal(self.conv2, y)
        out = self.act(x + y)
        return out.permute(0, 2, 1, 3, 4).contiguous()


class HighResDepthwiseDetail(nn.Module):
    """Cheap high-resolution per-channel 3x3 detail residual.

    Zero initialization preserves the baseline conv23 -> ReLU behavior initially.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=True
        )
        nn.init.zeros_(self.dw.weight)
        if self.dw.bias is not None:
            nn.init.zeros_(self.dw.bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.dw(x))


def _copy_conv(dst: nn.Conv2d, src: nn.Conv2d) -> None:
    if tuple(dst.weight.shape) != tuple(src.weight.shape):
        raise ValueError(f"conv shape mismatch {dst.weight.shape} vs {src.weight.shape}")
    with torch.no_grad():
        dst.weight.copy_(src.weight.to(device=dst.weight.device, dtype=dst.weight.dtype))
        if dst.bias is not None:
            if src.bias is None:
                dst.bias.zero_()
            else:
                dst.bias.copy_(src.bias.to(device=dst.bias.device, dtype=dst.bias.dtype))


def _init_spatial3_from_memblock(
    dst: SpatialResBlock3,
    src: MemBlock,
    indices: Sequence[int],
    teacher_width: int,
) -> None:
    index = torch.tensor(indices, device=src.conv[0].weight.device, dtype=torch.long)
    weight0 = src.conv[0].weight.detach().index_select(0, index)
    current = weight0.index_select(1, index)
    past_index = index + int(teacher_width)
    past = weight0.index_select(1, past_index)
    collapsed = current + past
    with torch.no_grad():
        dst.conv1.weight.copy_(collapsed.to(dst.conv1.weight))
        if dst.conv1.bias is not None:
            if src.conv[0].bias is None:
                dst.conv1.bias.zero_()
            else:
                dst.conv1.bias.copy_(src.conv[0].bias.detach().index_select(0, index).to(dst.conv1.bias))

    _copy_conv2d_subset(dst.conv2, src.conv[2], out_indices=indices, in_indices=indices)
    _copy_conv2d_subset(dst.conv3, src.conv[4], out_indices=indices, in_indices=indices)


def _apply_stack(stack: nn.Sequential, video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"decoder input must be [B,T,C,H,W], got {tuple(video.shape)}")
    batch, frames, channels, height, width = video.shape
    hidden = video.reshape(batch * frames, channels, height, width)

    for index, layer in enumerate(stack):
        if isinstance(layer, MemBlock):
            _, channels, height, width = hidden.shape
            current = hidden.reshape(batch, frames, channels, height, width)
            past = torch.cat([torch.zeros_like(current[:, :1]), current[:, :-1]], dim=1)
            hidden = layer(hidden, past.reshape(batch * frames, channels, height, width))
        elif isinstance(layer, CausalTemporalAdapter):
            _, channels, height, width = hidden.shape
            current = hidden.reshape(batch, frames, channels, height, width)
            current = layer.forward_clip(current)
            hidden = current.reshape(batch * frames, channels, height, width)
        elif isinstance(layer, TGrow):
            hidden = layer(hidden)
        else:
            hidden = layer(hidden)

        if hidden.shape[0] % batch:
            raise RuntimeError(
                f"decoder layer {index} produced leading dimension {hidden.shape[0]} for batch={batch}"
            )
        frames = int(hidden.shape[0] // batch)

    _, channels, height, width = hidden.shape
    return hidden.reshape(batch, frames, channels, height, width)


class M9A1FactorizedReAEDecoder(nn.Module):
    def __init__(
        self,
        *,
        channels: Sequence[int] = M9_A1_CHANNELS,
        latent_channels: int = 48,
        patch_size: int = 2,
        frames_to_trim: int = 3,
    ) -> None:
        super().__init__()
        self.channels = tuple(int(v) for v in channels)
        if self.channels != M9_A1_CHANNELS:
            raise ValueError(f"M9-A1 is fixed to channels={M9_A1_CHANNELS}, got {self.channels}")
        self.latent_channels = int(latent_channels)
        self.patch_size = int(patch_size)
        self.frames_to_trim = int(frames_to_trim)
        self.image_channels = 3
        c0, c1, c2, c3 = self.channels
        self.decoder = nn.Sequential(
            Clamp(),                         # 0
            conv(self.latent_channels, c0), # 1
            nn.ReLU(inplace=True),          # 2
            MemBlock(c0, c0),               # 3
            MemBlock(c0, c0),               # 4
            MemBlock(c0, c0),               # 5
            nn.Upsample(scale_factor=2),     # 6
            TGrow(c0, 1),                    # 7
            conv(c0, c1, bias=False),        # 8
            MemBlock(c1, c1),               # 9
            MemBlock(c1, c1),               # 10
            MemBlock(c1, c1),               # 11
            nn.Upsample(scale_factor=2),     # 12
            TGrow(c1, 2),                    # 13
            conv(c1, c2, bias=False),        # 14
            SpatialResBlock3(c2),            # 15
            SpatialResBlock3(c2),            # 16
            SpatialResBlock3(c2),            # 17
            SpatialResBlock2(c2),            # 18
            CausalTemporalAdapter(c2),       # 19
            nn.Upsample(scale_factor=2),     # 20
            TGrow(c2, 2),                    # 21
            conv(c2, c3, bias=False),        # 22
            HighResDepthwiseDetail(c3),      # 23
            conv(c3, self.image_channels * self.patch_size**2), # 24
        )
        self.pruning_metadata: dict[str, object] = {}

    @property
    def config_dict(self) -> dict[str, object]:
        return {
            "format_version": FORMAT_VERSION,
            "class_name": type(self).__name__,
            "architecture": "m9_a1_spatial_temporal_factorized_v1",
            "channels": list(self.channels),
            "latent_channels": self.latent_channels,
            "patch_size": self.patch_size,
            "frames_to_trim": self.frames_to_trim,
            "estimated_gmac_per_output_frame_1920x1088": M9_A1_GMAC_1920X1088,
            "pruning_metadata": dict(self.pruning_metadata),
        }

    def forward(
        self,
        latents: torch.Tensor,
        *,
        output_frames: int | None = None,
        clamp: bool = False,
    ) -> torch.Tensor:
        if latents.ndim != 5 or int(latents.shape[2]) != self.latent_channels:
            raise ValueError(
                f"latents must be [B,F,{self.latent_channels},H,W], got {tuple(latents.shape)}"
            )
        pixels = _apply_stack(self.decoder, latents)
        batch, frames, channels, height, width = pixels.shape
        flat = F.pixel_shuffle(
            pixels.reshape(batch * frames, channels, height, width), self.patch_size
        )
        pixels = flat.reshape(batch, frames, *flat.shape[1:])
        if self.frames_to_trim:
            pixels = pixels[:, self.frames_to_trim:]
        if output_frames is not None:
            output_frames = int(output_frames)
            if pixels.shape[1] < output_frames:
                raise RuntimeError(
                    f"M9-A1 emitted {pixels.shape[1]} frames; requested {output_frames}"
                )
            pixels = pixels[:, :output_frames]
        return pixels.clamp(0.0, 1.0) if clamp else pixels

    def initialize_from_reae(
        self,
        teacher: ReAE,
        stage_indices: Sequence[Sequence[int]],
        *,
        score_method: str = "activation_rms",
    ) -> dict[str, object]:
        indices = _validate_indices(stage_indices, self.channels)
        d = self.decoder
        t = teacher.decoder
        i0, i1, i2, i3 = indices

        _copy_conv2d_subset(d[1], t[1], out_indices=i0, in_indices=None)
        for layer in (3, 4, 5):
            _copy_memblock_subset(d[layer], t[layer], i0, TEACHER_CHANNELS[0])
        _copy_tgrow_subset(d[7], t[7], i0)
        _copy_conv2d_subset(d[8], t[8], out_indices=i1, in_indices=i0)

        for layer in (9, 10, 11):
            _copy_memblock_subset(d[layer], t[layer], i1, TEACHER_CHANNELS[1])
        _copy_tgrow_subset(d[13], t[13], i1)
        _copy_conv2d_subset(d[14], t[14], out_indices=i2, in_indices=i1)

        for dst_idx, src_idx in zip((15, 16, 17), (15, 16, 17)):
            _init_spatial3_from_memblock(d[dst_idx], t[src_idx], i2, TEACHER_CHANNELS[2])

        _copy_tgrow_subset(d[21], t[19], i2)
        _copy_conv2d_subset(d[22], t[20], out_indices=i3, in_indices=i2)
        _copy_conv2d_subset(d[24], t[22], out_indices=None, in_indices=i3)

        self.pruning_metadata = {
            "scheme": "m9_a1_factorized_stage2_from_reae_v1",
            "score_method": str(score_method),
            "teacher_channels": list(TEACHER_CHANNELS),
            "student_channels": list(self.channels),
            "stage_indices": [list(values) for values in indices],
            "stage2_initialization": "teacher_memblock_current_plus_past_spatial_collapse",
            "extra_spatial_block_init": "identity_zero_last_conv",
            "temporal_adapter_init": "identity_zero_last_conv",
            "highres_detail_init": "identity_zero_depthwise",
        }
        return dict(self.pruning_metadata)

    def save_pretrained(self, output_dir: str | Path) -> Path:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / CONFIG_FILENAME).write_text(
            json.dumps(self.config_dict, indent=2, sort_keys=True), encoding="utf-8"
        )
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
            str(root / WEIGHTS_FILENAME),
        )
        return root

    @classmethod
    def from_pretrained(
        cls,
        root: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "M9A1FactorizedReAEDecoder":
        root = Path(root).expanduser().resolve()
        config = json.loads((root / CONFIG_FILENAME).read_text(encoding="utf-8"))
        model = cls(
            channels=config["channels"],
            latent_channels=int(config.get("latent_channels", 48)),
            patch_size=int(config.get("patch_size", 2)),
            frames_to_trim=int(config.get("frames_to_trim", 3)),
        )
        model.pruning_metadata = dict(config.get("pruning_metadata", {}))
        model.load_state_dict(load_file(str(root / WEIGHTS_FILENAME), device="cpu"), strict=True)
        model.to(device=device, dtype=dtype)
        return model


def m9_a1_compute_breakdown_1920x1088() -> dict[str, float]:
    return {
        "input_conv": 0.11280384,
        "stage0_memblocks": 3.60972288,
        "transition01": 1.03612416,
        "stage1_memblocks": 8.12187648,
        "transition12": 5.41458432,
        "stage2_spatial3_x3": 21.65833728,
        "stage2_spatial2_x1": 4.81296384,
        "stage2_temporal_adapter": 1.60432128,
        "transition23": 25.66914048,
        "highres_depthwise_detail": 0.30081024,
        "output_head": 3.60972288,
    }
