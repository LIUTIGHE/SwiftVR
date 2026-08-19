"""FlashVSR-style conditional decoder family adapted to the SwiftVR latent contract.

Reference implementation:
  OpenImagingLab/FlashVSR
  examples/WanVSR/utils/TCDecoder.py
  commit 6dd38e57203af4efca97df82c659f5d5a2dcf51a

The official FlashVSR TC decoder directly concatenates a packed RGB condition
with the diffusion latent, uses TAEHV-style causal MemBlocks, channel-to-time
1x1 TGrow layers, nearest spatial upsampling, and widths [512,256,128,128].

SwiftVR differs in two important contracts:
  * latent channels: 48 rather than 16;
  * spatial compression: 16x rather than 8x.

Therefore packed RGB is 3*4*16*16=3072 channels and an extra final nearest x2
spatial upsample is required.  No learned condition bottleneck is used in this
family.  The decoder outputs RGB directly; there is no PixelShuffle output head.

Three presets are provided:
  repo_faithful   : repository-style widths/depth plus IdentityConv deepening;
  core_faithful   : same TC core without the repository deepening layers;
  moderate_100g   : direct condition + full-width MemBlocks, compressed to an
                    estimated <100 GMAC/output-frame at padded 1080p.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .reae import Clamp, MemBlock, TGrow
from .tiny_conditional_decoder import _apply_video_stack, _validate_video


CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
FORMAT_VERSION = 1
SUPPORTED_FORMAT_VERSIONS = frozenset({1})
REFERENCE_COMMIT = "6dd38e57203af4efca97df82c659f5d5a2dcf51a"
REFERENCE_PATH = "examples/WanVSR/utils/TCDecoder.py"

PRESETS: dict[str, dict[str, object]] = {
    "repo_faithful": {
        "channels": (512, 256, 128, 128),
        "blocks_per_stage": (3, 3, 3, 0),
        "deepen_input": True,
        "deepen_output": True,
    },
    "core_faithful": {
        "channels": (512, 256, 128, 128),
        "blocks_per_stage": (3, 3, 3, 0),
        "deepen_input": False,
        "deepen_output": False,
    },
    "moderate_100g": {
        "channels": (256, 128, 64, 64),
        "blocks_per_stage": (2, 2, 2, 0),
        "deepen_input": False,
        "deepen_output": False,
    },
}


def _conv(in_channels: int, out_channels: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


class IdentityConv2d(nn.Conv2d):
    """FlashVSR repository-style identity-initialized same-width 3x3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels, channels, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            nn.init.dirac_(self.weight)


class FlashVSRTGrow(TGrow):
    """Exact channel-to-time 1x1 TGrow pattern used by FlashVSR TCDecoder.

    Subclassing SwiftVR's TGrow keeps the existing whole-clip and streaming
    executors compatible because they dispatch through ``isinstance(TGrow)``.
    """

    def __init__(self, channels: int, stride: int) -> None:
        nn.Module.__init__(self)
        self.stride = int(stride)
        self.n_f = int(channels)
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        self.conv = nn.Conv2d(
            self.n_f,
            self.n_f * self.stride,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, channels, height, width = x.shape
        if int(channels) != self.n_f:
            raise ValueError(f"Expected C={self.n_f}, got C={channels}")
        x = self.conv(x)
        return x.reshape(nt * self.stride, self.n_f, height, width)


def pack_rgb_condition_flashvsr(
    condition: torch.Tensor,
    *,
    temporal_factor: int = 4,
    spatial_factor: int = 16,
) -> torch.Tensor:
    """Pack RGB in the same channel order as FlashVSR PixelShuffle3d.

    FlashVSR rearranges
      b c (f ff) (h hh) (w ww) -> b f (c ff hh ww) h w.

    Prefix padding repeats frame 0, matching the official implementation and
    SwiftVR's causal warm-up alignment.
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
    groups = padded_frames // temporal_factor
    out_h = height // spatial_factor
    out_w = width // spatial_factor

    packed = condition.reshape(
        batch,
        groups,
        temporal_factor,
        channels,
        out_h,
        spatial_factor,
        out_w,
        spatial_factor,
    )
    packed = packed.permute(0, 1, 3, 2, 5, 7, 4, 6).contiguous()
    return packed.reshape(
        batch,
        groups,
        channels * temporal_factor * spatial_factor * spatial_factor,
        out_h,
        out_w,
    )


def preset_config(name: str) -> dict[str, object]:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; expected one of {sorted(PRESETS)}")
    return dict(PRESETS[name])


class FlashVSRConditionalDecoder(nn.Module):
    """Direct-condition TAEHV-style decoder adapted to SwiftVR's 4x16x latent."""

    def __init__(
        self,
        *,
        channels: Sequence[int] = (512, 256, 128, 128),
        blocks_per_stage: Sequence[int] = (3, 3, 3, 0),
        deepen_input: bool = False,
        deepen_output: bool = False,
        latent_channels: int = 48,
        temporal_factor: int = 4,
        spatial_factor: int = 16,
        frames_to_trim: int = 3,
        preset_name: str | None = None,
    ) -> None:
        super().__init__()
        self.channels = tuple(int(v) for v in channels)
        self.blocks_per_stage = tuple(int(v) for v in blocks_per_stage)
        self.deepen_input = bool(deepen_input)
        self.deepen_output = bool(deepen_output)
        self.latent_channels = int(latent_channels)
        self.temporal_factor = int(temporal_factor)
        self.spatial_factor = int(spatial_factor)
        self.frames_to_trim = int(frames_to_trim)
        self.preset_name = preset_name
        self.condition_injection = "flashvsr_direct_packed_rgb_concat"
        self.output_head = "nearest_resize_conv"

        if len(self.channels) != 4 or len(self.blocks_per_stage) != 4:
            raise ValueError("channels and blocks_per_stage must each have four entries")
        if any(v <= 0 for v in self.channels):
            raise ValueError("all channel widths must be positive")
        if any(v < 0 for v in self.blocks_per_stage):
            raise ValueError("block counts must be non-negative")
        if self.latent_channels != 48:
            raise ValueError("SwiftVR-adapted TC decoder currently requires latent_channels=48")
        if self.temporal_factor != 4:
            raise ValueError("SwiftVR temporal factor is fixed to 4")
        if self.spatial_factor != 16:
            raise ValueError("SwiftVR spatial factor is fixed to 16")
        if self.frames_to_trim < 0:
            raise ValueError("frames_to_trim must be non-negative")

        self.packed_condition_channels = 3 * self.temporal_factor * self.spatial_factor**2
        c0, c1, c2, c3 = self.channels

        layers: list[nn.Module] = [
            Clamp(),
            _conv(self.latent_channels + self.packed_condition_channels, c0),
            nn.ReLU(inplace=False),
        ]
        if self.deepen_input:
            layers.extend([IdentityConv2d(c0), nn.ReLU(inplace=False)])

        layers.extend(MemBlock(c0, c0) for _ in range(self.blocks_per_stage[0]))
        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                FlashVSRTGrow(c0, 1),
                _conv(c0, c1, bias=False),
            ]
        )
        layers.extend(MemBlock(c1, c1) for _ in range(self.blocks_per_stage[1]))
        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                FlashVSRTGrow(c1, 2),
                _conv(c1, c2, bias=False),
            ]
        )
        layers.extend(MemBlock(c2, c2) for _ in range(self.blocks_per_stage[2]))
        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                FlashVSRTGrow(c2, 2),
                _conv(c2, c3, bias=False),
            ]
        )
        layers.extend(MemBlock(c3, c3) for _ in range(self.blocks_per_stage[3]))

        # FlashVSR starts from an 8x spatial latent and reaches full resolution
        # after the third nearest upsample. SwiftVR starts at 16x, so one extra
        # spatial-only nearest x2 is inserted before the final ReLU/RGB head.
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.append(nn.ReLU(inplace=False))
        if self.deepen_output:
            layers.extend([IdentityConv2d(c3), nn.ReLU(inplace=False)])
        layers.append(_conv(c3, 3))
        self.decoder = nn.Sequential(*layers)

    @classmethod
    def from_preset(cls, name: str) -> "FlashVSRConditionalDecoder":
        cfg = preset_config(name)
        return cls(**cfg, preset_name=name)

    @property
    def config_dict(self) -> dict[str, object]:
        return {
            "format_version": FORMAT_VERSION,
            "class_name": type(self).__name__,
            "preset_name": self.preset_name,
            "channels": list(self.channels),
            "blocks_per_stage": list(self.blocks_per_stage),
            "deepen_input": self.deepen_input,
            "deepen_output": self.deepen_output,
            "latent_channels": self.latent_channels,
            "temporal_factor": self.temporal_factor,
            "spatial_factor": self.spatial_factor,
            "frames_to_trim": self.frames_to_trim,
            "packed_condition_channels": self.packed_condition_channels,
            "condition_injection": self.condition_injection,
            "condition_projection": None,
            "condition_value_contract": "SwiftVR pixels in [0,1]",
            "output_head": self.output_head,
            "reference_repository": "OpenImagingLab/FlashVSR",
            "reference_commit": REFERENCE_COMMIT,
            "reference_path": REFERENCE_PATH,
            "tgrow": "flashvsr_channel_to_time_1x1",
        }

    def pack_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return pack_rgb_condition_flashvsr(
            condition,
            temporal_factor=self.temporal_factor,
            spatial_factor=self.spatial_factor,
        )

    def project_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """Compatibility name: direct TC conditioning has no learned projection."""
        return self.pack_condition(condition)

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

        packed = self.pack_condition(condition)
        if tuple(packed.shape[:2]) != tuple(latents.shape[:2]) or tuple(
            packed.shape[-2:]
        ) != tuple(latents.shape[-2:]):
            raise ValueError(
                "Packed condition does not match latent grid: "
                f"condition={tuple(packed.shape)}, latent={tuple(latents.shape)}"
            )

        hidden = torch.cat([packed, latents], dim=2)
        pixels = _apply_video_stack(self.decoder, hidden)

        if self.frames_to_trim:
            pixels = pixels[:, self.frames_to_trim :]
        if output_frames is not None:
            output_frames = int(output_frames)
            if output_frames <= 0:
                raise ValueError("output_frames must be positive")
            if int(pixels.shape[1]) < output_frames:
                raise RuntimeError(
                    f"TC decoder emitted {pixels.shape[1]} valid frames; "
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
    ) -> "FlashVSRConditionalDecoder":
        root = Path(root).expanduser().resolve()
        config = json.loads((root / CONFIG_FILENAME).read_text(encoding="utf-8"))
        if int(config.get("format_version", -1)) not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(f"Unsupported decoder format {config.get('format_version')!r}")
        if config.get("class_name") != cls.__name__:
            raise ValueError(
                f"Expected class_name={cls.__name__!r}, got {config.get('class_name')!r}"
            )
        model = cls(
            channels=config["channels"],
            blocks_per_stage=config["blocks_per_stage"],
            deepen_input=bool(config.get("deepen_input", False)),
            deepen_output=bool(config.get("deepen_output", False)),
            latent_channels=int(config.get("latent_channels", 48)),
            temporal_factor=int(config.get("temporal_factor", 4)),
            spatial_factor=int(config.get("spatial_factor", 16)),
            frames_to_trim=int(config.get("frames_to_trim", 3)),
            preset_name=config.get("preset_name"),
        )
        weights = load_file(str(root / WEIGHTS_FILENAME), device="cpu")
        model.load_state_dict(weights, strict=True)
        model.to(device=device, dtype=dtype)
        return model


def count_parameters(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def estimate_steady_state_macs(
    model_or_config: FlashVSRConditionalDecoder | Mapping[str, object],
    *,
    output_width: int = 1920,
    output_height: int = 1088,
) -> dict[str, object]:
    """Exact analytical Conv MAC count for steady-state output frames.

    The convention matches the existing SwiftVR runtime profiler: one MAC is one
    multiply-accumulate.  At 1080p deployment SwiftVR pads height to 1088, hence
    the default 1920x1088 geometry.
    """

    if isinstance(model_or_config, FlashVSRConditionalDecoder):
        channels = model_or_config.channels
        blocks = model_or_config.blocks_per_stage
        deepen_input = model_or_config.deepen_input
        deepen_output = model_or_config.deepen_output
        packed_channels = model_or_config.packed_condition_channels
        latent_channels = model_or_config.latent_channels
    else:
        channels = tuple(int(v) for v in model_or_config["channels"])
        blocks = tuple(int(v) for v in model_or_config["blocks_per_stage"])
        deepen_input = bool(model_or_config.get("deepen_input", False))
        deepen_output = bool(model_or_config.get("deepen_output", False))
        packed_channels = int(model_or_config.get("packed_condition_channels", 3072))
        latent_channels = int(model_or_config.get("latent_channels", 48))

    if output_width % 16 or output_height % 16:
        raise ValueError("output geometry must be divisible by 16")
    c0, c1, c2, c3 = channels
    b0, b1, b2, b3 = blocks
    full = int(output_width * output_height)
    a2, a4, a8, a16 = full // 4, full // 16, full // 64, full // 256

    def conv(area: int, rate: float, cin: int, cout: int, kernel: int = 3) -> float:
        return float(area) * rate * cin * cout * kernel * kernel

    def mem(area: int, rate: float, width: int, count: int) -> float:
        return float(area) * rate * count * 36 * width * width

    values: dict[str, float] = {
        "input_fusion": conv(a16, 0.25, latent_channels + packed_channels, c0),
        "stage0_memblocks": mem(a16, 0.25, c0, b0),
        "tgrow01": float(a8) * 0.25 * c0 * c0,
        "transition01": conv(a8, 0.25, c0, c1),
        "stage1_memblocks": mem(a8, 0.25, c1, b1),
        # FlashVSR TGrow stride2 is C->2C at the input temporal rate.
        "tgrow12": float(a4) * 0.25 * c1 * (2 * c1),
        "transition12": conv(a4, 0.50, c1, c2),
        "stage2_memblocks": mem(a4, 0.50, c2, b2),
        "tgrow23": float(a2) * 0.50 * c2 * (2 * c2),
        "transition23": conv(a2, 1.00, c2, c3),
        "stage3_memblocks": mem(a2, 1.00, c3, b3),
        "rgb_head": conv(full, 1.00, c3, 3),
    }
    if deepen_input:
        values["input_identity_deepen"] = conv(a16, 0.25, c0, c0)
    if deepen_output:
        # Extra SwiftVR spatial x2 is placed before final ReLU/head so the
        # repository-style final deepening operates at full RGB resolution.
        values["output_identity_deepen"] = conv(full, 1.00, c3, c3)

    total = float(sum(values.values()))
    return {
        "output_width": int(output_width),
        "output_height": int(output_height),
        "gmacs_per_output_frame": total / 1e9,
        "gflops_per_output_frame_if_1mac_2flops": 2.0 * total / 1e9,
        "by_component_gmacs_per_output_frame": {
            name: value / 1e9 for name, value in values.items()
        },
        "mac_convention": "1 MAC = one multiply-accumulate",
    }
