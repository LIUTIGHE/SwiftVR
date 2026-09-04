"""Structurally slimmed ReAE decoder for behavior-preserving Stage-B compression.

This module deliberately keeps the original ReAE decoder topology intact:
three causal MemBlocks at each of the first three spatial stages, the same
nearest-neighbour spatial upsampling, the same TGrow temporal expansion, and the
same PixelShuffle RGB output contract.  Compression changes only the four stage
interface widths.

A slim decoder is initialized by selecting a consistent subset of channels from
an already trained full ReAE decoder.  The same stage subset is used for every
MemBlock, TGrow, transition convolution and residual interface in that stage, so
the student is a literal structured subnetwork of the teacher rather than a new
architecture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .reae import Clamp, MemBlock, ReAE, TGrow, conv


CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
FORMAT_VERSION = 1
TEACHER_CHANNELS = (512, 256, 128, 64)
SLIM100_CHANNELS = (256, 128, 64, 64)
AGGRESSIVE_CHANNELS = (256, 128, 64, 32)
VARIANT_CHANNELS = {
    "slim100": SLIM100_CHANNELS,
    "aggressive": AGGRESSIVE_CHANNELS,
}
STAGE_SCORE_LAYER_INDICES = {
    0: (3, 4, 5),
    1: (9, 10, 11),
    2: (15, 16, 17),
    3: (21,),
}


def _validate_indices(indices: Sequence[Sequence[int]], widths: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    if len(indices) != 4 or len(widths) != 4:
        raise ValueError("ReAE slimming requires four stages")
    normalized: list[tuple[int, ...]] = []
    for stage, (values, width, teacher_width) in enumerate(zip(indices, widths, TEACHER_CHANNELS)):
        chosen = tuple(int(value) for value in values)
        if len(chosen) != int(width):
            raise ValueError(
                f"stage{stage} expected {width} selected channels, got {len(chosen)}"
            )
        if len(set(chosen)) != len(chosen):
            raise ValueError(f"stage{stage} channel indices contain duplicates")
        if any(value < 0 or value >= teacher_width for value in chosen):
            raise ValueError(
                f"stage{stage} channel indices must lie in [0,{teacher_width})"
            )
        if tuple(sorted(chosen)) != chosen:
            raise ValueError("selected stage channels must be sorted in teacher order")
        normalized.append(chosen)
    return tuple(normalized)


def topk_stage_indices(
    stage_scores: Sequence[torch.Tensor],
    widths: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Select top-k activation-score channels and restore original channel order."""
    if len(stage_scores) != 4 or len(widths) != 4:
        raise ValueError("stage_scores/widths must contain four stages")
    result: list[tuple[int, ...]] = []
    for stage, (scores, width, teacher_width) in enumerate(
        zip(stage_scores, widths, TEACHER_CHANNELS)
    ):
        scores = scores.detach().float().flatten().cpu()
        if int(scores.numel()) != teacher_width:
            raise ValueError(
                f"stage{stage} score width {scores.numel()} != teacher width {teacher_width}"
            )
        width = int(width)
        if width <= 0 or width > teacher_width:
            raise ValueError(f"invalid stage{stage} target width {width}")
        chosen = torch.topk(scores, k=width, largest=True, sorted=False).indices.sort().values
        result.append(tuple(int(value) for value in chosen.tolist()))
    return tuple(result)


def _apply_decoder_stack(stack: nn.Sequential, video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"decoder input must be [B,T,C,H,W], got {tuple(video.shape)}")
    batch, frames, channels, height, width = video.shape
    hidden = video.reshape(batch * frames, channels, height, width)
    for index, layer in enumerate(stack):
        if isinstance(layer, MemBlock):
            _, channels, height, width = hidden.shape
            current = hidden.reshape(batch, frames, channels, height, width)
            past = torch.cat(
                [torch.zeros_like(current[:, :1]), current[:, :-1]], dim=1
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
                f"decoder layer {index} produced leading dimension {hidden.shape[0]} "
                f"for batch={batch}"
            )
        frames = int(hidden.shape[0] // batch)
    _, channels, height, width = hidden.shape
    return hidden.reshape(batch, frames, channels, height, width)


def _copy_conv2d_subset(
    dst: nn.Conv2d,
    src: nn.Conv2d,
    *,
    out_indices: Sequence[int] | None,
    in_indices: Sequence[int] | None,
) -> None:
    out_index = (
        torch.arange(src.out_channels, device=src.weight.device)
        if out_indices is None
        else torch.tensor(out_indices, device=src.weight.device, dtype=torch.long)
    )
    in_index = (
        torch.arange(src.in_channels, device=src.weight.device)
        if in_indices is None
        else torch.tensor(in_indices, device=src.weight.device, dtype=torch.long)
    )
    weight = src.weight.detach().index_select(0, out_index).index_select(1, in_index)
    if tuple(weight.shape) != tuple(dst.weight.shape):
        raise ValueError(
            f"Conv2d subset shape {tuple(weight.shape)} != target {tuple(dst.weight.shape)}"
        )
    with torch.no_grad():
        dst.weight.copy_(weight.to(device=dst.weight.device, dtype=dst.weight.dtype))
        if dst.bias is not None:
            if src.bias is None:
                dst.bias.zero_()
            else:
                bias = src.bias.detach().index_select(0, out_index)
                dst.bias.copy_(bias.to(device=dst.bias.device, dtype=dst.bias.dtype))


def _copy_memblock_subset(
    dst: MemBlock,
    src: MemBlock,
    indices: Sequence[int],
    teacher_width: int,
) -> None:
    chosen = tuple(int(value) for value in indices)
    doubled = chosen + tuple(teacher_width + value for value in chosen)
    _copy_conv2d_subset(dst.conv[0], src.conv[0], out_indices=chosen, in_indices=doubled)
    _copy_conv2d_subset(dst.conv[2], src.conv[2], out_indices=chosen, in_indices=chosen)
    _copy_conv2d_subset(dst.conv[4], src.conv[4], out_indices=chosen, in_indices=chosen)
    if not isinstance(dst.skip, nn.Identity) or not isinstance(src.skip, nn.Identity):
        raise TypeError("same-width ReAE MemBlocks are expected to use identity skips")


def _copy_tgrow_subset(
    dst: TGrow,
    src: TGrow,
    indices: Sequence[int],
) -> None:
    if dst.stride != src.stride:
        raise ValueError("TGrow stride mismatch")
    index = torch.tensor(indices, device=next(src.parameters()).device, dtype=torch.long)
    with torch.no_grad():
        if src.stride == 1:
            assert src.proj is not None and dst.proj is not None
            weight = src.proj.weight.detach().index_select(0, index).index_select(1, index)
            dst.proj.weight.copy_(weight.to(device=dst.proj.weight.device, dtype=dst.proj.weight.dtype))
        else:
            assert src.conv3d is not None and dst.conv3d is not None
            weight = src.conv3d.weight.detach().index_select(0, index).index_select(1, index)
            dst.conv3d.weight.copy_(
                weight.to(device=dst.conv3d.weight.device, dtype=dst.conv3d.weight.dtype)
            )


class SlimReAEDecoder(nn.Module):
    """Decoder-only ReAE with configurable stage widths and unchanged topology."""

    def __init__(
        self,
        *,
        channels: Sequence[int],
        latent_channels: int = 48,
        patch_size: int = 2,
        frames_to_trim: int = 3,
    ) -> None:
        super().__init__()
        self.channels = tuple(int(value) for value in channels)
        if len(self.channels) != 4 or any(value <= 0 for value in self.channels):
            raise ValueError("channels must contain four positive widths")
        self.latent_channels = int(latent_channels)
        self.patch_size = int(patch_size)
        self.frames_to_trim = int(frames_to_trim)
        self.image_channels = 3
        c0, c1, c2, c3 = self.channels
        self.decoder = nn.Sequential(
            Clamp(),
            conv(self.latent_channels, c0),
            nn.ReLU(inplace=True),
            MemBlock(c0, c0),
            MemBlock(c0, c0),
            MemBlock(c0, c0),
            nn.Upsample(scale_factor=2),
            TGrow(c0, 1),
            conv(c0, c1, bias=False),
            MemBlock(c1, c1),
            MemBlock(c1, c1),
            MemBlock(c1, c1),
            nn.Upsample(scale_factor=2),
            TGrow(c1, 2),
            conv(c1, c2, bias=False),
            MemBlock(c2, c2),
            MemBlock(c2, c2),
            MemBlock(c2, c2),
            nn.Upsample(scale_factor=2),
            TGrow(c2, 2),
            conv(c2, c3, bias=False),
            nn.ReLU(inplace=True),
            conv(c3, self.image_channels * self.patch_size**2),
        )
        self.pruning_metadata: dict[str, object] = {}

    @property
    def config_dict(self) -> dict[str, object]:
        return {
            "format_version": FORMAT_VERSION,
            "class_name": type(self).__name__,
            "channels": list(self.channels),
            "latent_channels": self.latent_channels,
            "patch_size": self.patch_size,
            "frames_to_trim": self.frames_to_trim,
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
        pixels = _apply_decoder_stack(self.decoder, latents)
        batch, frames, channels, height, width = pixels.shape
        flat = F.pixel_shuffle(
            pixels.reshape(batch * frames, channels, height, width), self.patch_size
        )
        pixels = flat.reshape(batch, frames, *flat.shape[1:])
        if self.frames_to_trim:
            pixels = pixels[:, self.frames_to_trim :]
        if output_frames is not None:
            output_frames = int(output_frames)
            if pixels.shape[1] < output_frames:
                raise RuntimeError(
                    f"Slim ReAE emitted {pixels.shape[1]} frames; requested {output_frames}"
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
        if tuple(getattr(teacher, "decoder", ())[1].weight.shape[:1]) != (TEACHER_CHANNELS[0],):
            raise ValueError("teacher does not match the expected width_mult=2 ReAE decoder")
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

        for layer in (15, 16, 17):
            _copy_memblock_subset(d[layer], t[layer], i2, TEACHER_CHANNELS[2])
        _copy_tgrow_subset(d[19], t[19], i2)
        _copy_conv2d_subset(d[20], t[20], out_indices=i3, in_indices=i2)
        _copy_conv2d_subset(d[22], t[22], out_indices=None, in_indices=i3)

        self.pruning_metadata = {
            "scheme": "reae_structured_stage_channel_subset_v1",
            "score_method": str(score_method),
            "teacher_channels": list(TEACHER_CHANNELS),
            "student_channels": list(self.channels),
            "stage_indices": [list(values) for values in indices],
        }
        return dict(self.pruning_metadata)

    def save_pretrained(self, output_dir: str | Path) -> Path:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / CONFIG_FILENAME).write_text(
            json.dumps(self.config_dict, indent=2, sort_keys=True), encoding="utf-8"
        )
        state = {
            name: tensor.detach().cpu().contiguous()
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
    ) -> "SlimReAEDecoder":
        root = Path(root).expanduser().resolve()
        config = json.loads((root / CONFIG_FILENAME).read_text(encoding="utf-8"))
        if int(config.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(f"Unsupported SlimReAE format: {config.get('format_version')}")
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
