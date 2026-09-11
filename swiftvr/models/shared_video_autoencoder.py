"""Reusable deterministic video latent codec built from SwiftVR ReAE components.

This module packages the original Restoration-aware Autoencoder (ReAE) encoder
with a compatible ReAE-family decoder such as the Stage-B1 Slim100 decoder.
It is intentionally called an *autoencoder*, not a VAE: the encoder emits a
single deterministic 48-channel latent and there is no mean/log-variance head,
reparameterization, or KL prior.

The default whole-clip contract follows SwiftVR training clips:

    video [B,T,3,H,W], T = 4k+1, H/W divisible by 16
        -> repeat the final RGB frame 3 times
        -> encoder temporal compression x4
        -> latent [B,(T+3)/4,48,H/16,W/16]
        -> decoder temporal expansion x4
        -> trim the first 3 causal warm-up frames
        -> reconstruction [B,T,3,H,W]

The codec can be assembled directly from the existing ReAE + SlimReAE
checkpoints or exported once into a single ``config.json`` +
``model.safetensors`` folder for reuse in other low-level video tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .reae import MemBlock, ReAE, TPool
from .reae_slim_decoder import SlimReAEDecoder, TEACHER_CHANNELS


CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
FORMAT_VERSION = 1


def _apply_encoder_stack(stack: nn.Sequential, video: torch.Tensor) -> torch.Tensor:
    """Whole-clip ReAE encoder execution with causal MemBlock semantics."""
    if video.ndim != 5:
        raise ValueError(f"encoder input must be [B,T,C,H,W], got {tuple(video.shape)}")
    batch, frames, channels, height, width = video.shape
    hidden = video.reshape(batch * frames, channels, height, width)

    for index, layer in enumerate(stack):
        if isinstance(layer, MemBlock):
            _, channels, height, width = hidden.shape
            current = hidden.reshape(batch, frames, channels, height, width)
            past = torch.cat([torch.zeros_like(current[:, :1]), current[:, :-1]], dim=1)
            hidden = layer(hidden, past.reshape(batch * frames, channels, height, width))
        elif isinstance(layer, TPool):
            if frames % int(layer.stride):
                raise RuntimeError(
                    f"encoder TPool layer {index} stride={layer.stride} cannot consume T={frames}"
                )
            hidden = layer(hidden)
            frames //= int(layer.stride)
        else:
            hidden = layer(hidden)
        if hidden.shape[0] != batch * frames:
            raise RuntimeError(
                f"encoder layer {index} produced leading dimension {hidden.shape[0]} "
                f"for batch={batch}, frames={frames}"
            )

    _, channels, height, width = hidden.shape
    return hidden.reshape(batch, frames, channels, height, width)


class SharedVideoAutoencoder(nn.Module):
    """Deterministic causal video autoencoder for a shared latent space.

    ``encoder`` is the original ReAE encoder. ``decoder`` is a
    :class:`SlimReAEDecoder`, normally the Stage-B1 Slim100 model. All methods
    preserve autograd. Call :meth:`freeze` when the codec should stay fixed;
    gradients can still flow through the frozen decoder into an upstream
    task-specific Transformer/UNet output latent.
    """

    def __init__(self, encoder: nn.Sequential, decoder: SlimReAEDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.image_channels = 3
        self.patch_size = int(decoder.patch_size)
        self.latent_channels = int(decoder.latent_channels)
        self.frames_to_trim = int(decoder.frames_to_trim)

    @property
    def decoder_channels(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.decoder.channels)

    @property
    def spatial_compression(self) -> int:
        return 16

    @property
    def temporal_compression(self) -> int:
        return 4

    @property
    def config_dict(self) -> dict[str, object]:
        return {
            "format_version": FORMAT_VERSION,
            "class_name": type(self).__name__,
            "codec_type": "deterministic_causal_video_autoencoder",
            "is_variational": False,
            "image_channels": self.image_channels,
            "latent_channels": self.latent_channels,
            "patch_size": self.patch_size,
            "frames_to_trim": self.frames_to_trim,
            "spatial_compression": self.spatial_compression,
            "temporal_compression": self.temporal_compression,
            "decoder_channels": list(self.decoder_channels),
            "decoder_pruning_metadata": dict(self.decoder.pruning_metadata),
            "whole_clip_contract": "T=4k+1; repeat-last x3 before encode; trim-first x3 after decode",
            "latent_scaling": None,
        }

    def _validate_video(self, video: torch.Tensor) -> tuple[int, int, int, int, int]:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,T,3,H,W], got {tuple(video.shape)}")
        batch, frames, channels, height, width = (int(v) for v in video.shape)
        if channels != self.image_channels:
            raise ValueError(f"expected {self.image_channels} RGB channels, got {channels}")
        if frames <= 0 or frames % 4 != 1:
            raise ValueError(
                f"whole-clip codec expects T=4k+1 frames, got T={frames}; "
                "use 5/9/13/17/... frame clips or the existing StreamingTAE API"
            )
        if height % self.spatial_compression or width % self.spatial_compression:
            raise ValueError(
                f"H/W must be divisible by {self.spatial_compression}, got {height}x{width}"
            )
        return batch, frames, channels, height, width

    def _validate_latent(self, latents: torch.Tensor) -> None:
        if latents.ndim != 5:
            raise ValueError(f"latents must be [B,F,C,H,W], got {tuple(latents.shape)}")
        if int(latents.shape[2]) != self.latent_channels:
            raise ValueError(
                f"expected latent C={self.latent_channels}, got C={int(latents.shape[2])}"
            )

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a ``4k+1`` RGB clip into the deterministic ReAE latent.

        Input range should match SwiftVR training/inference, normally ``[0,1]``.
        No latent normalization or stochastic sampling is applied.
        """
        _, frames, _, _, _ = self._validate_video(video)
        padded = torch.cat([video, video[:, -1:].expand(-1, 3, -1, -1, -1)], dim=1)

        if self.patch_size > 1:
            batch, padded_frames, channels, height, width = padded.shape
            shuffled = F.pixel_unshuffle(
                padded.reshape(batch * padded_frames, channels, height, width),
                self.patch_size,
            )
            padded = shuffled.reshape(batch, padded_frames, *shuffled.shape[1:])

        latents = _apply_encoder_stack(self.encoder, padded)
        expected = (frames + 3) // self.temporal_compression
        if int(latents.shape[1]) != expected:
            raise RuntimeError(
                f"encoder emitted F={latents.shape[1]} latent frames; expected {expected} for T={frames}"
            )
        self._validate_latent(latents)
        return latents

    def decode(
        self,
        latents: torch.Tensor,
        *,
        output_frames: int | None = None,
        clamp: bool = True,
    ) -> torch.Tensor:
        """Decode a latent clip back to RGB.

        ``output_frames`` is useful after a task-specific latent network. For a
        latent obtained from :meth:`encode`, pass the original RGB frame count.
        """
        self._validate_latent(latents)
        return self.decoder(latents, output_frames=output_frames, clamp=clamp)

    def forward(self, video: torch.Tensor, *, clamp: bool = True) -> torch.Tensor:
        """Standalone autoencoder reconstruction ``decode(encode(video))``."""
        _, frames, _, _, _ = self._validate_video(video)
        latents = self.encode(video)
        return self.decode(latents, output_frames=frames, clamp=clamp)

    def freeze(self) -> "SharedVideoAutoencoder":
        """Freeze codec weights without disabling gradient flow through inputs."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    @staticmethod
    def _move_model(
        model: "SharedVideoAutoencoder",
        device: str | torch.device,
        dtype: torch.dtype | None,
    ) -> "SharedVideoAutoencoder":
        if dtype is None:
            model.to(device=device)
        else:
            model.to(device=device, dtype=dtype)
        return model

    @classmethod
    def from_component_checkpoints(
        cls,
        reae_checkpoint: str | Path,
        slim_decoder_checkpoint: str | Path | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "SharedVideoAutoencoder":
        """Assemble the codec from current SwiftVR component checkpoints.

        Args:
            reae_checkpoint: path to ``reae.safetensors``.
            slim_decoder_checkpoint: Stage-B1 ``.../tiny_decoder`` directory.
                If omitted, the original full ReAE decoder is wrapped instead.
        """
        reae_path = Path(reae_checkpoint).expanduser().resolve()
        if not reae_path.is_file():
            raise FileNotFoundError(reae_path)
        base = ReAE(str(reae_path))

        if slim_decoder_checkpoint is not None:
            decoder = SlimReAEDecoder.from_pretrained(
                slim_decoder_checkpoint, device="cpu", dtype=torch.float32
            )
        else:
            decoder = SlimReAEDecoder(
                channels=TEACHER_CHANNELS,
                latent_channels=base.latent_channels,
                patch_size=base.patch_size,
                frames_to_trim=base.frames_to_trim,
            )
            decoder.decoder.load_state_dict(base.decoder.state_dict(), strict=True)
            decoder.pruning_metadata = {
                "scheme": "original_full_reae_decoder",
                "teacher_channels": list(TEACHER_CHANNELS),
                "student_channels": list(TEACHER_CHANNELS),
            }

        if int(decoder.latent_channels) != int(base.latent_channels):
            raise ValueError("encoder/decoder latent-channel mismatch")
        if int(decoder.patch_size) != int(base.patch_size):
            raise ValueError("encoder/decoder patch-size mismatch")
        if int(decoder.frames_to_trim) != int(base.frames_to_trim):
            raise ValueError("encoder/decoder temporal-trim mismatch")

        return cls._move_model(cls(base.encoder, decoder), device, dtype)

    def save_pretrained(self, output_dir: str | Path) -> Path:
        """Export encoder + decoder into one portable codec checkpoint folder."""
        root = Path(output_dir).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty directory: {root}")
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
    ) -> "SharedVideoAutoencoder":
        """Load a single-folder codec exported by :meth:`save_pretrained`."""
        root = Path(root).expanduser().resolve()
        config = json.loads((root / CONFIG_FILENAME).read_text(encoding="utf-8"))
        if int(config.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(f"unsupported codec format: {config.get('format_version')}")
        if bool(config.get("is_variational", True)):
            raise ValueError("SharedVideoAutoencoder expects a deterministic codec config")

        latent_channels = int(config["latent_channels"])
        patch_size = int(config["patch_size"])
        frames_to_trim = int(config["frames_to_trim"])
        template = ReAE(
            checkpoint_path=None,
            patch_size=patch_size,
            latent_channels=latent_channels,
        )
        decoder = SlimReAEDecoder(
            channels=tuple(int(v) for v in config["decoder_channels"]),
            latent_channels=latent_channels,
            patch_size=patch_size,
            frames_to_trim=frames_to_trim,
        )
        decoder.pruning_metadata = dict(config.get("decoder_pruning_metadata", {}))
        model = cls(template.encoder, decoder)
        model.load_state_dict(load_file(str(root / WEIGHTS_FILENAME), device="cpu"), strict=True)
        return cls._move_model(model, device, dtype)
