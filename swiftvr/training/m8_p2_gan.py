"""Training-only video PatchGAN for GT post-training of SwiftVR M8/A1.

Input uses 13 temporally aligned RGB frames, randomly aligned *spatial* crops.
The discriminator sees both RGB and actual adjacent-frame differences.
It is never instantiated during deployment inference.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class TemporalPatchDiscriminator(nn.Module):
    """Spectrally normalized 3D patch critic, no batch-dependent normalization."""

    def __init__(self, width: int = 32):
        super().__init__()
        if width < 8:
            raise ValueError("Discriminator width must be at least 8")
        channels = [6, width, width * 2, width * 3, width * 4]
        layers = []
        for idx in range(4):
            layers.extend([
                spectral_norm(nn.Conv3d(channels[idx], channels[idx + 1],
                                        kernel_size=(3, 3, 3),
                                        stride=(1, 2, 2), padding=1)),
                nn.LeakyReLU(0.2, inplace=False),
            ])
        layers.append(spectral_norm(nn.Conv3d(channels[-1], 1, 3, padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[2] != 3 or rgb.shape[1] < 3:
            raise ValueError("Expected RGB [B,T,3,H,W], T>=3")
        if min(rgb.shape[-2:]) < 32:
            raise ValueError("Video patches must be at least 32x32")
        # Both real and fake see the same bounds as deployment RGB.
        image = rgb.float().clamp(0, 1).mul(2).sub(1)
        delta = torch.cat([torch.zeros_like(image[:, :1]),
                           image[:, 1:] - image[:, :-1]], dim=1)
        features = torch.cat([image, delta], dim=2)
        return self.net(features.permute(0, 2, 1, 3, 4).contiguous())


def paired_spatial_crop(fake: torch.Tensor, real: torch.Tensor, crop: int):
    """Pairwise-identical crops, preserving every temporal frame and phase.

    Per-sample random offsets are taken from the current per-rank torch RNG.
    No temporal shifts, frame drops, or independent fake/real augmentation.
    """
    if fake.shape != real.shape or fake.ndim != 5 or fake.shape[2] != 3:
        raise ValueError("Expected aligned fake/real RGB [B,T,3,H,W]")
    if crop < 32 or crop > min(fake.shape[-2:]):
        raise ValueError("Invalid crop size")
    height, width = fake.shape[-2:]
    if crop == height == width:
        return fake, real
    pieces_f, pieces_r = [], []
    for b in range(fake.shape[0]):
        top = int(torch.randint(height - crop + 1, (), device=fake.device))
        left = int(torch.randint(width - crop + 1, (), device=fake.device))
        pieces_f.append(fake[b:b+1, :, :, top:top+crop, left:left+crop])
        pieces_r.append(real[b:b+1, :, :, top:top+crop, left:left+crop])
    return torch.cat(pieces_f, dim=0), torch.cat(pieces_r, dim=0)


def hinge_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    return F.relu(1 - real_logits.float()).mean() + F.relu(1 + fake_logits.float()).mean()


def generator_adversarial_loss(fake_logits: torch.Tensor):
    return -fake_logits.float().mean()


def adversarial_weight(step: int, *, weight: float, ramp_steps: int):
    """Ramp only the G adversarial term; D always learns from real/fake."""
    if step < 1 or weight <= 0 or ramp_steps < 0 or not math.isfinite(weight):
        raise ValueError("Invalid GAN loss schedule")
    if ramp_steps == 0:
        return float(weight)
    return float(weight) * min(float(step) / ramp_steps, 1.0)


def critic_monitor(real_logits, fake_logits):
    return {
        "critic_real": real_logits.detach().float().mean(),
        "critic_fake": fake_logits.detach().float().mean(),
        "critic_margin": (real_logits.detach().float().mean() -
                          fake_logits.detach().float().mean()),
    }
