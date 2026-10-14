"""GT output-space post-training for the existing M8/A1 deployment topology.

This is not a new diffusion objective. The old Stage-3 RGB L1 / frame-difference
objective is retained as the control; the experimental difference is GT LPIPS
through the *frozen deployment A1* into M8. No teacher targets or GAN are used.
"""
from __future__ import annotations

from pathlib import Path
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class M8TaskForward(nn.Module):
    """Wrap the canonical MoE velocity forward; never detach restored latents."""

    def __init__(self, backbone, decoder, *, checkpoint_decoder=True):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.checkpoint_decoder = bool(checkpoint_decoder)
        backbone.reae.requires_grad_(False)
        decoder.requires_grad_(False)
        self.train()

    @property
    def transformer(self):
        return self.backbone.transformer

    def train(self, mode=True):
        super().train(mode)
        self.backbone.reae.eval()
        self.decoder.eval()
        return self

    def forward(self, batch):
        out = self.backbone(batch)
        target = out["target"]
        if not isinstance(target, torch.Tensor):
            raise ValueError("GT task training requires a real HR tensor")
        latent = (out["z_lq"] - out["velocity"]).permute(0, 2, 1, 3, 4).contiguous()

        def decode(value):
            return self.decoder(value, output_frames=target.shape[1], clamp=False)

        prediction = (checkpoint(decode, latent, use_reentrant=False)
                      if self.checkpoint_decoder and torch.is_grad_enabled() else decode(latent))
        if prediction.shape != target.shape:
            raise ValueError("A1 output/GT geometry mismatch")
        return {"prediction": prediction, "target": target,
                "router_balance_loss": out["router_balance_loss"]}


class LocalLPIPS(nn.Module):
    """Load a COMPLETE AlexNet-LPIPS state locally; never download/fall back.

    pnet_rand=True/pretrained=False constructs architecture only. All random
    tensors are then replaced using strict=True before this module can be used.
    A lpips/weights/v0.1/alex.pth file alone is NOT a complete network state.
    """

    def __init__(self, state_path, *, frame_batch=8, checkpoint_features=True):
        super().__init__()
        path = Path(state_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if frame_batch <= 0:
            raise ValueError("LPIPS frame batch must be positive")
        import lpips
        self.metric = lpips.LPIPS(net="alex", pnet_rand=True, pretrained=False, verbose=False)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.metric.load_state_dict(state, strict=True)
        self.metric.requires_grad_(False).eval()
        self.frame_batch = int(frame_batch)
        self.checkpoint_features = bool(checkpoint_features)

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, prediction, target):
        # Deliberately no output clamp: reconstruction must also correct out-of-range pixels.
        pred = prediction.float().flatten(0, 1).mul(2).sub(1)
        ref = target.detach().float().flatten(0, 1).mul(2).sub(1)
        total = pred.new_zeros(())
        for start in range(0, len(pred), self.frame_batch):
            a, b = pred[start:start+self.frame_batch], ref[start:start+self.frame_batch]
            def evaluate(x, y):
                # Metric network stays FP32, independent of DiT autocast.
                with torch.autocast(device_type=x.device.type, enabled=False):
                    return self.metric(x.float(), y.float()).mean()
            value = (checkpoint(evaluate, a, b, use_reentrant=False)
                     if self.checkpoint_features and a.requires_grad else evaluate(a, b))
            total = total + len(a) * value
        return total / len(pred)


def task_objective(out, *, perceptual=None, pixel_weight=1.0,
                   temporal_weight=1.0, lpips_weight=0.1, router_weight=0.01):
    weights = (pixel_weight, temporal_weight, lpips_weight, router_weight)
    if any(not math.isfinite(v) or v < 0 for v in weights) or sum(weights[:3]) <= 0:
        raise ValueError("Need finite nonnegative weights and a nonzero GT objective")
    pred, gt = out["prediction"].float(), out["target"].detach().float()
    if pred.ndim != 5 or pred.shape != gt.shape or pred.shape[2] != 3:
        raise ValueError("Expected paired RGB videos [B,T,3,H,W]")
    pixel = (pred - gt).abs().mean()
    # Same first-order target as historical stage3.temporal_difference_mse.
    error = pred - gt
    temporal = (error[:, 1:] - error[:, :-1]).square().mean() if pred.shape[1] > 1 else pixel * 0
    lp = pixel.new_zeros(())
    if lpips_weight:
        if perceptual is None:
            raise ValueError("Positive LPIPS weight requires locally loaded perceptual network")
        lp = perceptual(pred, gt)
    router = out["router_balance_loss"].float()
    gt_loss = pixel_weight * pixel + temporal_weight * temporal + lpips_weight * lp
    return {"loss": gt_loss + router_weight * router, "gt_loss": gt_loss,
            "pixel_l1": pixel, "temporal_mse": temporal, "lpips": lp, "router": router,
            "out_of_range_fraction": ((pred.detach() < 0) | (pred.detach() > 1)).float().mean()}


def phase_errors(pred, target, trim=3):
    """Diagnostic only. Each RGB frame keeps its own corresponding GT."""
    mse = (pred.float() - target.float()).square().mean(dim=(0, 2, 3, 4))
    phase = (torch.arange(len(mse), device=mse.device) + trim) % 4
    return {f"phase_{p}_mse": mse[phase == p].mean() for p in range(4) if (phase == p).any()}
