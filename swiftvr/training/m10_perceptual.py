"""R2 perceptual supervision and small diagnostics; no inference changes."""
from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def resolve_alexnet_weights(path: str | Path | None = None) -> Path:
    """Use existing A1-era torchvision weights. Never call a download helper."""
    if path is not None:
        result = Path(path).expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(f"Local AlexNet weights not found: {result}")
        return result
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    preferred = cache / "alexnet-owt-7be5be79.pth"
    if preferred.is_file():
        return preferred.resolve()
    matches = sorted(cache.glob("alexnet*.pth"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one cached AlexNet under {cache}; found {len(matches)}. "
            "Pass --alexnet-weights /existing/alexnet.pth (or set ALEXNET_WEIGHTS). "
            "No download was attempted."
        )
    return matches[0].resolve()


def _load_alexnet_trunk(trunk: nn.Module, state: dict) -> None:
    # Reference LPIPS preserves torchvision feature indices in slice1..slice5.
    mapped = {}
    for key, value in trunk.state_dict().items():
        group, suffix = key.split(".", 1)
        source = "features." + suffix
        if not group.startswith("slice") or source not in state:
            raise ValueError(f"Unsupported AlexNet/LPIPS state layout: {key} / {source}")
        if state[source].shape != value.shape:
            raise ValueError(f"AlexNet weight shape mismatch for {source}")
        mapped[key] = state[source]
    if not mapped:
        raise ValueError("LPIPS AlexNet trunk has no weights")
    trunk.load_state_dict(mapped, strict=True)


def load_local_lpips(alexnet_weights: str | Path | None = None):
    """Reference LPIPS-Alex v0.1, fully initialized from local files.

    Random construction prevents torchvision's pretrained loader from accessing
    the network. Both calibrated heads and ALL trunk tensors are then loaded.
    The resulting loss is not a random-feature substitute for LPIPS.
    """
    path = resolve_alexnet_weights(alexnet_weights)
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("R2 requires the same lpips package used by A1 training") from exc
    calibration = Path(inspect.getfile(lpips.LPIPS)).resolve().parent / "weights/v0.1/alex.pth"
    if not calibration.is_file():
        raise FileNotFoundError(f"Missing local LPIPS calibration: {calibration}")
    metric = lpips.LPIPS(net="alex", version="0.1", pnet_rand=True,
                         pretrained=False, verbose=False)
    heads = torch.load(calibration, map_location="cpu", weights_only=True)
    for i in range(5):
        if f"lin{i}.model.1.weight" not in heads:
            raise ValueError("LPIPS v0.1 Alex calibration is incomplete")
    incompatible = metric.load_state_dict(heads, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected LPIPS head weights: {incompatible.unexpected_keys}")
    _load_alexnet_trunk(metric.net, torch.load(path, map_location="cpu", weights_only=True))
    metric.pnet_rand = False  # Fully loaded above, not random features.
    metric.requires_grad_(False).eval()
    return metric, {"alexnet_weights": str(path), "lpips_calibration": str(calibration),
                    "definition": "reference LPIPS Alex v0.1; local pretrained trunk and calibrated heads"}


def lpips_frame_values(metric: nn.Module, prediction: torch.Tensor, target: torch.Tensor,
                       *, microbatch_frames: int = 4) -> torch.Tensor:
    """Return [B,T] values, preserving A1's clamp/[-1,1]/all-frame semantics.

    FP32 perceptual forward plus per-microbatch activation recomputation bounds
    activation memory without omitting temporal phases from supervision.
    """
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[2] != 3:
        raise ValueError("LPIPS expects matching [B,T,3,H,W] videos")
    if microbatch_frames <= 0 or prediction.shape[0] * prediction.shape[1] == 0:
        raise ValueError("LPIPS microbatch and frame count must be positive")
    b, t = prediction.shape[:2]
    pred = prediction.float().clamp(0, 1).flatten(0, 1) * 2 - 1
    ref = target.detach().float().clamp(0, 1).flatten(0, 1) * 2 - 1

    def evaluate(x, y):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return metric(x, y).reshape(x.shape[0], -1).mean(1)

    values = []
    for start in range(0, b * t, microbatch_frames):
        x, y = pred[start:start + microbatch_frames], ref[start:start + microbatch_frames]
        if torch.is_grad_enabled() and x.requires_grad:
            values.append(checkpoint(evaluate, x, y, use_reentrant=False))
        else:
            values.append(evaluate(x, y))
    return torch.cat(values).reshape(b, t)


def phase_totals(prediction: torch.Tensor, target: torch.Tensor,
                 perceptual_values: torch.Tensor) -> dict[str, float]:
    """Per-phase spatial/perceptual totals for val; exclude t0, never use GT.

    Report counts explicitly. These are teacher discrepancies, not absolute
    phase quality or evidence of resolving teacher's shared flicker.
    """
    b, t = prediction.shape[:2]
    if perceptual_values.shape != (b, t):
        raise ValueError("Expected one LPIPS value per frame")
    error = (prediction.detach().float().clamp(0, 1) - target.detach().float().clamp(0, 1))
    mae = error.abs().mean(dim=(2, 3, 4))
    indices = torch.arange(t, device=prediction.device)
    result = {}
    for phase in range(4):
        selected = (indices > 0) & ((indices + 3) % 4 == phase)
        result[f"phase{phase}_frames"] = b * int(selected.sum())
        result[f"phase{phase}_rgb_l1"] = float(mae[:, selected].sum())
        result[f"phase{phase}_lpips"] = float(perceptual_values.detach()[:, selected].sum())
    return result


def loss_gradient_probe(terms: dict, velocity: torch.Tensor) -> dict:
    """One-batch dLoss/dVelocity check, NOT a full parameter-gradient survey.

    autograd.grad stops at velocity and leaves parameter .grad untouched. The
    usual backward must still run afterward. Router gradients are not included.
    """
    losses = {
        "representation": terms["weighted_velocity_nmse"] + terms["weighted_velocity_cosine_loss"],
        "rgb": terms["weighted_rgb_l1"],
        "perceptual": terms["weighted_lpips"],
    }
    gradients = {}
    for name, loss in losses.items():
        grad, = torch.autograd.grad(loss, velocity, retain_graph=True)
        grad = grad.detach().float()
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError(f"Non-finite R2 {name} gradient at velocity")
        gradients[name] = grad
    anchor = gradients["representation"].flatten()
    appearance = (gradients["rgb"] + gradients["perceptual"]).flatten()
    an, pn = anchor.norm(), appearance.norm()
    valid = float(an) > 0 and float(pn) > 0
    return {
        "scope": "first rank0 train microbatch; weighted dLoss/dVelocity; excludes router",
        **{f"{k}_gradient_norm": float(v.norm()) for k, v in gradients.items()},
        "appearance_gradient_norm": float(pn),
        "anchor_to_appearance_norm_ratio": float(an / pn) if float(pn) > 0 else None,
        "anchor_appearance_cosine": float(torch.dot(anchor, appearance) / (an * pn)) if valid else None,
        "weights_automatically_changed": False,
    }
