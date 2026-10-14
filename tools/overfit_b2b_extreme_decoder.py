#!/usr/bin/env python3
"""B2B-0B one-sample capacity gate for the 13.36-GMAC extreme ReAE decoder.

This gate deliberately answers only one question: given the *correct* frozen
Stage-A teacher latent, can the proposed [96,48,24,16] decoder fit the frozen
full-ReAE teacher RGB for one deterministic clip?

GT is reported for diagnosis but is not part of the default optimization target.
That keeps decoder capacity separate from the (real) teacher-vs-GT disagreement.
The Stage-A source is executed only once to produce z_SR and the teacher RGB, then
released before decoder optimization.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the already validated B1 deterministic sample / Stage-A source path.
from smoke_tiny_conditional_decoder import (  # noqa: E402
    _build_sample,
    _load_source,
    _metrics,
    _save_visuals,
    _source_targets,
)
from smoke_training_forward import resolve_runtime_dtype, validate_folded_checkpoint  # noqa: E402
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder  # noqa: E402


EXTREME_CHANNELS = (96, 48, 24, 16)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--source-checkpoint", type=Path, required=True,
                   help="Stage-A no-time/no-prompt delta checkpoint, e.g. the 200k teacher.")
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--split", default="train")
    p.add_argument("--clip-length", type=int, default=13)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--view-seed", type=int, default=20260805)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--allow-dtype-mismatch", action="store_true")
    p.add_argument("--attention-backend", choices=("sdpa",), default="sdpa")
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")

    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("B2B-0B requires CUDA for the frozen Stage-A source")
    if args.steps <= 0 or args.learning_rate <= 0 or args.log_every <= 0:
        raise ValueError("steps, learning-rate and log-every must be positive")
    if args.max_grad_norm <= 0 or args.weight_decay < 0:
        raise ValueError("max-grad-norm must be positive and weight-decay non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    folded = validate_folded_checkpoint(
        args.base_checkpoint,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
    )
    source_dtype = resolve_runtime_dtype(
        args.dtype,
        folded,
        device,
        allow_mismatch=args.allow_dtype_mismatch,
    )
    if source_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")

    print("[1/5] Loading deterministic one-sample view...", flush=True)
    batch = _build_sample(args)
    print("[2/5] Loading frozen Stage-A source and computing teacher latent/RGB once...", flush=True)
    source = _load_source(args, device=device, dtype=source_dtype)
    targets = _source_targets(source, batch, device=device, dtype=source_dtype)

    # Use a literal structured subnetwork initialization.  This is intentionally
    # simple for the capacity gate; B2B-0C can replace it with activation-based
    # selection once the architecture has demonstrated one-sample fitting ability.
    decoder = SlimReAEDecoder(channels=EXTREME_CHANNELS).to(device=device, dtype=torch.float32)
    prefix_indices = tuple(tuple(range(width)) for width in EXTREME_CHANNELS)
    init_report = decoder.initialize_from_reae(
        source.reae,
        prefix_indices,
        score_method="b2b0b_teacher_prefix",
    )

    # Stage-A is no longer needed.  Keep only fixed tensors in FP32 for a clean,
    # numerically boring decoder-capacity test.
    del source
    gc.collect()
    torch.cuda.empty_cache()

    z_sr = targets["z_sr"].detach().to(device=device, dtype=torch.float32)
    teacher_rgb = targets["reae_teacher"].detach().to(device=device, dtype=torch.float32)
    gt = targets["target"].detach().to(device=device, dtype=torch.float32)
    lq = targets["lq_input"].detach().to(device=device, dtype=torch.float32)
    output_frames = int(gt.shape[1])

    def predict() -> torch.Tensor:
        return decoder(z_sr, output_frames=output_frames, clamp=False)

    decoder.eval()
    with torch.no_grad():
        initial = predict()
    initial_metrics = _metrics(initial, gt, teacher_rgb)
    _save_visuals(
        output_dir,
        prefix="initial",
        lq_input=lq,
        target=gt,
        reae_teacher=teacher_rgb,
        tiny=initial,
    )
    print(
        "Initial: "
        f"teacher_psnr={initial_metrics['tiny_teacher_psnr']:.4f} "
        f"gt_psnr={initial_metrics['tiny_gt_psnr']:.4f}",
        flush=True,
    )
    del initial

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    print("[3/5] Overfitting extreme decoder to frozen teacher RGB (teacher-only MSE)...", flush=True)
    decoder.train()
    history: list[dict[str, float | int]] = []
    best_teacher_psnr = float("-inf")
    best_step = 0

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = predict()
        loss = F.mse_loss(prediction, teacher_rgb)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite B2B-0B loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            decoder.parameters(),
            max_norm=args.max_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                teacher_mse_clamped = F.mse_loss(
                    prediction.clamp(0, 1), teacher_rgb.clamp(0, 1)
                ).item()
                gt_mse_clamped = F.mse_loss(
                    prediction.clamp(0, 1), gt.clamp(0, 1)
                ).item()
            teacher_psnr = _psnr_from_mse(float(teacher_mse_clamped))
            gt_psnr = _psnr_from_mse(float(gt_mse_clamped))
            if teacher_psnr > best_teacher_psnr:
                best_teacher_psnr = teacher_psnr
                best_step = step
            record = {
                "step": step,
                "teacher_mse_raw": float(loss.detach().item()),
                "teacher_mse_clamped": float(teacher_mse_clamped),
                "teacher_psnr_clamped": float(teacher_psnr),
                "gt_psnr_clamped": float(gt_psnr),
                "grad_norm": float(grad_norm.detach().item()),
            }
            history.append(record)
            print(
                f"step={step:4d} teacher_mse={loss.item():.8f} "
                f"teacher_psnr={teacher_psnr:.3f}dB "
                f"gt_psnr={gt_psnr:.3f}dB grad={grad_norm.item():.4f}",
                flush=True,
            )
        del prediction

    print("[4/5] Final evaluation and checkpoint round-trip...", flush=True)
    decoder.eval()
    with torch.no_grad():
        final_prediction = predict()
    final_metrics = _metrics(final_prediction, gt, teacher_rgb)
    _save_visuals(
        output_dir,
        prefix="final",
        lq_input=lq,
        target=gt,
        reae_teacher=teacher_rgb,
        tiny=final_prediction,
    )

    checkpoint_dir = output_dir / "extreme_decoder"
    decoder.save_pretrained(checkpoint_dir)
    reloaded = SlimReAEDecoder.from_pretrained(
        checkpoint_dir,
        device=device,
        dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        reloaded_prediction = reloaded(z_sr, output_frames=output_frames, clamp=False)
    torch.testing.assert_close(final_prediction, reloaded_prediction, rtol=0, atol=0)

    final_teacher_psnr = float(final_metrics["tiny_teacher_psnr"])
    if final_teacher_psnr >= 35.0:
        interpretation = "STRONG_PASS"
    elif final_teacher_psnr >= 32.0:
        interpretation = "PASS"
    else:
        interpretation = "CAPACITY_OR_OPTIMIZATION_CONCERN"

    summary = {
        "status": "PASS",
        "gate": "B2B-0B",
        "interpretation": interpretation,
        "architecture": {
            "channels": list(EXTREME_CHANNELS),
            "parameters": sum(p.numel() for p in decoder.parameters()),
            "initialization": init_report,
        },
        "source_checkpoint": str(args.source_checkpoint.expanduser().resolve()),
        "sample_index": int(args.sample_index),
        "view_seed": int(args.view_seed),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "optimization_target": "teacher_rgb_mse_only",
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "best_logged_teacher_psnr": float(best_teacher_psnr),
        "best_logged_step": int(best_step),
        "history": history,
        "checkpoint": str(checkpoint_dir),
        "note": (
            "GT is diagnostic only in B2B-0B. This gate tests one-sample decoder capacity, "
            "not generalization and not the final B2B joint objective."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("[5/5] B2B-0B complete")
    print("================ B2B-0B one-sample capacity gate ================")
    print(f"Initial teacher PSNR : {float(initial_metrics['tiny_teacher_psnr']):.4f} dB")
    print(f"Final teacher PSNR   : {final_teacher_psnr:.4f} dB")
    print(f"Final GT PSNR        : {float(final_metrics['tiny_gt_psnr']):.4f} dB")
    print(f"Teacher vs GT PSNR   : {float(final_metrics['reae_teacher_gt_psnr']):.4f} dB")
    print(f"Interpretation       : {interpretation}")
    print(f"Saved                : {output_dir}")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
