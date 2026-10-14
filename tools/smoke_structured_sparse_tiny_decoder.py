#!/usr/bin/env python3
"""Learn structured Tiny-Decoder channel importance and export compact candidates.

This is a pre-cache Stage-B1 gate.  It starts from the already overfit dense Tiny
Conditional Decoder, converts every MemBlock to an exactly initialized gated
supernet, learns only the internal channel gates on the same deterministic sample,
and materializes several top-k ratios as real compact dense Conv blocks.  A short
per-candidate recovery checks whether the pruned topology remains trainable before
formal multi-view caching/training is started.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smoke_tiny_conditional_decoder import (
    _build_sample,
    _load_source,
    _metrics,
    _save_visuals,
    _source_targets,
)
from smoke_training_forward import resolve_runtime_dtype, validate_folded_checkpoint
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder
from swiftvr.models.tiny_decoder_sparsity import (
    StructuredSparseMemBlock,
    convert_dense_decoder_to_sparse,
    materialize_sparse_decoder,
    structured_gate_summary,
    structured_sparsity_penalty,
)
from swiftvr.training import build_fp32_adamw, build_grad_scaler
from swiftvr.training.tiny_decoder import LPIPSAlexLoss, tiny_decoder_objective


def _ratios(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(not 0.0 < item <= 1.0 for item in result):
        raise argparse.ArgumentTypeError("keep ratios must be comma-separated values in (0,1]")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("keep ratios must be unique")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--dense-decoder", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--view-seed", type=int, default=20260811)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--allow-dtype-mismatch", action="store_true")
    parser.add_argument("--attention-backend", choices=("sdpa",), default="sdpa")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")

    parser.add_argument("--gate-steps", type=int, default=50)
    parser.add_argument("--gate-learning-rate", type=float, default=5e-3)
    parser.add_argument("--sparsity-weight", type=float, default=0.05)
    parser.add_argument("--keep-ratios", type=_ratios, default=(0.75, 0.55, 0.40))
    parser.add_argument("--channel-multiple", type=int, default=8)
    parser.add_argument("--recovery-steps", type=int, default=50)
    parser.add_argument("--recovery-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gt-l2-weight", type=float, default=1.0)
    parser.add_argument("--teacher-l2-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=2.0)
    parser.add_argument("--lpips-microbatch-frames", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _objective(
    model: TinyConditionalDecoder,
    targets: dict[str, torch.Tensor],
    perceptual,
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    enabled = dtype in (torch.float16, torch.bfloat16)
    with torch.autocast("cuda", dtype=dtype, enabled=enabled):
        prediction = model(
            targets["z_sr"],
            targets["lq_input"],
            output_frames=int(targets["target"].shape[1]),
            clamp=False,
        )
        objective = tiny_decoder_objective(
            prediction,
            targets["target"],
            targets["reae_teacher"],
            perceptual=perceptual,
            gt_l2_weight=args.gt_l2_weight,
            teacher_l2_weight=args.teacher_l2_weight,
            lpips_weight=args.lpips_weight,
            lpips_microbatch_frames=args.lpips_microbatch_frames,
        )
    return prediction, objective


def _log_record(step: int, objective: dict[str, torch.Tensor], **extra) -> dict[str, float | int]:
    result: dict[str, float | int] = {"step": int(step)}
    for key, value in objective.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            result[key] = float(value.detach().float().item())
    result.update(extra)
    return result


def _candidate_name(ratio: float) -> str:
    return f"keep_{int(round(ratio * 100)):03d}"


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Structured-sparsity smoke requires CUDA")
    if args.gate_steps <= 0 or args.recovery_steps < 0:
        raise ValueError("gate-steps must be positive and recovery-steps non-negative")
    if args.gate_learning_rate <= 0 or args.recovery_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.sparsity_weight < 0 or args.channel_multiple <= 0:
        raise ValueError("sparsity-weight must be non-negative and channel-multiple positive")
    if args.max_grad_norm <= 0 or args.log_every <= 0:
        raise ValueError("max-grad-norm and log-every must be positive")

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    folded_config = validate_folded_checkpoint(
        args.base_checkpoint,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
    )
    dtype = resolve_runtime_dtype(
        args.dtype,
        folded_config,
        device,
        allow_mismatch=args.allow_dtype_mismatch,
    )
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")

    print("[1/6] Loading deterministic sample and Stage-A source targets...", flush=True)
    batch = _build_sample(args)
    source = _load_source(args, device=device, dtype=dtype)
    targets = _source_targets(source, batch, device=device, dtype=dtype)
    del source
    gc.collect()
    torch.cuda.empty_cache()

    print("[2/6] Loading dense Tiny Decoder and exact sparse supernet...", flush=True)
    dense = TinyConditionalDecoder.from_pretrained(
        args.dense_decoder, device=device, dtype=torch.float32
    ).eval()
    if dense.block_mode != "dense":
        raise ValueError(f"--dense-decoder must be dense, got {dense.block_mode!r}")
    sparse = convert_dense_decoder_to_sparse(dense).to(device=device).eval()

    enabled = dtype in (torch.float16, torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=enabled):
        dense_ref = dense(
            targets["z_sr"], targets["lq_input"],
            output_frames=int(targets["target"].shape[1]), clamp=False,
        )
        sparse_ref = sparse(
            targets["z_sr"], targets["lq_input"],
            output_frames=int(targets["target"].shape[1]), clamp=False,
        )
    equivalence_max_abs = float(
        (dense_ref.float() - sparse_ref.float()).abs().max().item()
    )
    print(json.dumps({"dense_sparse_max_abs": equivalence_max_abs}), flush=True)
    if equivalence_max_abs > 1e-5:
        raise RuntimeError(
            f"Dense->sparse initialization is not exact enough: max_abs={equivalence_max_abs}"
        )
    del dense_ref, sparse_ref, dense
    gc.collect()
    torch.cuda.empty_cache()

    perceptual = None
    if args.lpips_weight > 0:
        perceptual = LPIPSAlexLoss().to(device=device).eval()
        for parameter in perceptual.parameters():
            parameter.requires_grad_(False)

    print("[3/6] Learning structured channel gates only...", flush=True)
    for parameter in sparse.parameters():
        parameter.requires_grad_(False)
    gate_count = 0
    for module in sparse.modules():
        if isinstance(module, StructuredSparseMemBlock):
            module.channel_gate.requires_grad_(True)
            gate_count += int(module.channel_gate.numel())
    if gate_count <= 0:
        raise RuntimeError("Sparse supernet exposes no channel gates")
    sparse.train()
    gate_optimizer = build_fp32_adamw(
        sparse,
        learning_rate=args.gate_learning_rate,
        weight_decay=0.0,
        eps=1e-8,
    )
    gate_scaler = build_grad_scaler(device, dtype)
    gate_history: list[dict[str, float | int]] = []
    for step in range(1, args.gate_steps + 1):
        gate_optimizer.zero_grad(set_to_none=True)
        prediction, task = _objective(sparse, targets, perceptual, args, dtype=dtype)
        penalty = structured_sparsity_penalty(sparse)
        total = task["loss"] + float(args.sparsity_weight) * penalty
        gate_scaler.scale(total).backward()
        gate_scaler.unscale_(gate_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in sparse.parameters() if p.requires_grad], args.max_grad_norm
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gate grad norm at step {step}")
        gate_scaler.step(gate_optimizer)
        gate_scaler.update()
        if step == 1 or step % args.log_every == 0 or step == args.gate_steps:
            record = _log_record(
                step,
                task,
                task_loss=float(task["loss"].detach().float().item()),
                sparsity_penalty=float(penalty.detach().float().item()),
                total_loss=float(total.detach().float().item()),
                grad_norm=float(grad_norm.detach().float().item()),
            )
            gate_history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        del prediction

    sparse.eval()
    sparse_dir = output / "sparse_supernet"
    sparse.save_pretrained(sparse_dir)
    _write_json(output / "gate_history.json", gate_history)
    _write_json(output / "gate_summary.json", structured_gate_summary(sparse))

    print("[4/6] Materializing compact top-k candidates...", flush=True)
    candidates: dict[str, object] = {}
    for ratio in args.keep_ratios:
        name = _candidate_name(ratio)
        candidate_dir = output / name
        compact, manifest = materialize_sparse_decoder(
            sparse, keep_ratio=ratio, multiple=args.channel_multiple
        )
        compact = compact.to(device=device, dtype=torch.float32)
        compact.eval()
        with torch.no_grad():
            initial_prediction, initial_objective = _objective(
                compact, targets, perceptual, args, dtype=dtype
            )
        initial_metrics = _metrics(
            initial_prediction, targets["target"], targets["reae_teacher"]
        )
        _save_visuals(
            candidate_dir,
            prefix="initial",
            lq_input=targets["lq_input"],
            target=targets["target"],
            reae_teacher=targets["reae_teacher"],
            tiny=initial_prediction,
        )
        initial_loss = {
            key: float(value.detach().float().item())
            for key, value in initial_objective.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        del initial_prediction

        print(f"[5/6] Recovering {name} for {args.recovery_steps} steps...", flush=True)
        for parameter in compact.parameters():
            parameter.requires_grad_(True)
        compact.train()
        optimizer = build_fp32_adamw(
            compact,
            learning_rate=args.recovery_learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8,
        )
        scaler = build_grad_scaler(device, dtype)
        recovery_history: list[dict[str, float | int]] = []
        for step in range(1, args.recovery_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            prediction, objective = _objective(
                compact, targets, perceptual, args, dtype=dtype
            )
            scaler.scale(objective["loss"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                compact.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite recovery grad norm for {name} at step {step}"
                )
            scaler.step(optimizer)
            scaler.update()
            if step == 1 or step % args.log_every == 0 or step == args.recovery_steps:
                record = _log_record(
                    step, objective,
                    grad_norm=float(grad_norm.detach().float().item()),
                )
                recovery_history.append(record)
                print(json.dumps({"candidate": name, **record}, sort_keys=True), flush=True)
            del prediction

        compact.eval()
        with torch.no_grad():
            final_prediction, final_objective = _objective(
                compact, targets, perceptual, args, dtype=dtype
            )
        final_metrics = _metrics(
            final_prediction, targets["target"], targets["reae_teacher"]
        )
        _save_visuals(
            candidate_dir,
            prefix="final",
            lq_input=targets["lq_input"],
            target=targets["target"],
            reae_teacher=targets["reae_teacher"],
            tiny=final_prediction,
        )
        final_loss = {
            key: float(value.detach().float().item())
            for key, value in final_objective.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        compact.save_pretrained(candidate_dir / "tiny_decoder")
        _write_json(candidate_dir / "selection_manifest.json", manifest)
        _write_json(candidate_dir / "recovery_history.json", recovery_history)
        candidate_summary = {
            "keep_ratio": float(ratio),
            "block_internal_channels": list(compact.block_internal_channels),
            "parameters": int(sum(p.numel() for p in compact.parameters())),
            "initial_loss": initial_loss,
            "initial_metrics": initial_metrics,
            "final_loss": final_loss,
            "final_metrics": final_metrics,
        }
        _write_json(candidate_dir / "summary.json", candidate_summary)
        candidates[name] = candidate_summary
        del compact, final_prediction, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    print("[6/6] Structured-sparsity smoke complete.", flush=True)
    summary = {
        "status": "PASS",
        "dense_decoder": str(args.dense_decoder.expanduser().resolve()),
        "dense_sparse_max_abs": equivalence_max_abs,
        "gate_channels": gate_count,
        "gate_steps": int(args.gate_steps),
        "sparsity_weight": float(args.sparsity_weight),
        "keep_ratios": list(args.keep_ratios),
        "channel_multiple": int(args.channel_multiple),
        "recovery_steps": int(args.recovery_steps),
        "candidates": candidates,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
