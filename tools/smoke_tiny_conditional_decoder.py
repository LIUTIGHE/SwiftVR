#!/usr/bin/env python3
"""Single-sample Stage-B gate for the SwiftVR Tiny Conditional Decoder.

This intentionally freezes the complete long-run Stage-A source model. The source
ReAE encoder + prompt-free/no-time DiT are executed once to produce ``z_SR``; the
same ``z_SR`` is decoded by the frozen ReAE decoder to form the decoder-teacher RGB
target. Only the new tiny decoder is optimized.

The gate is not the formal training recipe. It answers four questions before DDP is
introduced:
  1. does the 4x16x16 condition packing align exactly with SwiftVR latents;
  2. can the tiny decoder backpropagate and overfit one deterministic view;
  3. does dual GT/ReAE supervision reduce the intended losses;
  4. can its checkpoint be saved/reloaded independently of Stage-A weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smoke_training_forward import (
    configure_train_scope,
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training import (
    build_fp32_adamw,
    build_grad_scaler,
    cast_trainable_parameters,
    load_delta_checkpoint,
)
from swiftvr.training.distillation import (
    DeterministicTripletViewDataset,
    SwiftVRVelocityDistillationForward,
)
from swiftvr.training.forward import decode_reae_clip, encode_reae_clip, prepare_training_batch
from swiftvr.training.perceptual_review import make_comparison_frame
from swiftvr.training.reference import extract_transformer_sample
from swiftvr.training.stage3 import VideoMetricAccumulator
from swiftvr.training.tiny_decoder import LPIPSAlexLoss, tiny_decoder_objective


def _csv_ints(value: str, *, length: int) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(parsed) != length:
        raise argparse.ArgumentTypeError(f"expected {length} comma-separated integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        required=True,
        help="Long-run Stage-A delta checkpoint (for example the 170k/best run).",
    )
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

    parser.add_argument("--condition-channels", type=int, default=32)
    parser.add_argument("--decoder-channels", default="192,128,64,32")
    parser.add_argument("--blocks-per-stage", default="2,2,2,1")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gt-l2-weight", type=float, default=1.0)
    parser.add_argument("--teacher-l2-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=2.0)
    parser.add_argument("--lpips-microbatch-frames", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _build_sample(args: argparse.Namespace) -> Mapping[str, object]:
    dataset = TripletVideoDataset(
        args.manifest,
        split=args.split,
        training=True,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        load_hq=False,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        drop_short_sequences=True,
        path_root=args.path_root.expanduser().resolve(),
        verify_paths=args.verify_paths,
    )
    views = DeterministicTripletViewDataset(
        dataset,
        views_per_record=1,
        view_seed=args.view_seed,
    )
    if not 0 <= args.sample_index < len(views):
        raise IndexError(f"sample-index={args.sample_index} outside [0,{len(views)})")
    loader = DataLoader(Subset(views, [args.sample_index]), batch_size=1, shuffle=False)
    return next(iter(loader))


def _load_source(
    args: argparse.Namespace,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SwiftVRVelocityDistillationForward:
    base = args.base_checkpoint.expanduser().resolve()
    reae = ReAE(str(base / args.reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(base),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    configure_train_scope(reae, transformer, "adapter")
    closure = SwiftVRVelocityDistillationForward(
        reae,
        transformer,
        attention_backend=args.attention_backend,
        prepare_transformer=False,
    )
    # Stage-A delta checkpoints were saved with FP32 adapter parameters.
    cast_trainable_parameters(closure, dtype=torch.float32)
    load_delta_checkpoint(args.source_checkpoint, closure, strict=True)
    for parameter in closure.parameters():
        parameter.requires_grad_(False)
    closure.to(device=device)
    closure.reae.eval()
    closure.transformer.eval()
    closure.transformer.prepare_for_inference(
        attention_backend=args.attention_backend,
        use_torch_compile=False,
    )
    return closure


@torch.no_grad()
def _source_targets(
    source: SwiftVRVelocityDistillationForward,
    batch: Mapping[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    moved = move_video_batch(batch, device=device, dtype=dtype)
    prepared = prepare_training_batch(moved)
    lq_input = prepared["lq_input"]
    target = prepared["target"]
    if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("prepared batch is missing lq_input/target")

    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        z_lq_ntchw = encode_reae_clip(source.reae, lq_input, require_4k_plus_1=True)
        z_lq = z_lq_ntchw.permute(0, 2, 1, 3, 4).contiguous()
        velocity = extract_transformer_sample(source.transformer(z_lq, return_dict=True))
        if velocity.shape != z_lq.shape:
            raise RuntimeError(
                f"source velocity shape {tuple(velocity.shape)} != latent {tuple(z_lq.shape)}"
            )
        z_sr = (z_lq - velocity).permute(0, 2, 1, 3, 4).contiguous()
        reae_teacher = decode_reae_clip(
            source.reae,
            z_sr,
            output_frames=int(target.shape[1]),
            clamp=False,
        )
    return {
        "lq_input": lq_input.detach(),
        "target": target.detach(),
        "z_sr": z_sr.detach(),
        "reae_teacher": reae_teacher.detach(),
    }


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reae_teacher: torch.Tensor,
) -> dict[str, float | int]:
    tiny_gt = VideoMetricAccumulator()
    tiny_teacher = VideoMetricAccumulator()
    teacher_gt = VideoMetricAccumulator()
    tiny_gt.update(prediction, target, clamp=True)
    tiny_teacher.update(prediction, reae_teacher, clamp=True)
    teacher_gt.update(reae_teacher, target, clamp=True)
    result: dict[str, float | int] = {}
    for prefix, accumulator in (
        ("tiny_gt", tiny_gt),
        ("tiny_teacher", tiny_teacher),
        ("reae_teacher_gt", teacher_gt),
    ):
        result.update({f"{prefix}_{key}": value for key, value in accumulator.compute().items()})
    return result


def _save_visuals(
    output_dir: Path,
    *,
    prefix: str,
    lq_input: torch.Tensor,
    target: torch.Tensor,
    reae_teacher: torch.Tensor,
    tiny: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = int(target.shape[1])
    indices = tuple(dict.fromkeys((0, frames // 2, frames - 1)))
    for frame_index in indices:
        image = make_comparison_frame(
            {
                "LQ bicubic": lq_input[0, frame_index].detach().float().clamp(0, 1).cpu(),
                "GT": target[0, frame_index].detach().float().clamp(0, 1).cpu(),
                "ReAE teacher": reae_teacher[0, frame_index].detach().float().clamp(0, 1).cpu(),
                "Tiny decoder": tiny[0, frame_index].detach().float().clamp(0, 1).cpu(),
            }
        )
        image.save(output_dir / f"{prefix}_frame_{frame_index:03d}.png")


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Tiny-decoder source gate requires CUDA")
    if args.steps <= 0 or args.learning_rate <= 0 or args.log_every <= 0:
        raise ValueError("steps, learning-rate and log-every must be positive")
    if args.max_grad_norm <= 0 or args.lpips_microbatch_frames <= 0:
        raise ValueError("max-grad-norm and lpips-microbatch-frames must be positive")
    if min(args.gt_l2_weight, args.teacher_l2_weight, args.lpips_weight) < 0:
        raise ValueError("loss weights must be non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print("[1/5] Loading deterministic overfit sample...", flush=True)
    batch = _build_sample(args)
    print("[2/5] Loading frozen long-run Stage-A source...", flush=True)
    source = _load_source(args, device=device, dtype=dtype)
    print("[3/5] Computing z_SR and frozen ReAE decoder target once...", flush=True)
    source_targets = _source_targets(source, batch, device=device, dtype=dtype)

    # The 3.8B source is not needed after z_SR/ReAE-target creation. Releasing it
    # before LPIPS/tiny-decoder allocation keeps the smoke gate bounded in memory.
    del source
    gc.collect()
    torch.cuda.empty_cache()

    channels = _csv_ints(args.decoder_channels, length=4)
    blocks = _csv_ints(args.blocks_per_stage, length=4)
    tiny = TinyConditionalDecoder(
        latent_channels=int(source_targets["z_sr"].shape[2]),
        condition_channels=args.condition_channels,
        channels=channels,
        blocks_per_stage=blocks,
        temporal_factor=4,
        spatial_factor=16,
        patch_size=2,
        frames_to_trim=3,
    ).to(device=device)
    cast_info = cast_trainable_parameters(tiny, dtype=torch.float32)
    tiny.train()

    perceptual = None
    if args.lpips_weight > 0:
        perceptual = LPIPSAlexLoss().to(device=device)
        perceptual.eval()

    optimizer = build_fp32_adamw(
        tiny,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    scaler = build_grad_scaler(device, dtype)
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)

    def predict() -> torch.Tensor:
        with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            return tiny(
                source_targets["z_sr"],
                source_targets["lq_input"],
                output_frames=int(source_targets["target"].shape[1]),
                clamp=False,
            )

    tiny.eval()
    with torch.no_grad():
        initial_prediction = predict()
    initial_metrics = _metrics(
        initial_prediction,
        source_targets["target"],
        source_targets["reae_teacher"],
    )
    _save_visuals(
        output_dir,
        prefix="initial",
        tiny=initial_prediction,
        lq_input=source_targets["lq_input"],
        target=source_targets["target"],
        reae_teacher=source_targets["reae_teacher"],
    )
    del initial_prediction

    print("[4/5] Overfitting Tiny Conditional Decoder...", flush=True)
    tiny.train()
    history: list[dict[str, float | int]] = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = predict()
        objective = tiny_decoder_objective(
            prediction,
            source_targets["target"],
            source_targets["reae_teacher"],
            perceptual=perceptual,
            gt_l2_weight=args.gt_l2_weight,
            teacher_l2_weight=args.teacher_l2_weight,
            lpips_weight=args.lpips_weight,
            lpips_microbatch_frames=args.lpips_microbatch_frames,
        )
        scaler.scale(objective["loss"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(tiny.parameters(), args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}: {grad_norm}")
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                **{
                    key: float(value.detach().float().item())
                    for key, value in objective.items()
                },
                "grad_norm": float(grad_norm.detach().float().item()),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        del prediction, objective

    print("[5/5] Saving/reloading decoder and final diagnostics...", flush=True)
    tiny.eval()
    model_dir = tiny.save_pretrained(output_dir / "tiny_decoder")
    reloaded = TinyConditionalDecoder.from_pretrained(
        model_dir,
        device=device,
        dtype=torch.float32,
    ).eval()
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=dtype, enabled=autocast_enabled
    ):
        final_prediction = reloaded(
            source_targets["z_sr"],
            source_targets["lq_input"],
            output_frames=int(source_targets["target"].shape[1]),
            clamp=False,
        )
    final_metrics = _metrics(
        final_prediction,
        source_targets["target"],
        source_targets["reae_teacher"],
    )
    _save_visuals(
        output_dir,
        prefix="final",
        tiny=final_prediction,
        lq_input=source_targets["lq_input"],
        target=source_targets["target"],
        reae_teacher=source_targets["reae_teacher"],
    )

    summary = {
        "kind": "swiftvr_stage_b1_tiny_decoder_smoke",
        "base_checkpoint": str(args.base_checkpoint.expanduser().resolve()),
        "source_checkpoint": str(args.source_checkpoint.expanduser().resolve()),
        "manifest": [str(path.expanduser().resolve()) for path in args.manifest],
        "sample_index": args.sample_index,
        "view_seed": args.view_seed,
        "runtime_dtype": str(dtype).removeprefix("torch."),
        "decoder_config": tiny.config_dict,
        "trainable": cast_info,
        "parameter_count": sum(parameter.numel() for parameter in tiny.parameters()),
        "loss_weights": {
            "gt_l2": args.gt_l2_weight,
            "teacher_l2": args.teacher_l2_weight,
            "lpips_each": args.lpips_weight,
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "history": history,
        "model_dir": str(model_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
