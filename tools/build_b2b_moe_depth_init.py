#!/usr/bin/env python3
"""Build a reduced-depth sparse-MoE student from a trained same-width MoE source.

This is the canonical depth-only initializer for later B2B/M8 architecture gates.
It preserves every retained block exactly, including attention, adapters, shared
and routed experts, and learned router weights.  Only whole Transformer blocks
are removed.

The default M8 use case is M5 D1024/L30 -> D1024/L20.  Block redundancy is
measured on a deterministic calibration prefix using

    residual_ratio + (1 - input_output_cosine)

and pruning is constrained by edge protection, a minimum number of kept blocks
per contiguous region, and a maximum run length of consecutive pruned blocks.
No decoder, teacher velocity, GT loss, or new random expert/router initialization
participates in this transfer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_b2b_moe_init import BlockRedundancyCollector
from tools.smoke_training_forward import (
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time_moe import (
    WanTransformer3DModelPromptFreeNoTimeMoE,
)
from swiftvr.training.b2b_moe import (
    B2BMoESpec,
    expected_moe_shape,
    parameter_accounting,
    transformer_moe_shape,
)
from swiftvr.training.b2b_moe_training import B2BMoEVelocityDistillationForward
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.reference import sha256_file


M8_ARCHITECTURE = "m8-d1024-l20"
M8_SPEC = B2BMoESpec(num_layers=20)
M5_SOURCE_SPEC = B2BMoESpec()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument(
        "--source-checkpoint",
        type=Path,
        required=True,
        help="Trained same-width MoE source checkpoint; M8 expects M5 D1024/L30.",
    )
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--architecture", choices=(M8_ARCHITECTURE,), default=M8_ARCHITECTURE)
    p.add_argument("--target-layers", type=int, default=M8_SPEC.num_layers)
    p.add_argument("--protect-edge-blocks", type=int, default=1)
    p.add_argument(
        "--region-size",
        type=int,
        default=6,
        help="Contiguous source-layer region size used by the spacing constraint.",
    )
    p.add_argument(
        "--min-keep-per-region",
        type=int,
        default=3,
        help="Minimum retained blocks in every region (last shorter region included).",
    )
    p.add_argument(
        "--max-consecutive-pruned",
        type=int,
        default=2,
        help="Reject a candidate if it would create a longer consecutive prune run.",
    )
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--split", default="train")
    p.add_argument("--clip-length", type=int, default=13)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--views-per-record", type=int, default=8)
    p.add_argument("--view-seed", type=int, default=20260805)
    p.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    p.add_argument("--vertical-flip-probability", type=float, default=0.0)
    p.add_argument("--calibration-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    p.add_argument("--allow-dtype-mismatch", action="store_true")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress-every", type=int, default=8)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "target_layers",
        "region_size",
        "min_keep_per_region",
        "max_consecutive_pruned",
        "clip_length",
        "crop_size",
        "scale",
        "views_per_record",
        "calibration_samples",
        "batch_size",
        "progress_every",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.protect_edge_blocks < 0:
        raise ValueError("--num-workers/--protect-edge-blocks must be non-negative")
    if args.architecture == M8_ARCHITECTURE and args.target_layers != M8_SPEC.num_layers:
        raise ValueError(f"{M8_ARCHITECTURE} is locked to {M8_SPEC.num_layers} layers")


def _slice_batch(batch: dict[str, object], count: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value[:count]
        elif isinstance(value, list):
            result[key] = value[:count]
        elif isinstance(value, tuple):
            result[key] = value[:count]
        else:
            result[key] = value
    return result


def _max_pruned_run(pruned: set[int], total: int) -> int:
    longest = current = 0
    for index in range(total):
        if index in pruned:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def select_blocks_constrained(
    residual_ratio: torch.Tensor,
    cosine_similarity: torch.Tensor,
    *,
    keep_layers: int,
    protect_edge_blocks: int,
    region_size: int,
    min_keep_per_region: int,
    max_consecutive_pruned: int,
) -> dict[str, object]:
    """Greedily prune lowest-redundancy-score blocks subject to spacing constraints."""
    residual = residual_ratio.detach().float().cpu().reshape(-1)
    cosine = cosine_similarity.detach().float().cpu().reshape(-1)
    if residual.shape != cosine.shape or residual.numel() == 0:
        raise ValueError("redundancy vectors must have the same non-empty shape")
    if not torch.isfinite(residual).all() or not torch.isfinite(cosine).all():
        raise ValueError("redundancy vectors contain non-finite values")
    total = int(residual.numel())
    keep_layers = int(keep_layers)
    protect = int(protect_edge_blocks)
    region_size = int(region_size)
    min_keep = int(min_keep_per_region)
    max_run = int(max_consecutive_pruned)
    if not 0 < keep_layers <= total:
        raise ValueError(f"keep_layers must lie in [1,{total}]")
    if protect * 2 >= total:
        raise ValueError("edge protection leaves no pruning interior")
    if min_keep > region_size:
        raise ValueError("min_keep_per_region cannot exceed region_size")

    redundancy = residual + (1.0 - cosine.clamp(-1.0, 1.0))
    candidates = list(range(protect, total - protect))
    candidates.sort(key=lambda index: (float(redundancy[index]), index))
    prune_target = total - keep_layers
    pruned: set[int] = set()

    def region_bounds(index: int) -> tuple[int, int]:
        start = (index // region_size) * region_size
        return start, min(start + region_size, total)

    def allowed(index: int) -> bool:
        trial = set(pruned)
        trial.add(index)
        start, end = region_bounds(index)
        region_len = end - start
        region_pruned = sum(value in trial for value in range(start, end))
        required = min(min_keep, region_len)
        if region_len - region_pruned < required:
            return False
        if _max_pruned_run(trial, total) > max_run:
            return False
        return True

    for index in candidates:
        if len(pruned) >= prune_target:
            break
        if allowed(index):
            pruned.add(index)
    if len(pruned) != prune_target:
        raise RuntimeError(
            f"Spacing constraints allowed only {len(pruned)}/{prune_target} required prunes; "
            "relax region/min-keep/max-run constraints explicitly."
        )

    pruned_list = sorted(pruned)
    kept = [index for index in range(total) if index not in pruned]
    regions = []
    for start in range(0, total, region_size):
        end = min(start + region_size, total)
        region_kept = [i for i in range(start, end) if i not in pruned]
        regions.append({"start": start, "end_exclusive": end, "kept": region_kept})
    return {
        "source_blocks": list(range(total)),
        "kept_source_blocks": kept,
        "pruned_source_blocks": pruned_list,
        "residual_ratio": [float(v) for v in residual.tolist()],
        "cosine_similarity": [float(v) for v in cosine.tolist()],
        "redundancy_score": [float(v) for v in redundancy.tolist()],
        "protect_edge_blocks": protect,
        "region_size": region_size,
        "min_keep_per_region": min_keep,
        "max_consecutive_pruned": max_run,
        "observed_max_consecutive_pruned": _max_pruned_run(pruned, total),
        "regions": regions,
    }


def _build_depth_student(source: WanTransformer3DModelPromptFreeNoTimeMoE, layers: int):
    shape = transformer_moe_shape(source)
    cfg = source.config
    return WanTransformer3DModelPromptFreeNoTimeMoE(
        patch_size=tuple(cfg.patch_size),
        num_attention_heads=int(shape["num_heads"]),
        attention_head_dim=int(shape["head_dim"]),
        in_channels=int(cfg.in_channels),
        out_channels=int(cfg.out_channels),
        text_dim=int(getattr(cfg, "text_dim", 4096)),
        freq_dim=int(getattr(cfg, "freq_dim", 256)),
        ffn_dim=int(shape["active_ffn_dim"]),
        num_layers=int(layers),
        cross_attn_norm=bool(getattr(cfg, "cross_attn_norm", True)),
        qk_norm=str(getattr(cfg, "qk_norm", "rms_norm_across_heads")),
        eps=float(getattr(cfg, "eps", 1e-6)),
        image_dim=getattr(cfg, "image_dim", None),
        added_kv_proj_dim=getattr(cfg, "added_kv_proj_dim", None),
        rope_max_seq_len=int(getattr(cfg, "rope_max_seq_len", 1024)),
        pos_embed_seq_len=getattr(cfg, "pos_embed_seq_len", None),
        enable_swa=bool(getattr(cfg, "enable_swa", True)),
        self_attn_window_hw=tuple(getattr(cfg, "self_attn_window_hw", (16, 16))),
        use_torch_compile=False,
        compile_mode="default",
        adapter_dim=int(shape["adapter_dim"]),
        folded_timestep=float(getattr(cfg, "folded_timestep", 1000.0)),
        time_condition_folded=True,
        shared_expert_dim=int(shape["shared_expert_dim"]),
        normal_expert_dim=int(shape["normal_expert_dim"]),
        num_experts=int(shape["num_experts"]),
        top_k=int(shape["top_k"]),
    )


def _copy_depth_subset(
    source: WanTransformer3DModelPromptFreeNoTimeMoE,
    student: WanTransformer3DModelPromptFreeNoTimeMoE,
    kept_blocks: list[int],
) -> None:
    source_state = source.state_dict()
    target_state = student.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for target_key, target_value in target_state.items():
        parts = target_key.split(".")
        if len(parts) >= 3 and parts[0] == "blocks" and parts[1].isdigit():
            student_index = int(parts[1])
            if student_index >= len(kept_blocks):
                raise RuntimeError(f"student block index out of mapping: {student_index}")
            source_key = ".".join(["blocks", str(kept_blocks[student_index]), *parts[2:]])
        else:
            source_key = target_key
        if source_key not in source_state:
            raise KeyError(f"Missing source tensor {source_key!r} for target {target_key!r}")
        source_value = source_state[source_key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise ValueError(
                f"Shape mismatch {source_key} {tuple(source_value.shape)} -> "
                f"{target_key} {tuple(target_value.shape)}"
            )
        mapped[target_key] = source_value.detach().clone()
    student.load_state_dict(mapped, strict=True)


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    base_root = args.base_checkpoint.expanduser().resolve()
    source_root = args.source_checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    folded_config = validate_folded_checkpoint(
        base_root,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
    )
    dtype = resolve_runtime_dtype(
        args.dtype,
        folded_config,
        device,
        allow_mismatch=args.allow_dtype_mismatch,
    )
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    source_config = source_root / args.transformer_subfolder / "config.json"
    source_weights = source_root / args.transformer_subfolder / "diffusion_pytorch_model.safetensors"
    if not source_config.is_file() or not source_weights.is_file():
        raise FileNotFoundError(
            "Source checkpoint must contain transformer/config.json and "
            "transformer/diffusion_pytorch_model.safetensors"
        )

    dataset = TripletVideoDataset(
        args.manifest,
        split=args.split,
        training=True,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        horizontal_flip_probability=args.horizontal_flip_probability,
        vertical_flip_probability=args.vertical_flip_probability,
        drop_short_sequences=True,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
    )
    views = DeterministicTripletViewDataset(
        dataset,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
    )
    calibration_limit = min(len(views), int(args.calibration_samples))
    loader = DataLoader(
        views,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    reae = ReAE(str(base_root / args.reae_filename))
    source = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
        str(source_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    source_shape = transformer_moe_shape(source)
    expected_source = expected_moe_shape(M5_SOURCE_SPEC)
    if source_shape != expected_source:
        raise ValueError(f"M8 depth source must be M5 D1024/L30: {source_shape} != {expected_source}")
    if args.target_layers >= int(source_shape["num_layers"]):
        raise ValueError("target depth must be smaller than source depth")

    for parameter in reae.parameters():
        parameter.requires_grad_(False)
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    reae.to(device=device, dtype=dtype).eval()
    source.to(device=device, dtype=dtype).eval()
    closure = B2BMoEVelocityDistillationForward(
        reae,
        source,
        attention_backend=args.attention_backend,
        gradient_checkpointing=False,
    ).eval()
    collector = BlockRedundancyCollector(source)
    processed = 0
    started = time.perf_counter()
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    try:
        with torch.no_grad():
            for batch_cpu in loader:
                if processed >= calibration_limit:
                    break
                frame_indices = batch_cpu.get("frame_indices")
                if not isinstance(frame_indices, torch.Tensor):
                    raise TypeError("Calibration batch is missing frame_indices")
                count = int(frame_indices.shape[0])
                remaining = calibration_limit - processed
                if count > remaining:
                    batch_cpu = _slice_batch(batch_cpu, remaining)
                    count = remaining
                batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                with torch.autocast(
                    "cuda",
                    dtype=dtype,
                    enabled=device.type == "cuda" and autocast_enabled,
                ):
                    closure(batch)
                processed += count
                if processed == calibration_limit or processed % args.progress_every == 0:
                    print(
                        f"calibration {processed}/{calibration_limit} "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
    finally:
        collector.close()
    if processed != calibration_limit:
        raise RuntimeError(f"Calibration processed {processed}, expected {calibration_limit}")

    scores = collector.scores()
    selection = select_blocks_constrained(
        scores["block_residual_ratio"],
        scores["block_cosine_similarity"],
        keep_layers=args.target_layers,
        protect_edge_blocks=args.protect_edge_blocks,
        region_size=args.region_size,
        min_keep_per_region=args.min_keep_per_region,
        max_consecutive_pruned=args.max_consecutive_pruned,
    )
    kept = [int(value) for value in selection["kept_source_blocks"]]
    print(
        f"building {args.architecture}: kept source blocks={kept}; "
        f"pruned={selection['pruned_source_blocks']}",
        flush=True,
    )

    source.to(device="cpu")
    student = _build_depth_student(source, args.target_layers)
    _copy_depth_subset(source, student, kept)
    target_shape = transformer_moe_shape(student)
    expected_target = expected_moe_shape(B2BMoESpec(num_layers=args.target_layers))
    if target_shape != expected_target:
        raise RuntimeError(f"Target shape mismatch: {target_shape} != {expected_target}")

    score_path = output / "block_redundancy.safetensors"
    save_file({key: value.contiguous() for key, value in scores.items()}, str(score_path))
    transformer_dir = output / args.transformer_subfolder
    student.to(device="cpu", dtype=dtype)
    student.save_pretrained(str(transformer_dir), safe_serialization=True)

    report = {
        "kind": "swiftvr_b2b_same_width_moe_depth_init",
        "architecture": args.architecture,
        "method": "trained_m5_same_width_exact_block_copy_with_constrained_redundancy_pruning",
        "base_checkpoint": str(base_root),
        "source_checkpoint": str(source_root),
        "source_config_sha256": sha256_file(source_config),
        "source_weights_sha256": sha256_file(source_weights),
        "source_shape": source_shape,
        "student_shape": target_shape,
        "parameter_accounting": parameter_accounting(student),
        "saved_dtype": str(dtype).removeprefix("torch."),
        "selection": selection,
        "calibration": {
            "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
            "split": args.split,
            "clip_length": args.clip_length,
            "crop_size": args.crop_size,
            "scale": args.scale,
            "views_per_record": args.views_per_record,
            "view_seed": args.view_seed,
            "samples": processed,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "design": {
            "same_width_exact_copy": True,
            "router_reinitialized": False,
            "experts_reinitialized": False,
            "decoder_used": False,
            "gt_used": False,
        },
        "artifacts": {
            "transformer": str(transformer_dir),
            "block_redundancy": str(score_path),
        },
    }
    (output / "moe_depth_init_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "architecture": args.architecture,
                "source_shape": source_shape,
                "student_shape": target_shape,
                "kept_source_blocks": kept,
                "pruned_source_blocks": selection["pruned_source_blocks"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
