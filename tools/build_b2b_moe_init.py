#!/usr/bin/env python3
"""Build D1536-TA initialized sparse-MoE students for M5 or M7A.

A deterministic calibration prefix measures activation-RMS importance. Residual
channels/attention heads are selected structurally; dense FFN neurons are
repartitioned into shared and routed experts.

For M7A (D1152/H9/L25), the same calibration also measures each D1536 teacher
block's normalized residual contribution and input/output cosine similarity. Five
interior blocks with the smallest ``residual_ratio + (1-cosine)`` are removed,
and the ordered 25-block teacher subset initializes the student. No uniform layer
skipping pattern is imposed.

The output is a reloadable transformer-only checkpoint plus an audit report. No
decoder or GT supervision is used.
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

from tools.smoke_training_forward import (
    move_video_batch,
    resolve_runtime_dtype,
    validate_folded_checkpoint,
)
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.training.b2a_width import (
    ActivationImportanceCollector,
    B2ACompactVelocityDistillationForward,
    B2AWidthSpec,
)
from swiftvr.training.b2b_moe import (
    MOE_ARCHITECTURES,
    M5_MOE_ARCHITECTURE,
    build_moe_transformer_from_teacher,
    moe_spec_from_name,
    parameter_accounting,
    select_teacher_blocks_by_redundancy,
    transfer_d1536_to_moe,
    transformer_moe_shape,
    validate_d1536_ta,
)
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.reference import sha256_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--architecture",
        choices=tuple(MOE_ARCHITECTURES),
        default=M5_MOE_ARCHITECTURE,
        help="Sparse-MoE student architecture; default preserves the validated M5 path.",
    )
    p.add_argument(
        "--protect-edge-blocks",
        type=int,
        default=1,
        help="For reduced-depth architectures, exclude this many teacher blocks at each edge from pruning.",
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
    p.add_argument("--router-seed", type=int, default=20260831)
    p.add_argument("--router-init-std", type=float, default=1e-3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress-every", type=int, default=8)
    return p


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


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


def _score_stats(value: torch.Tensor) -> dict[str, float]:
    x = value.detach().float().cpu()
    return {
        "min": float(x.min().item()),
        "mean": float(x.mean().item()),
        "max": float(x.max().item()),
        "std": float(x.std(unbiased=False).item()),
    }


class BlockRedundancyCollector:
    """Accumulate block input/output residual and cosine statistics on device."""

    def __init__(self, transformer: torch.nn.Module) -> None:
        self.num_layers = len(transformer.blocks)
        self.input_sq: list[torch.Tensor | None] = [None] * self.num_layers
        self.output_sq: list[torch.Tensor | None] = [None] * self.num_layers
        self.cross: list[torch.Tensor | None] = [None] * self.num_layers
        self.delta_sq: list[torch.Tensor | None] = [None] * self.num_layers
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for index, block in enumerate(transformer.blocks):
            self._handles.append(block.register_forward_hook(self._hook(index)))

    @staticmethod
    def _add(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
        value = value.detach()
        return value if current is None else current + value

    def _hook(self, index: int):
        def hook(_module, inputs, output):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"Block {index} redundancy hook expected tensor input")
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Block {index} redundancy hook expected tensor output")
            x = inputs[0].detach().float()
            y = output.detach().float()
            if x.shape != y.shape:
                raise ValueError(
                    f"Block {index} input/output shapes differ: {tuple(x.shape)} vs {tuple(y.shape)}"
                )
            delta = y - x
            self.input_sq[index] = self._add(self.input_sq[index], x.square().sum())
            self.output_sq[index] = self._add(self.output_sq[index], y.square().sum())
            self.cross[index] = self._add(self.cross[index], (x * y).sum())
            self.delta_sq[index] = self._add(self.delta_sq[index], delta.square().sum())
        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def scores(self) -> dict[str, torch.Tensor]:
        groups = (self.input_sq, self.output_sq, self.cross, self.delta_sq)
        if any(any(value is None for value in group) for group in groups):
            raise RuntimeError("Block redundancy calibration did not execute every block")
        input_sq = torch.stack([value for value in self.input_sq if value is not None]).double().cpu()
        output_sq = torch.stack([value for value in self.output_sq if value is not None]).double().cpu()
        cross = torch.stack([value for value in self.cross if value is not None]).double().cpu()
        delta_sq = torch.stack([value for value in self.delta_sq if value is not None]).double().cpu()
        residual_ratio = torch.sqrt(delta_sq / input_sq.clamp_min(1e-24)).float()
        cosine = (cross / torch.sqrt(input_sq * output_sq).clamp_min(1e-24)).float()
        redundancy = residual_ratio + (1.0 - cosine.clamp(-1.0, 1.0))
        return {
            "block_residual_ratio": residual_ratio,
            "block_cosine_similarity": cosine,
            "block_redundancy_score": redundancy,
        }


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "clip_length",
        "crop_size",
        "scale",
        "views_per_record",
        "calibration_samples",
        "batch_size",
        "progress_every",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.router_init_std < 0:
        raise ValueError("--router-init-std must be non-negative")
    if args.protect_edge_blocks < 0:
        raise ValueError("--protect-edge-blocks must be non-negative")


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    spec = moe_spec_from_name(args.architecture)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    base_root = args.base_checkpoint.expanduser().resolve()
    teacher_root = args.teacher_checkpoint.expanduser().resolve()
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

    teacher_config = teacher_root / args.transformer_subfolder / "config.json"
    teacher_weights = teacher_root / args.transformer_subfolder / "diffusion_pytorch_model.safetensors"
    if not teacher_config.is_file() or not teacher_weights.is_file():
        raise FileNotFoundError(
            "Teacher checkpoint must contain transformer/config.json and "
            "transformer/diffusion_pytorch_model.safetensors"
        )

    base_dataset = TripletVideoDataset(
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
        base_dataset,
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
    teacher = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(teacher_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    source_shape = validate_d1536_ta(teacher)
    if spec.num_layers > source_shape["num_layers"]:
        raise ValueError("Student cannot have more layers than the D1536 TA source")
    for parameter in reae.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    reae.to(device=device, dtype=dtype).eval()
    teacher.to(device=device, dtype=dtype).eval()
    closure = B2ACompactVelocityDistillationForward(
        reae,
        teacher,
        attention_backend=args.attention_backend,
        gradient_checkpointing=False,
    ).eval()
    closure.reae.eval()

    collector = ActivationImportanceCollector(teacher)
    block_collector = (
        BlockRedundancyCollector(teacher)
        if spec.num_layers < source_shape["num_layers"]
        else None
    )
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
        if block_collector is not None:
            block_collector.close()

    if processed != calibration_limit:
        raise RuntimeError(f"Calibration processed {processed}, expected {calibration_limit}")
    scores = collector.scores()
    block_scores = None if block_collector is None else block_collector.scores()

    selection_spec = B2AWidthSpec(
        hidden_dim=spec.hidden_dim,
        num_heads=spec.num_heads,
        head_dim=spec.head_dim,
        ffn_dim=spec.total_ffn_dim,
        num_layers=source_shape["num_layers"],
        adapter_dim=spec.adapter_dim,
    )
    selected = collector.select(selection_spec)

    if block_scores is None:
        layer_selection = {
            "kept_teacher_blocks": list(range(source_shape["num_layers"])),
            "pruned_teacher_blocks": [],
            "protect_edge_blocks": args.protect_edge_blocks,
        }
    else:
        layer_selection = select_teacher_blocks_by_redundancy(
            block_scores["block_residual_ratio"],
            block_scores["block_cosine_similarity"],
            keep_layers=spec.num_layers,
            protect_edge_blocks=args.protect_edge_blocks,
        )
    teacher_blocks = [int(value) for value in layer_selection["kept_teacher_blocks"]]
    if len(teacher_blocks) != spec.num_layers:
        raise RuntimeError(
            f"Layer selector kept {len(teacher_blocks)} blocks, expected {spec.num_layers}"
        )
    aligned_heads = [selected["heads"][index] for index in teacher_blocks]
    aligned_ffn_scores = torch.index_select(
        scores["ffn_by_block"], 0, torch.tensor(teacher_blocks, dtype=torch.long)
    )

    print(
        f"building {args.architecture} sparse-MoE student on CPU; "
        f"kept teacher blocks={teacher_blocks}",
        flush=True,
    )
    student = build_moe_transformer_from_teacher(teacher, spec)
    transfer = transfer_d1536_to_moe(
        teacher,
        student,
        hidden_indices=selected["hidden"],
        head_indices_by_block=aligned_heads,
        ffn_scores_by_block=aligned_ffn_scores,
        teacher_block_indices=teacher_blocks,
        spec=spec,
        router_seed=args.router_seed,
        router_init_std=args.router_init_std,
    )
    target_shape = transformer_moe_shape(student)
    params = parameter_accounting(student)

    importance_tensors = {
        "hidden_global": scores["hidden_global"].contiguous(),
        "hidden_by_block": scores["hidden_by_block"].contiguous(),
        "head_by_block": scores["head_by_block"].contiguous(),
        "ffn_by_block": scores["ffn_by_block"].contiguous(),
    }
    if block_scores is not None:
        importance_tensors.update(
            {key: value.contiguous() for key, value in block_scores.items()}
        )
    importance_path = output / "activation_importance.safetensors"
    save_file(importance_tensors, str(importance_path))

    student.to(device="cpu", dtype=dtype)
    transformer_dir = output / args.transformer_subfolder
    student.save_pretrained(str(transformer_dir), safe_serialization=True)

    report = {
        "kind": f"swiftvr_b2b_{args.architecture}_moe_init",
        "architecture": args.architecture,
        "method": "d1536_ta_activation_rms_plus_layer_redundancy_dense_to_sparse_moe",
        "design": {
            "shared_expert": "top activation-RMS dense FFN neurons",
            "normal_experts": "next-ranked neurons distributed round-robin",
            "router": "small deterministic near-uniform random projection",
            "layer_selection": (
                "residual_ratio_plus_one_minus_cosine"
                if block_scores is not None
                else "identity_all_teacher_blocks"
            ),
            "token_skipping": False,
        },
        "base_checkpoint": str(base_root),
        "teacher_checkpoint": str(teacher_root),
        "teacher_config_sha256": sha256_file(teacher_config),
        "teacher_weights_sha256": sha256_file(teacher_weights),
        "teacher_shape": source_shape,
        "student_shape": target_shape,
        "parameter_accounting": params,
        "saved_dtype": str(dtype).removeprefix("torch."),
        "router_seed": args.router_seed,
        "router_init_std": args.router_init_std,
        "calibration": {
            "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
            "split": args.split,
            "clip_length": args.clip_length,
            "crop_size": args.crop_size,
            "scale": args.scale,
            "views_per_record": args.views_per_record,
            "view_seed": args.view_seed,
            "horizontal_flip_probability": args.horizontal_flip_probability,
            "vertical_flip_probability": args.vertical_flip_probability,
            "samples": processed,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "importance_stats": {
            "hidden_global": _score_stats(scores["hidden_global"]),
            "head_by_block": _score_stats(scores["head_by_block"]),
            "ffn_by_block": _score_stats(scores["ffn_by_block"]),
            **(
                {
                    "block_residual_ratio": _score_stats(block_scores["block_residual_ratio"]),
                    "block_cosine_similarity": _score_stats(block_scores["block_cosine_similarity"]),
                    "block_redundancy_score": _score_stats(block_scores["block_redundancy_score"]),
                }
                if block_scores is not None
                else {}
            ),
        },
        "selection": {
            "hidden_indices": transfer["hidden_indices"],
            "teacher_block_indices": transfer["teacher_block_indices"],
            "pruned_teacher_blocks": layer_selection["pruned_teacher_blocks"],
            "head_indices_by_block": transfer["head_indices_by_block"],
            "expert_allocations": transfer["expert_allocations"],
            "layer_redundancy": layer_selection,
        },
        "artifacts": {
            "transformer": str(transformer_dir),
            "activation_importance": str(importance_path),
        },
    }
    _write_json(output / "moe_init_report.json", report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "architecture": args.architecture,
                "teacher_shape": source_shape,
                "student_shape": target_shape,
                "teacher_block_indices": transfer["teacher_block_indices"],
                "pruned_teacher_blocks": layer_selection["pruned_teacher_blocks"],
                "parameter_accounting": params,
                "calibration_samples": processed,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
