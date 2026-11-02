#!/usr/bin/env python3
"""Build the B2B D768/F4080 tiny DiT initialization from a trained B2-A DiT.

This is progressive *initialization* only.  The eventual B2B distillation target
remains the stronger Stage-A 200k teacher cache.  We calibrate activation RMS on
frozen B2-A, select a coherent D768 / 6-head / F4080 subnetwork, and save a normal
prompt-free/no-time Transformer checkpoint.  No decoder is involved here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.smoke_training_forward import move_video_batch
from swiftvr.data import TripletVideoDataset
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
from swiftvr.training.b2a_width import (
    ActivationImportanceCollector,
    B2ACompactVelocityDistillationForward,
    build_compact_transformer_from_teacher,
    transfer_structured_width,
    transformer_width_shape,
    validate_b2a_teacher_shape,
)
from swiftvr.training.b2b_joint import B2B_TINY_SPEC, b2b_compute_budget
from swiftvr.training.distillation import DeterministicTripletViewDataset


EXPECTED_B2A_SOURCE = {
    "hidden_dim": 1536,
    "num_heads": 12,
    "head_dim": 128,
    "ffn_dim": 8960,
    "num_layers": 30,
    "adapter_dim": 128,
}
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--b2a-checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
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
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--transformer-subfolder", default="transformer")
    p.add_argument("--progress-every", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    return p


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _slice_batch(batch: Mapping[str, object], count: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value[:count]
        elif isinstance(value, (list, tuple)):
            result[key] = value[:count]
        else:
            result[key] = value
    return result


def main() -> int:
    args = build_parser().parse_args()
    if args.calibration_samples <= 0 or args.batch_size <= 0:
        raise ValueError("calibration-samples and batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = DTYPES[args.dtype]
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")

    base_root = args.base_checkpoint.expanduser().resolve()
    source_root = args.b2a_checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

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
        path_root=args.path_root.expanduser().resolve(),
        verify_paths=args.verify_paths,
    )
    views = DeterministicTripletViewDataset(
        dataset,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
    )
    limit = min(len(views), int(args.calibration_samples))
    loader = DataLoader(
        views,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.num_workers > 0),
    )

    reae = ReAE(str(base_root / args.reae_filename)).to(device=device, dtype=dtype).eval()
    source = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(source_root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype).eval()
    source_shape = transformer_width_shape(source)
    if source_shape != EXPECTED_B2A_SOURCE:
        raise ValueError(f"Expected trained B2-A source {EXPECTED_B2A_SOURCE}, got {source_shape}")
    validate_b2a_teacher_shape(source, B2B_TINY_SPEC)
    for module in (reae, source):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    closure = B2ACompactVelocityDistillationForward(
        reae,
        source,
        attention_backend=args.attention_backend,
        gradient_checkpointing=False,
    ).eval()
    collector = ActivationImportanceCollector(source)
    processed = 0
    started = time.perf_counter()
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    try:
        with torch.inference_mode():
            for batch_cpu in loader:
                if processed >= limit:
                    break
                frame_indices = batch_cpu.get("frame_indices")
                if not isinstance(frame_indices, torch.Tensor):
                    raise TypeError("Calibration batch is missing frame_indices")
                batch_size = int(frame_indices.shape[0])
                if processed + batch_size > limit:
                    batch_size = limit - processed
                    batch_cpu = _slice_batch(batch_cpu, batch_size)
                batch = move_video_batch(batch_cpu, device=device, dtype=dtype)
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype if autocast_enabled else torch.float32,
                    enabled=device.type == "cuda" and autocast_enabled,
                ):
                    closure(batch)
                processed += batch_size
                if processed == limit or processed % args.progress_every == 0:
                    print(
                        f"B2B calibration {processed}/{limit} "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
    finally:
        collector.close()

    if processed != limit:
        raise RuntimeError(f"Processed {processed} calibration samples, expected {limit}")
    scores = collector.scores()
    selected = collector.select(B2B_TINY_SPEC)

    print("building B2B D768 compact Transformer on CPU...", flush=True)
    student = build_compact_transformer_from_teacher(source, B2B_TINY_SPEC)
    transfer = transfer_structured_width(
        source,
        student,
        hidden_indices=selected["hidden"],
        head_indices_by_block=selected["heads"],
        ffn_indices_by_block=selected["ffn"],
        spec=B2B_TINY_SPEC,
    )
    target_shape = transformer_width_shape(student)

    save_file(
        {
            "hidden_global": scores["hidden_global"].contiguous(),
            "hidden_by_block": scores["hidden_by_block"].contiguous(),
            "head_by_block": scores["head_by_block"].contiguous(),
            "ffn_by_block": scores["ffn_by_block"].contiguous(),
        },
        str(output / "activation_importance.safetensors"),
    )
    transformer_dir = output / args.transformer_subfolder
    student.to(device="cpu", dtype=dtype)
    student.save_pretrained(str(transformer_dir), safe_serialization=True)

    report = {
        "kind": "swiftvr_b2b_tiny_dit_structured_init",
        "method": "progressive_activation_rms_structured_width_pruning_from_b2a",
        "base_checkpoint": str(base_root),
        "b2a_source_checkpoint": str(source_root),
        "source_shape": source_shape,
        "student_shape": target_shape,
        "student_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "saved_dtype": args.dtype,
        "calibration_samples": processed,
        "calibration_manifests": [str(path.expanduser().resolve()) for path in args.manifest],
        "selection": transfer,
        "canonical_compute": b2b_compute_budget(),
        "note": "Initialization source is B2-A; B2B training target remains Stage-A 200k.",
    }
    _write_json(output / "metadata.json", report)

    budget = report["canonical_compute"]
    print("================ B2B tiny DiT init ================")
    print(f"Source shape                  : {source_shape}")
    print(f"Student shape                 : {target_shape}")
    print(f"Student parameters            : {report['student_parameters']:,}")
    print(f"Tiny DiT GMAC/frame           : {budget['dit_gmac_per_frame']:.6f}")
    print(f"DiT + extreme decoder GMAC/f  : {budget['dit_plus_decoder_gmac_per_frame']:.6f}")
    print(f"Headroom to 210 GMAC          : {budget['headroom_to_210_gmac']:.6f}")
    print(f"Saved                         : {output}")
    print("===================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
