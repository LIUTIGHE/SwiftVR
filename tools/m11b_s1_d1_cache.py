#!/usr/bin/env python3
"""M11B-S1-D1: immutable M11-B restored-latent cache for decoder-only training.

One domain per run. Reuse canonical LR/HR paired views, without duplicating HR.
This tool does NOT train the decoder or infer a full-resolution custom video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sample_file_name(index: int, pair_id: str) -> str:
    key = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:16]
    return f"samples/{index:08d}_{key}.safetensors"


def cache_plan(args, source_files: list[Path], *, total: int, selected: int) -> dict:
    if selected <= 0 or total <= 0 or selected > total:
        raise ValueError("Invalid sample counts")
    return {
        "kind": "M11B-S1-D1-restored-latent-plan-v1",
        "phase": "M11B-S1-D1",
        "domain": args.domain,
        "split": args.legacy_split if args.domain == "legacy" else "train",
        "base_checkpoint": str(args.base_checkpoint.expanduser().resolve()),
        "source_checkpoint": str(args.source_checkpoint.expanduser().resolve()),
        "clip_length_expected": 13,
        "scale_expected": 3,
        "source_files": {str(p): sha256(p) for p in sorted(set(source_files))},
        "dataset_count": total,
        "selected_count": selected,
        "runtime_dtype": args.dtype,
        "storage_dtype": "float16",
        "attention_backend": args.attention_backend,
        "note": "GT is bound through canonical pair_id and source metadata; not duplicated in this cache",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=("legacy", "ultra"), required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--legacy-view-cache", type=Path)
    p.add_argument("--legacy-split", choices=("train", "val"), default="train")
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--ultra-pairs", type=Path, action="append", default=[])
    p.add_argument("--allow-partial-ultra", action="store_true",
                   help="Only for smoke, not a formal full cache")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0, help="0=all; >0 is a pilot prefix")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--resume", action="store_true",
                   help="Only continue a prior run with an identical plan and verified per-sample files")
    p.add_argument("--progress-every", type=int, default=50)
    return p


def validate_args(args) -> None:
    if args.max_samples < 0 or min(args.batch_size, args.progress_every) < 1 or args.num_workers < 0:
        raise ValueError("Invalid sample/worker/batch/progress arguments")
    if args.domain == "legacy":
        if args.legacy_view_cache is None or args.ultra_pairs or args.allow_partial_ultra:
            raise ValueError("legacy requires only --legacy-view-cache")
    elif not args.ultra_pairs or args.legacy_view_cache is not None:
        raise ValueError("ultra requires --ultra-pairs and no legacy cache")
    if args.allow_partial_ultra and not args.max_samples:
        raise ValueError("Partial UltraVideo is only permitted with a limited smoke --max-samples")


def _source_files(args) -> list[Path]:
    base = args.base_checkpoint.expanduser().resolve() / "reae.safetensors"
    model = args.source_checkpoint.expanduser().resolve() / "transformer"
    model_files = [model / "config.json", model / "diffusion_pytorch_model.safetensors"]
    files = [base, *model_files]
    if args.domain == "legacy":
        files.append(args.legacy_view_cache.expanduser().resolve() / "metadata.json")
    else:
        for raw in args.ultra_pairs:
            p = raw.expanduser().resolve()
            files.extend([p, p.parent / "hr_metadata.json"])
    for p in files:
        if not p.is_file():
            raise FileNotFoundError(p)
    return files


def _dataset(args):
    from swiftvr.data.task_pairs import build_legacy_pairs, build_ultra_pairs
    if args.domain == "legacy":
        dataset, meta = build_legacy_pairs(
            args.legacy_view_cache, path_root=args.path_root, split=args.legacy_split)
        meta_info = {
            "legacy_view_cache": str(args.legacy_view_cache.expanduser().resolve()),
            "clip_length": int(meta["clip_length"]),
            "crop_size": int(meta["crop_size"]),
            "scale": int(meta["scale"]),
        }
    else:
        dataset, metas = build_ultra_pairs(
            args.ultra_pairs, allow_partial=args.allow_partial_ultra)
        meta_info = {
            "ultra_pairs": [str(p.expanduser().resolve()) for p in args.ultra_pairs],
            "shards": len(metas),
            "clip_length": int(metas[0].get("clip_length", 13)),
            "scale": 3,
        }
    return dataset, meta_info


def _verify_cached(path: Path, expected_shape: tuple[int, ...] | None = None) -> tuple[int, ...]:
    import torch
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if set(f.keys()) != {"z_sr"}:
            raise ValueError(f"Invalid cache tensor keys: {path}")
        tensor = f.get_tensor("z_sr")
    if tensor.dtype != torch.float16 or tensor.ndim != 4 or int(tensor.shape[1]) != 48:
        raise ValueError(f"Invalid cache dtype/shape: {path}, {tensor.dtype}, {tuple(tensor.shape)}")
    shape = tuple(int(v) for v in tensor.shape)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"Cached geometry mismatch: {path}, got {shape}, expected {expected_shape}")
    return shape


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.dtype == "float16" and args.device == "cpu":
        raise ValueError("Do not use CPU float16 for this model")
    import torch
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader, Subset

    from swiftvr.models import ReAE
    from swiftvr.models.transformer_prompt_free_no_time_moe import (
        WanTransformer3DModelPromptFreeNoTimeMoE,
    )
    from swiftvr.training.forward import encode_reae_clip, prepare_training_batch
    from swiftvr.training.reference import extract_transformer_sample

    files = _source_files(args)
    dataset, metadata = _dataset(args)
    count = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    if not count:
        raise ValueError("Dataset is empty")
    if metadata.get("clip_length", 13) != 13 or metadata.get("scale", 3) != 3:
        raise ValueError("D1 currently requires 13-frame, 3x training views")
    plan = cache_plan(args, files, total=len(dataset), selected=count)
    plan["data"] = metadata
    plan["pilot"] = bool(args.max_samples)
    plan["partial_ultra"] = bool(args.allow_partial_ultra)
    output = args.output_dir.expanduser().resolve()
    plan_file = output / "plan.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
        if not plan_file.is_file() or json.loads(plan_file.read_text()) != plan:
            raise ValueError("Resume plan mismatch; use the exact original inputs and options")
    else:
        if args.resume:
            raise FileNotFoundError(f"No existing cache to resume: {output}")
        (output / "samples").mkdir(parents=True)
        _atomic_json(plan_file, plan)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable on chosen device")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    reae = ReAE(str(args.base_checkpoint / "reae.safetensors")).to(device=device, dtype=dtype).eval()
    transformer = WanTransformer3DModelPromptFreeNoTimeMoE.from_pretrained(
        str(args.source_checkpoint), subfolder="transformer",
        torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
    ).to(device=device, dtype=dtype).eval()
    reae.requires_grad_(False)
    transformer.requires_grad_(False)
    transformer.prepare_for_inference(
        attention_backend=args.attention_backend, use_torch_compile=False)
    workers = {"num_workers": args.num_workers}
    if args.num_workers:
        workers.update({"prefetch_factor": 2, "persistent_workers": True})
    loader = DataLoader(Subset(dataset, list(range(count))), batch_size=args.batch_size,
                        shuffle=False, drop_last=False, pin_memory=device.type=="cuda",
                        **workers)
    records = []
    begin = time.monotonic()
    global_index = 0
    with torch.inference_mode():
        for batch in loader:
            lr = batch["lr"]
            hr = batch["hr"]
            pair_ids = list(batch["pair_id"])
            if lr.ndim != 5 or hr.ndim != 5 or lr.shape[1:3] != (13, 3):
                raise ValueError("Expected paired [B,13,3,H,W] inputs")
            if hr.shape[1:3] != (13, 3) or hr.shape[-2:] != (lr.shape[-2]*3, lr.shape[-1]*3):
                raise ValueError("LR/GT shape or scale mismatch")
            if not bool(torch.isfinite(lr).all()) or not bool(torch.isfinite(hr).all()):
                raise ValueError("Nonfinite LR/GT")
            if hr.shape[-2] % 16 or hr.shape[-1] % 16:
                raise ValueError("GT spatial dimensions must be divisible by ReAE total stride 16")
            expected_latent_shape = (4, 48, int(hr.shape[-2]) // 16, int(hr.shape[-1]) // 16)
            pending = []
            for j, pair_id in enumerate(pair_ids):
                relative_file = sample_file_name(global_index+j, pair_id)
                out_file = output / relative_file
                record = {
                    "cache_index": global_index+j, "dataset_index": global_index+j,
                    "pair_id": pair_id, "domain": args.domain, "file": relative_file,
                    "lr_shape": list(lr[j].shape), "gt_shape": list(hr[j].shape),
                    "phase_formula": "(output_index+3)%4",
                }
                if out_file.is_file():
                    if not args.resume:
                        raise FileExistsError(out_file)
                    record["z_sr_shape"] = list(_verify_cached(out_file, expected_shape=expected_latent_shape))
                else:
                    pending.append(j)
                records.append(record)
            if pending:
                selected = lr[pending].to(device=device, dtype=dtype, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=dtype,
                                    enabled=device.type=="cuda" and dtype != torch.float32):
                    # SwiftVR encodes bilinearly upscaled LR at HR output geometry.
                    # Encoding native LR would produce an incorrectly sized latent grid.
                    prepared = prepare_training_batch(
                        {"lr": selected, "hr": hr[pending]}, auxiliary_key=None
                    )
                    z_lq = encode_reae_clip(
                        reae, prepared["lq_input"], require_4k_plus_1=True
                    )
                    z_lq = z_lq.permute(0, 2, 1, 3, 4).contiguous()
                    velocity = extract_transformer_sample(
                        transformer(z_lq, return_dict=True))
                    if velocity.shape != z_lq.shape:
                        raise RuntimeError("Transformer velocity geometry mismatch")
                    z_sr = (z_lq - velocity).permute(0, 2, 1, 3, 4).contiguous()
                for k, j in enumerate(pending):
                    index = global_index + j
                    latent = z_sr[k].detach().to("cpu", dtype=torch.float16).contiguous()
                    if tuple(latent.shape) != expected_latent_shape or not bool(torch.isfinite(latent).all()):
                        raise ValueError(f"Invalid latent geometry or values: sample {index}")
                    out_file = output / records[index]["file"]
                    tmp = out_file.with_name(out_file.stem + ".tmp.safetensors")
                    if tmp.exists():
                        tmp.unlink()  # Incomplete prior write; final filename is atomic.
                    save_file({"z_sr": latent}, str(tmp))
                    os.replace(tmp, out_file)
                    records[index]["z_sr_shape"] = list(latent.shape)
            global_index += len(pair_ids)
            if global_index == count or global_index % args.progress_every < len(pair_ids):
                print(f"[M11B-S1-D1] {args.domain}: {global_index}/{count} samples "
                      f"elapsed={time.monotonic()-begin:.1f}s", flush=True)
    if len(records) != count:
        raise RuntimeError("Dataset/cache sample count mismatch")
    completed = {
        **plan, "kind": "M11B-S1-D1-restored-latent-v1",
        "complete": True,
        "sample_count": len(records), "samples": records,
        "duration_seconds": time.monotonic()-begin,
        "notes": [
            "Frozen M11-B 30k restored latents, not teacher velocities.",
            "GT HR remains in paired source dataset; cache does not duplicate it.",
            "Pair identity and source file hashes must be revalidated by S1-T1.",
            "Clip T=13 is whole-clip training view, not a deployment C24 streaming sequence.",
        ],
    }
    _atomic_json(output / "metadata.json", completed)
    print(json.dumps({"status": "cache_complete", "output": str(output),
                      "domain": args.domain, "count": count, "pilot": bool(args.max_samples)},
                     indent=2))


if __name__ == "__main__":
    main()
