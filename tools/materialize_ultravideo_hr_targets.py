#!/usr/bin/env python3
"""Add lossless paired HR targets to EXISTING LR views; never regenerate LR/cache.

Per-clip bundles avoid one-file-per-frame I/O. Shards can run independently.
Only a complete paired manifest with a hash is published at successful finish.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lr-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--decode-batch-size", type=int, default=4)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-clips", type=int, default=0, help="Smoke subset only")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--qa-views", type=int, default=4, help="Reproduce LR center pixels and export paired contact sheets")
    p.add_argument("--resume", action="store_true", help="Resume identical HR plan; validate completed bundle metadata")
    args = p.parse_args()
    if min(args.decode_batch_size, args.shard_count, args.progress_every) <= 0 or not 0 <= args.shard_index < args.shard_count or args.max_clips < 0 or args.qa_views < 0:
        p.error("Invalid batch/shard/progress settings")
    import numpy as np
    import torch
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import save_file
    from swiftvr.data.task_pairs import read_jsonl, sha256_file, lr_row_hash, clean_hr_crop
    from tools.materialize_ultravideo_lr_bundles import _decord_reader, _batch_to_numpy, _write_json, _write_jsonl, _degrade_hq, _stable_seed

    source = args.lr_manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Use a fresh HR output directory or --resume: {output}")
    rows = read_jsonl(source)
    lr_summary = json.loads((source.parent / "summary.json").read_text())
    seed = lr_summary["seed"]
    qa_indices = {int(v["materialized_index"]) for v in rows[:args.qa_views]}
    qa_results = []
    groups = defaultdict(list)
    for row in rows:
        required = {"canonical_hr_width", "canonical_hr_height", "crop_box_hq", "frame_indices", "raw_video", "materialized_index"}
        if not required.issubset(row):
            raise ValueError(f"Use full materialization metadata; missing {required - set(row)}")
        if not Path(row["bundle_path"]).is_file():
            raise FileNotFoundError(row["bundle_path"])
        groups[row["record_uid"]].append(row)
    selected = list(groups.items())[args.shard_index::args.shard_count]
    if args.max_clips:
        selected = selected[:args.max_clips]
    if not selected:
        raise ValueError("No clips selected")
    output.mkdir(parents=True, exist_ok=True)
    plan = {"source_sha256": sha256_file(source), "shard_count": args.shard_count,
            "shard_index": args.shard_index, "max_clips": args.max_clips}
    plan_file = output / "plan.json"
    if plan_file.exists():
        if json.loads(plan_file.read_text()) != plan:
            raise ValueError("Cannot resume a different HR plan")
    elif args.resume and any(output.iterdir()):
        raise ValueError("Nonempty HR output is missing its plan")
    _write_json(plan_file, plan)
    result, source_hashes, total_bytes = [], {}, 0
    start = time.perf_counter()
    for clip_n, (uid, views) in enumerate(selected, 1):
        raw_path = Path(views[0]["raw_video"]).resolve()
        if any(Path(v["raw_video"]).resolve() != raw_path for v in views):
            raise ValueError("One clip group contains different raw sources")
        source_hashes[str(raw_path)] = sha256_file(raw_path)
        bundle = output / f"clip_{int(views[0]['materialized_index']):08d}.safetensors"
        bundle_meta = {"kind": "ultravideo_hr_targets_v1",
                       "raw_sha256": source_hashes[str(raw_path)],
                       "view_hashes": json.dumps([lr_row_hash(v) for v in views])}
        if args.resume and bundle.exists():
            with safe_open(str(bundle), framework="pt", device="cpu") as cached:
                if cached.metadata() != bundle_meta or len(cached.keys()) != len(views):
                    raise ValueError(f"Existing HR bundle identity mismatch: {bundle}")
                for view in views:
                    key = f"hr_{int(view['materialized_index']):08d}"
                    tensor = cached.get_tensor(key)
                    expected = (len(view["frame_indices"]), 3, int(view["target_height"]), int(view["target_width"]))
                    if tensor.dtype != torch.uint8 or tuple(tensor.shape) != expected:
                        raise ValueError("Existing HR bundle geometry mismatch")
                    result.append({**view, "hr_bundle_path": str(bundle), "hr_bundle_key": key,
                                   "lr_row_sha256": lr_row_hash(view)})
            total_bytes += bundle.stat().st_size
            continue
        reader = _decord_reader(str(raw_path))
        positions = sorted({int(i) for v in views for i in v["frame_indices"]})
        if positions[-1] >= len(reader) or positions[0] < 0:
            raise ValueError("Selected raw frame index out of range")
        frames_by_view = {int(v["materialized_index"]): {} for v in views}
        at_position = defaultdict(list)
        for view in views:
            for pos in view["frame_indices"]:
                at_position[int(pos)].append(view)
        for offset in range(0, len(positions), args.decode_batch_size):
            batch_pos = positions[offset:offset+args.decode_batch_size]
            frames = _batch_to_numpy(reader.get_batch(batch_pos))
            for pos, raw in zip(batch_pos, frames):
                for view in at_position[pos]:
                    hr = clean_hr_crop(raw, view)
                    index = int(view["materialized_index"])
                    if index in qa_indices and pos == view["frame_indices"][len(view["frame_indices"])//2]:
                        hq = hr.resize((int(view["crop_size"]), int(view["crop_size"])), Image.Resampling.BOX)
                        reconstructed = _degrade_hq(hq, view["degradation"], noise_seed=_stable_seed(seed, view["sample_id"], view["view_index"], pos, "noise"))
                        clean = hr
                        if view["horizontal_flip"]:
                            reconstructed = reconstructed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                            clean = clean.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        if view["vertical_flip"]:
                            reconstructed = reconstructed.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                            clean = clean.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                        with safe_open(view["bundle_path"], framework="pt", device="cpu") as lr_handle:
                            actual = lr_handle.get_tensor(view["bundle_key"])[len(view["frame_indices"])//2].permute(1,2,0).numpy()
                        if not np.array_equal(np.asarray(reconstructed), actual):
                            raise ValueError(f"HR/LR reproduction mismatch for index {index}; do not train")
                        sheet = Image.new("RGB", (2*clean.width, clean.height))
                        sheet.paste(Image.fromarray(actual).resize(clean.size, Image.Resampling.BICUBIC), (0,0))
                        sheet.paste(clean, (clean.width,0))
                        sheet.save(output / f"pair_{index:08d}_LQ_GT.png")
                        qa_results.append({"index":index, "source_frame":pos, "lr_exact_match":True})
                    if view["horizontal_flip"]:
                        hr = hr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    if view["vertical_flip"]:
                        hr = hr.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    frames_by_view[int(view["materialized_index"])][pos] = torch.from_numpy(np.array(hr, copy=True)).permute(2, 0, 1).contiguous()
        del reader
        tensors = {}
        bundle = output / f"clip_{int(views[0]['materialized_index']):08d}.safetensors"
        for view in views:
            index = int(view["materialized_index"])
            key = f"hr_{index:08d}"
            tensors[key] = torch.stack([frames_by_view[index][int(pos)] for pos in view["frame_indices"]])
            expected = (len(view["frame_indices"]), 3, int(view["target_height"]), int(view["target_width"]))
            if tuple(tensors[key].shape) != expected:
                raise ValueError("HR tensor geometry mismatch")
            result.append({**view, "hr_bundle_path": str(bundle), "hr_bundle_key": key,
                           "lr_row_sha256": lr_row_hash(view)})
        tmp = bundle.with_suffix(".tmp")
        save_file(tensors, str(tmp), metadata=bundle_meta)
        tmp.replace(bundle)
        total_bytes += bundle.stat().st_size
        if clip_n % args.progress_every == 0 or clip_n == len(selected):
            print(f"HR {clip_n}/{len(selected)} clips, {len(result)} views, {total_bytes/2**30:.2f} GiB", flush=True)
    result.sort(key=lambda v: int(v["materialized_index"]))
    manifest = output / "paired_views.jsonl"
    _write_jsonl(manifest, result)
    meta = {"kind": "ultravideo_hr_targets_v1", "source_manifest": str(source),
            "source_manifest_sha256": sha256_file(source), "source_view_count": len(rows),
            "source_complete": len(rows) == int(lr_summary.get("input_view_count", -1)),
            "paired_manifest_sha256": sha256_file(manifest),
            "shard_count": args.shard_count, "shard_index": args.shard_index,
            "views": len(result), "clips": len(selected), "bundle_bytes": total_bytes,
            "smoke_subset": bool(args.max_clips), "raw_source_sha256": source_hashes,
            "seconds": time.perf_counter()-start,
            "hr_definition": "canonical clean HR before BOX downsample; exact LR frame/crop/flip identity",
            "lr_unchanged": True}
    _write_json(output / "hr_metadata.json", meta)
    _write_json(output / "qa_report.json", {"center_views_checked_this_invocation":qa_results, "note":"On resume, already completed clips are not re-decoded; prior PNGs remain."})
    print(json.dumps({k:v for k,v in meta.items() if k != "raw_source_sha256"}, indent=2))

if __name__ == "__main__":
    main()
