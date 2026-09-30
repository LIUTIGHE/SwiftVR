#!/usr/bin/env python3
"""Compare two canonical chunk traces with identical inputs and components.

No teacher/GT quality claim: errors measure chunk sensitivity, not improvement.
Writes per-latent/per-RGB diagnostics and unresized, lossless ROI comparisons.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors.torch import load_file


def load_trace(root):
    root = Path(root).expanduser().resolve()
    info = json.loads((root / "trace.json").read_text())
    if info.get("kind") != "swiftvr_chunk_trace_v1" or not info.get("complete"):
        raise ValueError(f"Not a completed canonical trace: {root}")
    return root, info


def validate_pair(a, b):
    for key in ("input_sha256", "frames", "latent_frames", "frames_to_trim", "png_files"):
        if a[key] != b[key]:
            raise ValueError(f"Traces differ in {key}; refusing unpaired comparison")
    ignore = {"output", "clip_len", "chunk_trace"}
    aa = {k: v for k, v in a["arguments"].items() if k not in ignore}
    bb = {k: v for k, v in b["arguments"].items() if k not in ignore}
    if aa != bb:
        raise ValueError("Only clip-len/output/trace may differ; input, components and precision must match")


def frame_map(info, stage):
    rows = []
    for chunk in info["chunks"]:
        start, count = int(chunk[f"{stage}_start"]), int(chunk[f"{stage}_count"])
        if start != len(rows):
            raise ValueError("Non-contiguous observed frame offsets")
        for local in range(count):
            rows.append({"chunk": chunk["chunk_index"], "kind": chunk["chunk_type"],
                         "local": local, "edge_distance": min(local, count - 1 - local)})
    return rows


def stats(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite diagnostic values")
    error = a - b
    mse = float(np.mean(error * error, dtype=np.float64))
    power = float(np.mean(a * a, dtype=np.float64))
    return {"mse": mse, "mae": float(np.mean(np.abs(error), dtype=np.float64)),
            "max_abs": float(np.max(np.abs(error))), "relative_l2_to_a": float(np.sqrt(mse / max(power, 1e-12)))}


def highpass(x):
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = x.shape[:2]
    low = sum(p[y:y+h, z:z+w] for y in range(3) for z in range(3)) / 9.0
    return x - low


def mean_rows(rows, keys):
    return {"count": len(rows), **{
        key: float(np.mean([r[key] for r in rows])) if rows else None for key in keys}}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def native_panel(a, b, text_a, text_b):
    h, w = a.shape[:2]
    # No spatial resize. The 32-pixel label bar does not modify the image pixels.
    panel = Image.new("RGB", (2 * w, h + 32), "black")
    for x, image, text in ((0, a, text_a), (w, b, text_b)):
        panel.paste(Image.fromarray(image), (x, 32))
        ImageDraw.Draw(panel).text((x + 5, 8), text, fill="white")
    return panel


def compare(root_a, root_b, output, crop, fps=30.0, no_video=False):
    root_a, a = load_trace(root_a)
    root_b, b = load_trace(root_b)
    validate_pair(a, b)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new comparison directory: {output}")
    maps = {stage: (frame_map(a, stage), frame_map(b, stage)) for stage in ("latent", "output")}
    if len(maps["output"][0]) != a["frames"] or len(maps["latent"][0]) != a["latent_frames"]:
        raise ValueError("Trace count mismatch")
    # Check every expected frame before writing derived results.
    for info in (a, b):
        if not all((Path(info["output"]) / name).is_file() for name in info["png_files"]):
            raise FileNotFoundError("Incomplete RGB output sequence")
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "native_roi_frames"
    frame_dir.mkdir()
    latent_rows = []
    for key in ("z_lq", "z_sr"):
        signals = []
        for root, info in ((root_a, a), (root_b, b)):
            tensors = [load_file(str(root / r["tensor_file"]), device="cpu")[key]
                       for r in info["chunks"]]
            signal = torch.cat(tensors, dim=1)
            if signal.shape[1] != info["latent_frames"]:
                raise ValueError("Stored latent count mismatch")
            signals.append(signal)
        if signals[0].shape != signals[1].shape:
            raise ValueError(f"{key} geometry mismatch")
        for i in range(a["latent_frames"]):
            aa, bb = (s[:, i].float().numpy() for s in signals)
            latent_rows.append({"stage": key, "latent_index": i,
                                "chunk_a": maps["latent"][0][i]["chunk"],
                                "chunk_b": maps["latent"][1][i]["chunk"], **stats(aa, bb)})
        del signals, tensors, signal
    rgb_rows, previous = [], None
    for i, filename in enumerate(a["png_files"]):
        arrays = []
        for info in (a, b):
            with Image.open(Path(info["output"]) / filename) as image:
                arrays.append(np.asarray(image.convert("RGB")))
        if arrays[0].shape != arrays[1].shape:
            raise ValueError("RGB geometry mismatch")
        h, w = arrays[0].shape[:2]
        x, y, cw, ch = crop
        if min(x, y) < 0 or min(cw, ch) < 3 or x + cw > w or y + ch > h:
            raise ValueError(f"Output-pixel ROI {crop} exceeds {w}x{h}")
        arrays = [arr[y:y+ch, x:x+cw] for arr in arrays]
        aa, bb = [arr.astype(np.float32) / 255.0 for arr in arrays]
        ha, hb = highpass(aa), highpass(bb)
        error = ha - hb
        ma, mb = (mapping[i] for mapping in maps["output"])
        phase = (i + a["frames_to_trim"]) % 4
        row = {"output_index": i, "filename": filename, "phase": phase,
               "chunk_a": ma["chunk"], "type_a": ma["kind"], "chunk_b": mb["chunk"], "type_b": mb["kind"],
               "edge_distance_a": ma["edge_distance"], "edge_distance_b": mb["edge_distance"],
               "away_from_edges": ma["edge_distance"] > 2 and mb["edge_distance"] > 2,
               **stats(aa, bb), "hf_mae": float(np.mean(np.abs(error))),
               "hf_energy_a_not_quality": float(np.mean(np.abs(ha))),
               "hf_energy_b_not_quality": float(np.mean(np.abs(hb))),
               "hf_error_change": None if previous is None else float(np.mean(np.abs(error - previous)))}
        rgb_rows.append(row)
        previous = error
        label_a = f"C{a['arguments']['clip_len']} frame={i} phase={phase} {ma['kind']}:{ma['local']}"
        label_b = f"C{b['arguments']['clip_len']} frame={i} phase={phase} {mb['kind']}:{mb['local']}"
        panel = native_panel(*arrays, label_a, label_b)
        panel.save(frame_dir / f"{i:08d}.png")
    write_csv(output / "latent_frames.csv", latent_rows)
    write_csv(output / "rgb_frames.csv", rgb_rows)
    keys = ("mae", "mse", "hf_mae", "hf_energy_a_not_quality", "hf_energy_b_not_quality")
    summary = {
        "kind": "swiftvr_chunk_sensitivity_v1", "trace_a": str(root_a), "trace_b": str(root_b),
        "frames": a["frames"], "latent_frames": a["latent_frames"], "crop_output_pixels": list(crop),
        "clip_lengths": [a["arguments"]["clip_len"], b["arguments"]["clip_len"]],
        "output_chunk_boundaries_a": [r["output_start"] for r in a["chunks"]][1:],
        "output_chunk_boundaries_b": [r["output_start"] for r in b["chunks"]][1:],
        "latent": {k: mean_rows([r for r in latent_rows if r["stage"] == k], ("mae", "mse", "max_abs", "relative_l2_to_a"))
                   for k in ("z_lq", "z_sr")},
        "rgb_roi": mean_rows(rgb_rows, keys),
        "rgb_roi_by_phase": {str(p): mean_rows([r for r in rgb_rows if r["phase"] == p], keys) for p in range(4)},
        "rgb_roi_away_from_edges": mean_rows([r for r in rgb_rows if r["away_from_edges"]], keys),
        "video_written": False,
        "notes": ["Paired chunk-sensitivity diagnostics, NOT GT quality/flicker scores.",
                  "Different temporal context/window geometry can legitimately change DiT predictions.",
                  "Low difference does not rule out a shared 4-phase artifact.",
                  "FIRST/LAST and boundary neighborhoods remain labeled; see CSV before attributing a cause.",
                  "Checkpoint paths/settings are matched; checkpoints must remain immutable.",
                  "Native ROI PNG and FFV1 retain pixel geometry; display at 100% to inspect details."],
    }
    report = output / "summary.json"
    report.write_text(json.dumps(summary, indent=2) + "\n")
    if not no_video:
        log = output / "ffmpeg.log"
        command = ["ffmpeg", "-nostdin", "-n", "-v", "error", "-framerate", str(fps),
                   "-start_number", "0", "-i", str(frame_dir / "%08d.png"),
                   "-frames:v", str(a["frames"]), "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0",
                   str(output / "native_roi.mkv")]
        with log.open("w") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
        summary["video_written"] = True
        report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True, help="First --chunk-trace directory")
    p.add_argument("--b", type=Path, required=True, help="Second --chunk-trace directory")
    p.add_argument("--crop", required=True, help="x,y,w,h in output pixels; never resized")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fps", type=float, default=30.0, help="Playback only; no time remapping")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    crop = tuple(int(v) for v in args.crop.split(","))
    if len(crop) != 4 or args.fps <= 0:
        p.error("Need four crop integers and positive fps")
    compare(args.a, args.b, args.output_dir, crop, args.fps, args.no_video)


if __name__ == "__main__":
    main()
