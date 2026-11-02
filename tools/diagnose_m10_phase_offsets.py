#!/usr/bin/env python3
"""Two reset-origin runs of the canonical runner, aligned by physical frame ID.

No alternate Encoder/DiT/Decoder implementation. Preparation decodes each source
frame once and shares its lossless PNG bytes between offset0 and offset1. Both
runs have equal 4k+1 length. Offset changes context as well as phase: this is NOT
an isolated TGrow causal ablation and does NOT automatically select a teacher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.m10_phase_metrics import (
    aggregate, compare_frames, decoder_phase, high_pass, mean_abs, mse, write_csv, write_json,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="Prepare, run canonical inference twice, then compare.")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--base-checkpoint", type=Path, required=True)
    run.add_argument("--transformer-checkpoint", type=Path, required=True)
    run.add_argument("--transformer-type", choices=("dense", "moe"), default="dense")
    run.add_argument("--decoder-type", choices=("original", "m9a1", "slim"), default="original")
    run.add_argument("--decoder-checkpoint", type=Path)
    run.add_argument("--start-frame", type=int, default=0, help="Zero-based physical source position.")
    run.add_argument("--frames", type=int, default=81, help="Frames per run, 4k+1; needs one more source frame.")
    run.add_argument("--clip-len", type=int, default=24)
    run.add_argument("--upscale", type=int, default=3)
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    run.add_argument("--attention-backend", default="sdpa")
    comp = sub.add_parser("compare", help="Re-analyze already generated outputs; no GPU/model dependencies.")
    comp.add_argument("--root", type=Path, required=True)
    for parser in (run, comp):
        parser.add_argument("--crop", default=None, help="x,y,w,h in unpadded OUTPUT pixels; omit for full frame.")
        parser.add_argument("--comparison-name", default="comparison")
        parser.add_argument("--panel-width", type=int, default=640)
        parser.add_argument("--fps", type=float, default=30.0, help="Playback only, no resampling.")
    return p


def output_frame_map(specs, source_start: int, *, edge_margin: int = 2):
    """Map actual canonical chunk emissions; use only MIDDLE interiors for summary."""
    rows, local = [], 0
    for spec in specs:
        kind = spec.ctype.value
        emitted = spec.frame_count + (3 if kind == "last" else 0) - (3 if spec.is_first_decode else 0)
        for j in range(emitted):
            rows.append({"source_id": source_start + local, "local_frame": local,
                         "phase": decoder_phase(local), "chunk": spec.clip_idx,
                         "chunk_type": kind, "chunk_local_frame": j,
                         "interior": kind == "middle" and edge_margin <= j < emitted - edge_margin})
            local += 1
    return rows


def align_runs(a: list[dict], b: list[dict]) -> list[tuple[dict, dict]]:
    aa, bb = {r["source_id"]: r for r in a}, {r["source_id"]: r for r in b}
    if len(aa) != len(a) or len(bb) != len(b):
        raise ValueError("Duplicate physical frame IDs")
    return [(aa[i], bb[i]) for i in sorted(aa.keys() & bb.keys())]


def _prepare(a, root):
    # Import only when generating data, not for CPU --help/compare/tests.
    from swiftvr.io import _decord_batch_to_torch, _numeric_sort_key, IMAGE_EXTS
    from swiftvr.streaming.chunk import build_chunk_specs

    source = a.input.expanduser().resolve()
    required = a.start_frame + a.frames + 1
    if source.is_dir():
        paths = sorted((p for p in source.iterdir() if p.is_file() and not p.name.startswith(".")
                        and p.suffix.lower() in IMAGE_EXTS), key=_numeric_sort_key)
        total, source_fps = len(paths), None
        def read(index):
            with Image.open(paths[index]) as image:
                return image.convert("RGB").copy()
    else:
        import decord
        reader = decord.VideoReader(str(source))
        total, source_fps = len(reader), float(reader.get_avg_fps())
        def read(index):
            tensor = _decord_batch_to_torch(reader.get_batch([index]))
            return Image.fromarray(tensor[0].cpu().numpy())
    if total < required:
        raise ValueError(f"Need source positions through {required - 1}; only {total} available")
    root.mkdir(parents=True, exist_ok=False)
    originals = root / "source_png"
    originals.mkdir()
    hashes = {}
    for i in range(a.start_frame, required):
        path = originals / f"{i:08d}.png"
        read(i).save(path)
        hashes[str(i)] = hashlib.sha256(path.read_bytes()).hexdigest()
    specs = build_chunk_specs(a.frames, a.clip_len)
    runs = []
    for offset in (0, 1):
        inputs = root / f"input_offset{offset}"
        inputs.mkdir()
        mapping = output_frame_map(specs, a.start_frame + offset)
        if len(mapping) != a.frames:
            raise RuntimeError("Canonical chunk emission count differs from requested length")
        for row in mapping:
            filename = f"{row['source_id']:08d}.png"
            try:
                os.link(originals / filename, inputs / filename)
            except OSError:
                shutil.copyfile(originals / filename, inputs / filename)
            row["filename"] = filename  # canonical runner preserves folder frame names
        runs.append({"offset": offset, "input_dir": inputs.name,
                     "output_dir": f"output_offset{offset}", "frames": mapping})
    info = {"kind": "m10_same_physical_frame_two_origins", "source": str(source),
            "source_fps": source_fps, "start_frame": a.start_frame, "frames_per_run": a.frames,
            "phase_formula": "(local_output_frame + 3) % 4", "source_png_sha256": hashes,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            "runs": runs, "inference_commands": [],
            "note": "Phase AND context/chunk composition change; no motion/quality causal claim."}
    for run in runs:
        command = [sys.executable, str(ROOT / "scripts/inference_custom_components.py"),
                   "--input", str(root / run["input_dir"]), "--output", str(root / run["output_dir"]),
                   "--base-checkpoint", str(a.base_checkpoint.resolve()),
                   "--transformer-checkpoint", str(a.transformer_checkpoint.resolve()),
                   "--transformer-type", a.transformer_type, "--decoder-type", a.decoder_type,
                   "--upscale", str(a.upscale), "--clip-len", str(a.clip_len), "--dit-overlap", "0",
                   "--dtype", a.dtype, "--attention-backend", a.attention_backend,
                   "--device", a.device, "--png"]
        if a.decoder_checkpoint is not None:
            command += ["--decoder-checkpoint", str(a.decoder_checkpoint.resolve())]
        info["inference_commands"].append(command)
    write_json(root / "run.json", info)
    return info


def _load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _crop_box(value, width, height):
    if value is None:
        return 0, 0, width, height
    x, y, w, h = (int(v) for v in value.split(","))
    if x < 0 or y < 0 or w < 3 or h < 3 or x + w > width or y + h > height:
        raise ValueError(f"Invalid output-pixel crop {value} for {width}x{height}")
    return x, y, w, h


def _panel(images, labels, width):
    tiles = []
    for array, label in zip(images, labels):
        image = Image.fromarray(np.rint(array.clip(0, 1) * 255).astype(np.uint8))
        height = max(2, round(width * image.height / image.width))
        height += height % 2
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (width, height + 32), "white")
        tile.paste(image, (0, 32))
        ImageDraw.Draw(tile).text((5, 8), label, fill="black")
        tiles.append(tile)
    canvas = Image.new("RGB", (width * len(tiles), tiles[0].height), "white")
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * width, 0))
    return canvas


def compare(root, *, crop, name, panel_width, fps):
    root = root.expanduser().resolve()
    info = json.loads((root / "run.json").read_text())
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("comparison-name must be a single directory name")
    out = root / name
    if out.exists():
        raise FileExistsError(f"Use a fresh --comparison-name: {out}")
    run0, run1 = info["runs"]
    pairs = align_runs(run0["frames"], run1["frames"])
    if not any(a["interior"] and b["interior"] for a, b in pairs):
        raise ValueError("No shared MIDDLE interior: use at least 81 frames with clip-len24")
    # Refuse missing/extra frames before computing any comparison.
    for run in (run0, run1):
        expected = {r["filename"] for r in run["frames"]}
        actual = {p.name for p in (root / run["output_dir"]).glob("*.png")}
        if actual != expected:
            raise ValueError(f"{run['output_dir']} frame set mismatch: missing={sorted(expected-actual)[:5]}, extra={sorted(actual-expected)[:5]}")
    out.mkdir()
    frames_dir = out / "frames"
    frames_dir.mkdir()
    rows, contact, previous_error = [], [], None
    for a, b in pairs:
        source_id = a["source_id"]
        x0, x1 = [_load_rgb(root / run["output_dir"] / row["filename"])
                  for run, row in ((run0, a), (run1, b))]
        if x0.shape != x1.shape:
            raise ValueError("Offset output geometry differs")
        h, w = x0.shape[:2]
        x, y, cw, ch = _crop_box(crop, w, h)
        x0, x1 = x0[y:y+ch, x:x+cw], x1[y:y+ch, x:x+cw]
        h0, h1 = high_pass(x0), high_pass(x1)
        error = h0 - h1
        interior = bool(a["interior"] and b["interior"])
        temporal = (mean_abs(error - previous_error[1])
                    if previous_error is not None and source_id == previous_error[0] + 1 else None)
        rows.append({"source_id": source_id, "local0": a["local_frame"], "local1": b["local_frame"],
                     "phase0": a["phase"], "phase1": b["phase"], "chunk0": a["chunk"], "chunk1": b["chunk"],
                     "interior": interior, "interior_temporal_pair": interior and previous_error is not None and previous_error[2],
                     **compare_frames(x0, x1, h0, h1), "hf_temporal_mae": temporal,
                     "hf_rms0_not_quality": float(np.sqrt(mse(h0))), "hf_rms1_not_quality": float(np.sqrt(mse(h1)))})
        previous_error = (source_id, error, interior)
        # Bicubic LQ is a visual control, not a high-resolution quality target.
        with Image.open(root / "source_png" / a["filename"]) as im:
            im = im.convert("RGB")
            iw, ih = im.size
            im = im.crop((0, 0, iw // 8 * 8, ih // 8 * 8)).resize((w, h), Image.Resampling.BICUBIC)
            lq = np.asarray(im, dtype=np.float32)[y:y+ch, x:x+cw] / 255.0
        panel = _panel([lq, x0, x1], [f"LQ-up source={source_id}",
                       f"start0 source={source_id} t={a['local_frame']} p={a['phase']}",
                       f"start1 source={source_id} t={b['local_frame']} p={b['phase']}"], panel_width)
        panel.save(frames_dir / f"{source_id:08d}.png")
        if interior and len(contact) < 8:
            contact.append(panel)
    sheet = Image.new("RGB", (contact[0].width, sum(p.height for p in contact)), "white")
    y = 0
    for panel in contact:
        sheet.paste(panel, (0, y)); y += panel.height
    sheet.save(out / "contact_sheet.png")
    interior_rows = [dict(r) for r in rows if r["interior"]]
    for row in interior_rows:
        if not row["interior_temporal_pair"]:
            row["hf_temporal_mae"] = None
    write_csv(out / "per_frame.csv", rows)
    report = {"kind": "same_frame_offset_difference_not_quality", "source": info["source"],
              "crop_output_pixels": crop, "playback_fps": fps, "source_fps": info["source_fps"],
              "all_common": aggregate(rows, "phase0"),
              "middle_interiors": aggregate(interior_rows, "phase0"),
              "middle_interiors_grouped_by_offset1_phase": aggregate(interior_rows, "phase1"),
              "note": "RGB/HF errors compare the SAME physical frame across origins, not vs GT. "
                      "Larger HF energy is not necessarily better texture. Both phase AND context change. "
                      "All-frame CSV includes boundaries; main summary excludes FIRST/LAST and 2 edge frames in both runs.",
              "video_error": None}
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-v", "error", "-framerate", str(fps), "-start_number", str(pairs[0][0]["source_id"]),
                   "-i", str(frames_dir / "%08d.png"), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out / "comparison.mp4")]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            report["video_error"] = str(exc)
    else:
        report["video_error"] = "ffmpeg unavailable; all PNGs retained"
    write_json(out / "report.json", report)
    print(json.dumps(report["middle_interiors"], indent=2), flush=True)
    print(f"Saved: {out}", flush=True)



def main():
    a = build_parser().parse_args()
    if a.panel_width < 64 or a.panel_width % 2 or a.fps <= 0:
        raise ValueError("Use a positive even panel-width >=64 and positive playback fps")
    if a.action == "compare":
        compare(a.root, crop=a.crop, name=a.comparison_name, panel_width=a.panel_width, fps=a.fps)
        return 0
    if a.start_frame < 0 or a.frames % 4 != 1 or a.clip_len <= 0 or a.clip_len % 4 or a.upscale <= 0:
        raise ValueError("Require start>=0, frames=4k+1, clip-len positive multiple of4, upscale>0")
    if a.frames < 3 * a.clip_len + 9:
        raise ValueError("Use at least 3*clip-len+9 frames for two warmed MIDDLE chunks")
    if (a.decoder_type != "original") != (a.decoder_checkpoint is not None):
        raise ValueError("Provide decoder-checkpoint exactly for m9a1/slim")
    for path in (a.base_checkpoint / "reae.safetensors", a.transformer_checkpoint / "transformer/config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if a.decoder_checkpoint is not None:
        for file in ("config.json", "model.safetensors"):
            if not (a.decoder_checkpoint / file).is_file():
                raise FileNotFoundError(a.decoder_checkpoint / file)
    root = a.output_dir.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Use a fresh output-dir: {root}")
    info = _prepare(a, root)
    for command in info["inference_commands"]:
        print("Running canonical component inference:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    compare(root, crop=a.crop, name=a.comparison_name, panel_width=a.panel_width, fps=a.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
