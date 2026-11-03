#!/usr/bin/env python3
"""M12-1: native-ROI, lossless 4-way evaluation of the existing Slim100 decoder.

Order: Original SwiftVR | AISR/TinyCNN
       M11-B30k + Original ReAE | M11-B30k + Slim100

Only composition. Never runs or modifies the four models. The fixed reference
bank and M11-B+Original output are reused; full-size RGB frames are CROPPED,
never resized. AISR is aligned by historical length-normalized ordinal index,
not PTS. The native FFV1 montage is for visible temporal/texture assessment,
not a paired-GT quality metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from tools.compare_720p3x_outputs import FrameSource
from tools.compose_4way_video import _auto_basic_index

LABELS = (
    "Original SwiftVR",
    "AISR / TinyCNN",
    "M11-B 30k + Original",
    "M11-B 30k + Slim100",
)
SCENES = ("Wetland1", "Mangrove1")
SELECTED = (0, 12, 124, 245, 248)


def _contiguous_pngs(source: FrameSource, count: int) -> None:
    if source.kind != "images":
        raise ValueError(f"Expected native PNG sequence, got {source.path}")
    names = [p for p in source.files if p.suffix.lower() == ".png"]
    if (len(source.files) != count or len(names) != count
            or sorted(int(p.stem) for p in names) != list(range(count))):
        raise ValueError(f"Noncontiguous or incomplete {count}-frame PNG sequence: {source.path}")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_slim(scene: str, entry: dict, args, fingerprint: dict) -> None:
    """Render only a missing Slim100 sequence; never overwrite unverified PNGs."""
    run = args.slim_root / scene
    png = run / "png"
    provenance = run / "provenance.json"
    expected = {
        "kind": "M12-1-Slim100-C24-v1",
        "scene": scene,
        "source_input": entry["input"],
        "checkpoint_fingerprints": fingerprint,
        "inference": {
            "upscale": 3, "clip_len": 24, "dit_overlap": 0,
            "dtype": "bfloat16", "attention_backend": "sdpa", "png": True,
            "decoder_type": "slim", "transformer_type": "moe",
        },
    }
    if run.exists():
        if not provenance.is_file() or json.loads(provenance.read_text()) != expected:
            raise ValueError(
                f"Existing Slim output has incomplete/mismatched provenance: {run}. "
                "Preserve it; choose a fresh --slim-root."
            )
        _contiguous_pngs(FrameSource(png), int(entry["frames"]))
        print(f"[M12] reuse validated Slim100 frames: {png}", flush=True)
        return

    run.mkdir(parents=True)
    cmd = [
        sys.executable, "scripts/inference_custom_components.py",
        "--input", entry["input"]["path"], "--output", str(png),
        "--base-checkpoint", str(args.base_checkpoint),
        "--transformer-checkpoint", str(args.transformer_checkpoint),
        "--transformer-type", "moe",
        "--decoder-type", "slim",
        "--decoder-checkpoint", str(args.slim_checkpoint),
        "--upscale", "3", "--clip-len", "24", "--dit-overlap", "0",
        "--dtype", "bfloat16", "--attention-backend", "sdpa", "--png",
    ]
    print(f"[M12] rendering only M11-B+Slim100 on {scene}", flush=True)
    with (run / "inference.log").open("w") as log:
        result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{scene} Slim100 inference failed; inspect {run/'inference.log'}")
    _contiguous_pngs(FrameSource(png), int(entry["frames"]))
    provenance.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"[M12] Slim100 complete: {png}", flush=True)


def compose(scene: str, entry: dict, original_dir: Path, slim_dir: Path,
            output_dir: Path) -> None:
    count = int(entry["frames"])
    original = FrameSource(Path(entry["original"]))
    aisr = FrameSource(Path(entry["basic"]["path"]))
    m11b_original = FrameSource(original_dir)
    slim = FrameSource(slim_dir)
    sources = (original, aisr, m11b_original, slim)

    for item in (original, m11b_original, slim):
        _contiguous_pngs(item, count)
    if len(aisr) < count:
        raise ValueError(f"AISR has only {len(aisr)} frames, need {count}")
    geometry = (original.width, original.height)
    if any((s.width, s.height) != geometry for s in sources):
        raise ValueError(f"{scene}: four sources have different native output geometry")

    x, y, w, h = map(int, entry["roi"])
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > geometry[0] or y + h > geometry[1]:
        raise ValueError(f"{scene}: ROI outside native {geometry} output")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output_dir}")
    output_dir.mkdir(parents=True)

    fps = FrameSource(Path(entry["input"]["path"])).fps
    width, height = 2 * w, 2 * (h + 28)
    movie = output_dir / "native_fourway.mkv"
    args = [
        "ffmpeg", "-nostdin", "-n", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "ffv1",
        "-level", "3", "-pix_fmt", "bgr0", str(movie),
    ]
    align = []
    with (output_dir / "ffmpeg.log").open("w") as err:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=err)
        try:
            for i in range(count):
                ti = _auto_basic_index(i, basic_count=len(aisr), reference_count=count)
                align.append(ti)
                canvas = Image.new("RGB", (width, height), (0, 0, 0))
                draw = ImageDraw.Draw(canvas)
                for j, (label, source) in enumerate(zip(LABELS, sources)):
                    src_index = ti if j == 1 else i
                    rgb = source.frame(src_index)
                    col, row = j % 2, j // 2
                    xx, yy = col * w, row * (h + 28)
                    canvas.paste(Image.fromarray(rgb[y:y+h, x:x+w]), (xx, yy + 28))
                    text = f"{label} | frame {i} P{(i+3)%4}"
                    if j == 1:
                        text += f" | AISR index {ti}"
                    draw.text((xx + 6, yy + 7), text, fill=(255, 255, 255))
                if i in SELECTED:
                    canvas.save(output_dir / f"native_fourway_{i:06d}_p{(i+3)%4}.png")
                process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
        except BaseException:
            process.stdin.close()
            process.wait()
            raise
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {output_dir/'ffmpeg.log'}")

    summary = {
        "kind": "M12-Slim100-fourway-C24-v1",
        "scene": scene, "frames": count, "fps": fps,
        "source_input": entry["input"],
        "roi": [x, y, w, h], "native_frame_size": list(geometry),
        "order": list(LABELS),
        "sources": [str(s.path) for s in sources],
        "aisr_source_indices": align,
        "aisr_alignment": "length-normalized ordinal; not verified PTS synchronization",
        "phase_rule": "(global_output_frame+3)%4",
        "lossless": "FFV1 8-bit native ROI; RGB frames cropped without resampling",
        "caveat": "No paired HR GT; appearance/flicker comparison only",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[M12] {scene}: {count} frames, {movie}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-bank", type=Path, required=True)
    p.add_argument("--m11b-original-root", type=Path, required=True)
    p.add_argument("--slim-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--slim-checkpoint", type=Path, required=True)
    p.add_argument("--transformer-checkpoint", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    args = p.parse_args()

    bank = json.loads((args.reference_bank / "reference.json").read_text())
    if bank.get("kind") != "m8_task_custom_reference_v1":
        raise ValueError("Wrong fixed Original SwiftVR / AISR reference bank")
    expected = dict(upscale=3, clip_len=24, dit_overlap=0,
                    dtype="bfloat16", attention_backend="sdpa")
    if bank.get("inference") != expected:
        raise ValueError("Reference bank was not rendered using matched C24 settings")
    if {c["name"] for c in bank["clips"]} != set(SCENES):
        raise ValueError("Expected both original Wetland1 and Mangrove1 references")
    for record in bank["original_files"] + [
        c[k] for c in bank["clips"] for k in ("input", "basic")
    ]:
        path = Path(record["path"])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError(f"Frozen reference changed: {path}")
    files = {
        "reae": args.base_checkpoint / "reae.safetensors",
        "transformer": args.transformer_checkpoint / "transformer/diffusion_pytorch_model.safetensors",
        "transformer_config": args.transformer_checkpoint / "transformer/config.json",
        "slim_decoder": args.slim_checkpoint / "model.safetensors",
        "slim_config": args.slim_checkpoint / "config.json",
        "inference_script": ROOT / "scripts/inference_custom_components.py",
    }
    fingerprint = {
        name: {"path": str(path.resolve()), "sha256": _hash_file(path)}
        for name, path in files.items()
    }
    if args.output_root.resolve() == args.slim_root.resolve():
        raise ValueError("Comparison output must differ from Slim inference root")

    # Fail BEFORE expensive inference if either legacy baseline is absent.
    entries = {c["name"]: c for c in bank["clips"]}
    for scene in args.scenes:
        orig_dir = args.m11b_original_root / scene / "M11B_Original" / "png"
        _contiguous_pngs(FrameSource(orig_dir), int(entries[scene]["frames"]))

    for scene in args.scenes:
        entry = entries[scene]
        ensure_slim(scene, entry, args, fingerprint)
        orig_dir = args.m11b_original_root / scene / "M11B_Original" / "png"
        slim_dir = args.slim_root / scene / "png"
        output = args.output_root / scene
        if (output / "summary.json").is_file():
            existing = json.loads((output / "summary.json").read_text())
            if (existing.get("kind"), existing.get("scene")) != (
                "M12-Slim100-fourway-C24-v1", scene
            ):
                raise ValueError(f"Existing comparison does not match M12 scene: {output}")
            print(f"[M12] comparison already complete: {output}", flush=True)
        else:
            compose(scene, entry, orig_dir, slim_dir, output)


if __name__ == "__main__":
    main()
