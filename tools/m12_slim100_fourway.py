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
    if len(names) != count or sorted(int(p.stem) for p in names) != list(range(count)):
        raise ValueError(f"Noncontiguous or incomplete {count}-frame PNG sequence: {source.path}")


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
    for scene in SCENES:
        entry = next(c for c in bank["clips"] if c["name"] == scene)
        # Match the historical baseline's directory contract.
        orig_dir = args.m11b_original_root / scene / "M11B_Original" / "png"
        slim_dir = args.slim_root / scene / "png"
        compose(scene, entry, orig_dir, slim_dir, args.output_root / scene)


if __name__ == "__main__":
    main()
