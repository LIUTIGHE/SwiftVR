#!/usr/bin/env python3
"""Build a cheap contact sheet of ORIGINAL video frame indices to pick motion windows.

This is a visual locator, not an optical-flow score or a model-quality metric.
The displayed numbers are zero-based decoded frame indices; time labels are
informational only. The chunk diagnostic selects by ordinal frame number.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import decord
import numpy as np
from PIL import Image, ImageDraw


def select_indices(total: int, start: int, stop: int, count: int) -> list[int]:
    if total <= 0 or not 0 <= start < total:
        raise ValueError("invalid input length or start-frame")
    if not start < stop <= total:
        raise ValueError("end-frame must be greater than start-frame and at most video length")
    if count <= 0:
        raise ValueError("max-thumbs must be positive")
    n = min(stop - start, count)
    return [int(x) for x in np.linspace(start, stop - 1, n, dtype=np.int64)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None, help="Exclusive original frame index.")
    p.add_argument("--max-thumbs", type=int, default=24)
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--thumb-width", type=int, default=320)
    args = p.parse_args()
    if args.columns <= 0 or args.thumb_width <= 0:
        p.error("columns and thumb-width must be positive")

    source = args.input.expanduser().resolve()
    reader = decord.VideoReader(str(source), ctx=decord.cpu(0))
    total = len(reader)
    stop = total if args.end_frame is None else args.end_frame
    indices = select_indices(total, args.start_frame, stop, args.max_thumbs)
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        fps = 0.0

    # All tiles use the same dimensions; no inference and no full-res copies.
    first = Image.fromarray(reader[indices[0]].asnumpy())
    height = max(1, round(first.height * args.thumb_width / first.width))
    title_h = 30
    rows = math.ceil(len(indices) / args.columns)
    sheet = Image.new(
        "RGB",
        (args.columns * args.thumb_width, rows * (height + title_h)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for i, index in enumerate(indices):
        frame = Image.fromarray(reader[index].asnumpy()).convert("RGB")
        thumb = frame.resize((args.thumb_width, height), Image.Resampling.BILINEAR)
        x = (i % args.columns) * args.thumb_width
        y = (i // args.columns) * (height + title_h)
        sheet.paste(thumb, (x, y + title_h))
        timestamp = f" ~{index / fps:.1f}s" if fps else ""
        draw.rectangle((x, y, x + args.thumb_width, y + title_h - 1), fill="black")
        draw.text((x + 7, y + 8), f"source frame {index}{timestamp}", fill="white")

    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    print(f"Video: {source}")
    print(f"Total decoded frames: {total}; metadata fps: {fps:g}")
    print(f"Indices displayed: {indices}")
    print(f"Contact sheet: {target}")
    print("Choose START_FRAME about 48-64 source frames BEFORE sustained motion,")
    print("so the FIRST chunks have warm-up context before the motion ROI is evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
