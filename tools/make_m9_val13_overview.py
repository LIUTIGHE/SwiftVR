#!/usr/bin/env python3
"""Build a single val13 overview sheet from visualize_m9_val13_components output."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


METHODS = (
    ("GT", "gt"),
    ("LQ-up", "lq_up"),
    ("StageA+Orig", "stagea_orig"),
    ("M8A+Orig", "m8a_orig"),
    ("M8A+M9A1", "m8a_m9a1"),
    ("M8A+D76", "m8a_d76"),
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--frame", type=int, default=6)
    p.add_argument("--panel-size", type=int, default=192)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    samples = sorted(path for path in root.glob("sample_*") if path.is_dir())
    if len(samples) != 13:
        raise ValueError(f"Expected 13 sample directories under {root}, got {len(samples)}")

    methods = []
    for label, subdir in METHODS:
        probe = samples[0] / subdir / f"{args.frame:05d}.png"
        if probe.is_file():
            methods.append((label, subdir))
    if not methods:
        raise FileNotFoundError("No method PNGs found in val13 output")

    band = 22
    row_label = 58
    width = row_label + args.panel_size * len(methods)
    height = band + args.panel_size * len(samples)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for col, (label, _) in enumerate(methods):
        draw.text((row_label + col * args.panel_size + 4, 4), label, fill="black")

    for row, sample in enumerate(samples):
        y = band + row * args.panel_size
        draw.text((4, y + 4), sample.name, fill="black")
        for col, (_, subdir) in enumerate(methods):
            path = sample / subdir / f"{args.frame:05d}.png"
            if not path.is_file():
                continue
            image = Image.open(path).convert("RGB")
            image = image.resize((args.panel_size, args.panel_size), Image.Resampling.LANCZOS)
            sheet.paste(image, (row_label + col * args.panel_size, y))

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / f"overview_frame_{args.frame:03d}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
