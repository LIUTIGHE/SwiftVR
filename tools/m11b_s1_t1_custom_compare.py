#!/usr/bin/env python3
"""M11B-S1-T1 custom-video E99/250/500 phase comparison without GT.

All inputs must be full-origin C24 component-inference PNG sequences from the
same M11-B checkpoint, video, spatial resolution and temporal framing. HF
energy and phase metrics are descriptive, NEVER GT quality measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.m11b_s1_d0 import (
    _blur, _crop_rgb, _luma, _phase_means, _phase_spectrum, _png_list,
    _prepare_folder, parse_roi,
)

LABELS = ("e99", "step250", "step500")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e99", type=Path, required=True)
    p.add_argument("--step250", type=Path, required=True)
    p.add_argument("--step500", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--roi", type=parse_roi, default=(960, 540, 960, 960))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="Inclusive; default entire sequence")
    p.add_argument("--snapshots", default="245,248", help="Comma-separated global output frame IDs")
    p.add_argument("--low-sigma", type=float, default=8.0)
    p.add_argument("--high-sigma", type=float, default=1.5)
    return p


def _snapshot(path, images, *, frame, phase):
    h, w = next(iter(images.values())).shape[:2]
    canvas = Image.new("RGB", (w*len(LABELS), h+32), (0,0,0))
    draw = ImageDraw.Draw(canvas)
    for column,label in enumerate(LABELS):
        pic = np.uint8(np.clip(images[label] * 255 + 0.5, 0, 255))
        canvas.paste(Image.fromarray(pic),(column*w,32))
        draw.text((column*w+8,8),f"{label} global_frame={frame} phase={phase}",fill="white")
    canvas.save(path)


def compare(args):
    paths = {label:Path(getattr(args,label)) for label in LABELS}
    lists = {label:_png_list(paths[label]) for label in LABELS}
    expected = [x.name for x in lists["e99"]]
    if any([x.name for x in lists[label]] != expected for label in LABELS[1:]):
        raise ValueError("E99/250/500 PNG filenames must match exactly; no implicit temporal realignment")
    if any(not name.removesuffix(".png").isdigit() for name in expected):
        raise ValueError("Require canonical numeric frame filenames, not selected montage snapshots")
    if [int(name[:-4]) for name in expected] != list(range(len(expected))):
        raise ValueError("Require complete contiguous frame IDs starting at 0")
    n = len(expected)
    end = n-1 if args.end is None else args.end
    if args.start<0 or end>=n or end<args.start or end-args.start+1<8:
        raise ValueError(f"Need 8+ consecutive frames within [0,{n-1}]")
    if args.low_sigma<=0 or args.high_sigma<=0:
        raise ValueError("Blur sigma must be positive")
    snapshot_ids = {int(s.strip()) for s in str(args.snapshots).split(",") if s.strip()}
    rows = []
    _prepare_folder(args.output_dir)
    for i in range(args.start,end+1):
        images = {label:_crop_rgb(lists[label][i],args.roi) for label in LABELS}
        if len({images[label].shape for label in LABELS})!=1:
            raise ValueError(f"Frame {i}: spatial mismatch")
        low = {k:_blur(_luma(v),args.low_sigma) for k,v in images.items()}
        high = {k:_luma(v)-_blur(_luma(v),args.high_sigma) for k,v in images.items()}
        phase=(i+3)%4
        row={"output_frame":i,"phase":phase}
        reference_high=float(np.abs(high["e99"]).mean())
        for label in LABELS:
            row[f"luma_{label}"]=float(_luma(images[label]).mean())
            row[f"highfreq_energy_{label}"]=float(np.abs(high[label]).mean())
            row[f"highfreq_ratio_vs_e99_{label}"]=(
                float(np.abs(high[label]).mean())/max(reference_high,1e-9))
            row[f"hf_delta_mae_vs_e99_{label}"]=float(
                np.abs(high[label]-high["e99"]).mean())
            row[f"lf_delta_mae_vs_e99_{label}"]=float(
                np.abs(low[label]-low["e99"]).mean())
            row[f"mean_luma_delta_vs_e99_{label}"]=float(
                (_luma(images[label])-_luma(images["e99"])).mean())
        rows.append(row)
        if i in snapshot_ids:
            _snapshot(args.output_dir/f"global_{i:06d}_p{phase}_E99_250_500.png",
                      images,frame=i,phase=phase)

    with (args.output_dir/"frames.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fields=[key for key in rows[0] if key not in ("output_frame","phase")]
    summary=_phase_means(rows,fields)
    ratios={}
    for label in LABELS[1:]:
        ratios[label]={
            p:summary[p][f"highfreq_energy_{label}"] /
              max(summary[p]["highfreq_energy_e99"],1e-9)
            for p in summary
        }
    report={
        "kind":"M11B-S1-T1-custom-phase-v1",
        "inputs":{k:str(v.resolve()) for k,v in paths.items()},
        "n_complete_source_frames":n,
        "analyzed_inclusive":[args.start,end],
        "roi_native_output_pixels":list(args.roi) if args.roi else None,
        "phase_formula":"(output_frame+3)%4",
        "by_phase":summary,
        "phase_hf_energy_ratio_to_e99":ratios,
        "note":[
            "No paired GT: high-frequency energy or E99 disagreement alone does NOT establish quality.",
            "Phase-content and motion can bias phase-conditioned unaligned averages.",
            "Interpret together with native-resolution 3-way frame images and full-speed playback.",
            "Only comparable when all sources use identical origin/M11-B checkpoint/C24 settings.",
        ]
    }
    (args.output_dir/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","frames":[args.start,end],
                      "by_phase_ratio_vs_e99":ratios,
                      "output":str(args.output_dir)},indent=2))
    return report


if __name__=="__main__":
    compare(parser().parse_args())
