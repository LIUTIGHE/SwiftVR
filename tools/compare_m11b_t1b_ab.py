#!/usr/bin/env python3
"""Aggregate T1B-A/B four-way validations on the exact same source frames.

Produces a quantitative A/B phase table and FULL-NATIVE five-column reference
comparison: Original SwiftVR | M11-B+E99 | T1B-A | T1B-B | TinyCNN.
Does not re-infer models or treat highpass energy as ground-truth accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def run(args):
    a=args.root_a.resolve()
    b=args.root_b.resolve()
    out=args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    summary={"kind":"M11B-S1-T1B-AB-fixed-origin-v1","scenes":{}}
    out.mkdir(parents=True)
    for scene in ("Wetland1","Mangrove1"):
        sa=json.loads((a/scene/"comparison/summary.json").read_text())
        sb=json.loads((b/scene/"comparison/summary.json").read_text())
        pa=json.loads((a/scene/"provenance.json").read_text())
        pb=json.loads((b/scene/"provenance.json").read_text())
        for key in ("source","m11b_transformer_sha256","reae_sha256",
                    "full_origin_frame_count","inference"):
            if pa[key]!=pb[key]:
                raise ValueError(f"{scene}: A/B comparison invalid, mismatch on {key}")
        for key in ("frames","roi","phase_formula"):
            if sa[key]!=sb[key]:
                raise ValueError(f"{scene}: A/B evaluation mismatch on {key}")
        phases={}
        for p in ("0","1","2","3"):
            av=sa["by_phase"][p]
            bv=sb["by_phase"][p]
            phases[p]={
                "a_hf_energy":av["hf_energy_current"],
                "b_hf_energy":bv["hf_energy_current"],
                "e99_hf_energy":av["hf_energy_e99"],
                "original_swiftvr_hf_energy":av["hf_energy_original"],
                "tinycnn_hf_energy":av["hf_energy_tiny"],
                "a_over_e99":sa["hf_ratio_current_vs_e99_phase_means"][p],
                "b_over_e99":sb["hf_ratio_current_vs_e99_phase_means"][p],
                "a_hf_error_vs_e99":av["hf_mae_current_vs_e99"],
                "b_hf_error_vs_e99":bv["hf_mae_current_vs_e99"],
                "a_luma_delta_vs_e99":av["luma_delta_current_vs_e99"],
                "b_luma_delta_vs_e99":bv["luma_delta_current_vs_e99"],
            }
        summary["scenes"][scene]={"phase_comparison":phases,
                                  "a":str((a/scene).resolve()),
                                  "b":str((b/scene).resolve()),
                                  "no_paired_gt":True}
        crop_w,crop_h=sa["roi"][2],sa["roi"][3]
        one_h=crop_h+28
        # Native 960x960 content never resized. Captions remain attached to
        # each method's own quadrant; panels are composed from A/B validated
        # complete fourway PNGs.
        for frame in ((245,248) if scene=="Wetland1" else (12,124,245,248)):
            if frame >= int(sa["frames"]):
                continue
            n=f"native_fourway_{frame:06d}_p{(frame+3)%4}.png"
            ap=a/scene/"comparison"/n
            bp=b/scene/"comparison"/n
            if not (ap.is_file() and bp.is_file()):
                raise FileNotFoundError(f"Missing native fourway aligned frame {n}")
            with Image.open(ap) as ai,Image.open(bp) as bi:
                if ai.size!=(crop_w*2,one_h*2) or bi.size!=ai.size:
                    raise ValueError("Native fourway geometry mismatch")
                images=[
                    ai.crop((0,0,crop_w,one_h)),
                    ai.crop((crop_w,0,crop_w*2,one_h)),
                    ai.crop((0,one_h,crop_w,one_h*2)),
                    bi.crop((0,one_h,crop_w,one_h*2)),
                    ai.crop((crop_w,one_h,crop_w*2,one_h*2)),
                ]
                dest=Image.new("RGB",(crop_w*5,one_h),(0,0,0))
                for k,im in enumerate(images):
                    dest.paste(im,(k*crop_w,0))
                dest.save(out/f"{scene}_global_{frame:06d}_p{(frame+3)%4}_fiveway.png")
        print(json.dumps({"scene":scene,"by_phase_A_over_E99":{
            p:round(vals["a_over_e99"],4) for p,vals in phases.items()},
            "by_phase_B_over_E99":{
            p:round(vals["b_over_e99"],4) for p,vals in phases.items()}},indent=2))
    summary["interpretation"]="No GT; ratios indicate frequency response only, not correctness"
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-a",type=Path,required=True)
    p.add_argument("--root-b",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    return p


if __name__=="__main__":
    run(build_parser().parse_args())
