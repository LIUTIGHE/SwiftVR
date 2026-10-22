#!/usr/bin/env python3
"""One-Decoder custom validation: fixed Original SwiftVR/E99/TinyCNN references.

Run OUTSIDE the training process, after an optimizer checkpoint has been
closed and CUDA memory released. Renders only CURRENT M11-B + A1 decoder.
Original SwiftVR means complete conditional SwiftVR, NOT Original ReAE with
M11-B. TinyCNN is the stored external AISR video, NOT the tiny A1 decoder.

Same-origin C24 inference, integer output-frame indexing (i+3)%4, lossless PNG
retention, native ROI PNG panels and per-phase diagnostics. Custom video has no
GT; HF energy and teacher divergence are not objective restoration quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from tools.m11b_s1_d0 import _blur, _luma, _phase_means

SCENES = ("Wetland1", "Mangrove1")
DEFAULT_M11B = "outputs/b2b/m11b_stagea_ultra_bootstrap30k_v1/checkpoints/step_00030000"
DEFAULT_E99_ROOT = "outputs/m11b/M11B-S1-T1/custom_pilot500_v1"
DEFAULT_BANK = "outputs/custom/p1_reference_v1"


def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path:Path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    os.replace(temp,path)


def load_bank(root:Path) -> dict:
    info=json.loads((root/"reference.json").read_text())
    if info.get("kind")!="m8_task_custom_reference_v1":
        raise ValueError(f"Unexpected Original SwiftVR / TinyCNN reference bank: {root}")
    if info.get("inference") != dict(upscale=3,clip_len=24,dit_overlap=0,
                                      dtype="bfloat16",attention_backend="sdpa"):
        raise ValueError("Reference bank was not generated with matching C24 settings")
    if {s["name"] for s in info["clips"]}!=set(SCENES):
        raise ValueError("Expected Wetland1 and Mangrove1 references")
    # Do NOT re-run original conditional SwiftVR or TinyCNN. Check that the
    # signed bank inputs are still the same; original PNGs are immutable here.
    for entry in info["original_files"] + [
        row[key] for row in info["clips"] for key in ("input","basic")
    ]:
        file=Path(entry["path"])
        st=file.stat()
        if (st.st_size,st.st_mtime_ns)!=(entry["size"],entry["mtime_ns"]):
            raise ValueError(f"Source bank file size/mtime changed: {file}")
    return info


def scene_reference(bank:dict,scene:str,e99_root:Path,base:Path,dit:Path):
    entry=next(item for item in bank["clips"] if item["name"]==scene)
    prov=json.loads((e99_root/scene/"e99"/"provenance.json").read_text())
    if prov["kind"]!="M11B-S1-T1-custom-C24-v1":
        raise ValueError("Expected prior C24 E99 full-origin provenance")
    if prov["source_video"]["sha256"]!=entry["input"]["sha256"]:
        raise ValueError(f"Original and E99 input movie hashes differ: {scene}")
    if prov["transformer"]["sha256"]!=file_sha(dit/"transformer/diffusion_pytorch_model.safetensors"):
        raise ValueError(f"E99 M11-B checkpoint identity mismatch: {scene}")
    if prov["reae"]["sha256"]!=file_sha(base/"reae.safetensors"):
        raise ValueError(f"E99 ReAE checkpoint identity mismatch: {scene}")
    if prov["inference"] != (bank["inference"] | {"png": True}):
        raise ValueError(f"Original and E99 C24 settings differ: {scene}")
    return entry,prov


def _png_files(path:Path,n:int):
    files=sorted(path.glob("*.png"))
    numbers=sorted(int(p.stem) for p in files)
    if numbers!=list(range(n)):
        raise ValueError(f"Expected {n} contiguous full-origin PNGs at {path}, got {len(files)}")
    return sorted(files,key=lambda p:int(p.stem))


def _read_roi(src, index:int, roi:list[int]):
    x,y,w,h=roi
    frame=src.frame(index)
    if (frame.shape[1]<x+w or frame.shape[0]<y+h):
        raise ValueError("ROI outside source native pixel dimensions")
    return frame[y:y+h,x:x+w,:3]


def evaluate(scene:str,root:Path,bank:dict,
             e99_root:Path,compare_dir:Path,video:bool=False):
    entry=next(c for c in bank["clips"] if c["name"]==scene)
    roi=entry["roi"]
    n=int(entry["frames"])
    e99_dir=e99_root/scene/"e99"/"png"
    _png_files(e99_dir,n)
    _png_files(root/"png",n)
    orig=FrameSource(Path(entry["original"]))
    e99=FrameSource(e99_dir)
    current=FrameSource(root/"png")
    tiny=FrameSource(Path(entry["basic"]["path"]))
    for name,src in (("original",orig),("E99",e99),("current",current)):
        if len(src)!=n:
            raise ValueError(f"{scene}: {name} does not have {n} frames")
    if len(tiny)<n:
        raise ValueError(f"{scene}: TinyCNN video shorter than logical {n} frames")
    for src in (orig,e99,current,tiny):
        if (src.width,src.height)!=(orig.width,orig.height):
            raise ValueError("Native output geometry mismatch; refusing implicit resizing")
    if compare_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prior validation: {compare_dir}")
    compare_dir.mkdir(parents=True)
    images_at = set([0, 1, 12, n//2, 245, 248]) & set(range(n))
    roi_height,roi_width=roi[3],roi[2]
    ffv1=None;err=None
    if video:
        import subprocess as sp
        rate=FrameSource(Path(entry["input"]["path"])).fps
        width,height=roi_width*2,2*(roi_height+28)
        cmd=["ffmpeg","-nostdin","-n","-v","error",
             "-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}",
             "-r",str(rate),"-i","-","-an","-c:v","ffv1","-level","3",
             str(compare_dir/"native_fourway.mkv")]
        err=(compare_dir/"ffmpeg.log").open("w")
        ffv1=sp.Popen(cmd,stdin=sp.PIPE,stderr=err)
    rows=[]
    for i in range(n):
        tin_index=_auto_basic_index(i,basic_count=len(tiny),reference_count=n)
        names=("Original SwiftVR","E99","Current","TinyCNN")
        f={
            "original":_read_roi(orig,i,roi),
            "e99":_read_roi(e99,i,roi),
            "current":_read_roi(current,i,roi),
            "tiny":_read_roi(tiny,tin_index,roi),
        }
        yl={k:_luma(v.astype(np.float32)/255.0) for k,v in f.items()}
        hp={k:v-_blur(v,1.5) for k,v in yl.items()}
        lp={k:_blur(v,8.0) for k,v in yl.items()}
        row={"output_frame":i,"phase":(i+3)%4,"tiny_frame":tin_index}
        for k in f:
            row[f"hf_energy_{k}"]=float(np.abs(hp[k]).mean())
            row[f"luma_mean_{k}"]=float(yl[k].mean())
        for ref in ("e99","original","tiny"):
            row[f"hf_mae_current_vs_{ref}"]=float(np.abs(hp["current"]-hp[ref]).mean())
            row[f"lf_mae_current_vs_{ref}"]=float(np.abs(lp["current"]-lp[ref]).mean())
            row[f"luma_delta_current_vs_{ref}"]=float((yl["current"]-yl[ref]).mean())
        rows.append(row)
        if i in images_at or ffv1:
            canvas=Image.new("RGB",(roi_width*2,(roi_height+28)*2),(0,0,0))
            draw=ImageDraw.Draw(canvas)
            for j,(label,key) in enumerate(zip(names,("original","e99","current","tiny"))):
                x=(j%2)*roi_width
                y=(j//2)*(roi_height+28)
                canvas.paste(Image.fromarray(f[key]),(x,y+28))
                draw.text((x+7,y+7),
                          f"{label}: frame{i} phase{(i+3)%4}" + 
                          (f" mapped={tin_index}" if key=="tiny" else ""),
                          fill=(255,255,255))
            if i in images_at:
                canvas.save(compare_dir/f"native_fourway_{i:06d}_p{(i+3)%4}.png")
            if ffv1:
                ffv1.stdin.write(np.asarray(canvas,dtype=np.uint8).tobytes())
    if ffv1:
        ffv1.stdin.close()
        rc=ffv1.wait()
        err.close()
        if rc:
            raise RuntimeError(f"FFV1 writing failed, check {compare_dir/'ffmpeg.log'}")
    with (compare_dir/"frames.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    mean_by_phase=_phase_means(rows,[k for k in rows[0] if k not in ("output_frame","phase","tiny_frame")])
    ratio={p:mean_by_phase[p]["hf_energy_current"]/max(mean_by_phase[p]["hf_energy_e99"],1e-12)
           for p in mean_by_phase}
    summary={
        "kind":"M11B-custom-fourway-v1","scene":scene,
        "decoder_checkpoint":str(info["decoder"]["path"]),
        "frames":n,"roi":roi,"phase_formula":"(output_frame+3)%4",
        "by_phase":mean_by_phase,
        "hf_ratio_current_vs_e99_phase_means":ratio,
        "tiny_alignment":"ordinal length-normalized, NOT timestamp aligned",
        "caveat":"No GT. HF energy and reference disagreement do not establish actual image correctness.",
        "fixed_references":["Original SwiftVR complete conditional model","M11B+E99","External TinyCNN AISR video"],
    }
    atomic_json(compare_dir/"summary.json",summary)
    print(json.dumps({"scene":scene,"frames":n,"hf_ratio_current_vs_e99":ratio,
                      "summary":str(compare_dir/"summary.json")},indent=2))


def run(args):
    base=args.base.resolve();dit=args.m11b.resolve()
    bank_root=args.reference_bank.resolve()
    e99_root=args.e99_root.resolve()
    info=load_bank(bank_root)
    for scene in args.scenes:
        entry,e99=scene_reference(info,scene,e99_root,base,dit)
        ckpt=args.decoder_checkpoint.resolve()
        if not (ckpt/"model.safetensors").is_file() or not (ckpt/"config.json").is_file():
            raise FileNotFoundError(f"Missing A1 decoder checkpoint: {ckpt}")
        root=args.output_root.resolve()/scene
        manifest=root/"provenance.json"
        record={
            "kind":"M11B-custom-current-C24-v1",
            "decoder":{"path":str(ckpt),"sha256":file_sha(ckpt/"model.safetensors")},
            "decoder_config_sha256":file_sha(ckpt/"config.json"),
            "source":entry["input"],
            "m11b_transformer_sha256":e99["transformer"]["sha256"],
            "reae_sha256":e99["reae"]["sha256"],
            "full_origin_frame_count":entry["frames"],
            "inference":e99["inference"],
        }
        if root.exists():
            if not manifest.is_file() or json.loads(manifest.read_text())!=record:
                raise ValueError(f"Partial or mismatched custom output: {root}. Never overwrite.")
            _png_files(root/"png",int(entry["frames"]))
        else:
            root.mkdir(parents=True)
            cmd=[
                sys.executable,"scripts/inference_custom_components.py",
                "--input",entry["input"]["path"],"--output",str(root/"png"),
                "--base-checkpoint",str(base),
                "--transformer-checkpoint",str(dit),
                "--transformer-type","moe","--decoder-type","m9a1",
                "--decoder-checkpoint",str(ckpt),
                "--upscale","3","--clip-len","24","--dit-overlap","0",
                "--dtype","bfloat16","--attention-backend","sdpa","--png",
            ]
            with (root/"inference.log").open("w") as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Custom inference failed ({result.returncode}), see {root/'inference.log'}")
            _png_files(root/"png",int(entry["frames"]))
            atomic_json(manifest,record)
        # Reuse previously rendered Current video if analysis needs retry, but
        # NEVER overwrite a completed fourway diagnostic.
        analysis=root/"comparison"
        if (analysis/"summary.json").is_file():
            print(f"[custom-val] already complete, preserved: {analysis}")
        else:
            evaluate(scene,root,info,e99_root,analysis,video=args.video)
    return 0


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decoder-checkpoint",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--reference-bank",type=Path,default=Path(DEFAULT_BANK))
    p.add_argument("--e99-root",type=Path,default=Path(DEFAULT_E99_ROOT))
    p.add_argument("--m11b",type=Path,default=Path(DEFAULT_M11B))
    p.add_argument("--base",type=Path,default=Path("checkpoints_prompt_free_no_time"))
    p.add_argument("--scenes",nargs="+",choices=SCENES,default=list(SCENES))
    p.add_argument("--video",action="store_true",help="Also produce full-resolution FFV1 four-way")
    return p


if __name__=="__main__":
    raise SystemExit(run(build_parser().parse_args()))
