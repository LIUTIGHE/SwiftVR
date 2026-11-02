#!/usr/bin/env python3
"""GPU-safe train/checkpoint/custom-val/resume driver for A1-only experiments.

Runs the existing decoder trainer as a CHILD process up to each explicit
milestone, lets it terminate (freeing GPU), then invokes m11b_custom_val.py
on Wetland1 and Mangrove1. It resumes the SAME training output/checkpoint with
a stable source-code and optimizer/config fingerprint; never changes model or
loss settings behind the trainer's back.

Example:
  python tools/train_with_custom_val.py --train-config configs/t1b-a.json \
    --train-script tools/train_m11b_s1_t1b_decoder.py \
    --train-output outputs/m11b/M11B-S1-T1B/pilot500_a_custom \
    --custom-output outputs/m11b/M11B-S1-T1B/custom_auto_a \
    --milestones 125,250,375,500
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from tools.m11b_custom_val import DEFAULT_BANK,DEFAULT_E99_ROOT,DEFAULT_M11B


def milestones(value:str):
    values=[int(n.strip()) for n in value.split(",")]
    if not values or any(n<=0 for n in values) or values!=sorted(set(values)):
        raise argparse.ArgumentTypeError("milestones must be strictly increasing positive integers")
    return values


def cli_arguments(cfg:dict) -> list[str]:
    if not isinstance(cfg,dict) or not cfg:
        raise ValueError("train config must be a nonempty JSON object")
    forbidden={"resume","max-steps","output-dir","smoke-steps"}
    result=[]
    for name,value in cfg.items():
        key=name.replace("_","-")
        if key in forbidden:
            raise ValueError(f"Training argument must be controlled by orchestration: {key}")
        if not key or not all(c.isalnum() or c=="-" for c in key):
            raise ValueError(f"Invalid training argument name: {name}")
        if isinstance(value,bool):
            if value:
                result.append("--"+key)
            continue
        if not isinstance(value,(int,float,str)) or value is None:
            raise ValueError(f"Invalid scalar training arg: {name}")
        result.extend(["--"+key,str(value)])
    return result


def verified_checkpoint(run_dir:Path,step:int):
    ckpt=run_dir/"checkpoints"/f"step_{step:08d}"
    metadata=ckpt/"metadata.json"
    student=ckpt/"tiny_decoder"/"model.safetensors"
    opt=ckpt/"training_state.pt"
    if not all(p.is_file() for p in (metadata,student,opt)):
        raise ValueError(f"Incomplete training checkpoint for step={step}: {ckpt}")
    if int(json.loads(metadata.read_text())["step"])!=step:
        raise ValueError("Checkpoint step mismatch")
    return ckpt


def run(args):
    cfg=json.loads(args.train_config.read_text())
    config_args=cli_arguments(cfg)
    trainer=(ROOT/args.train_script).resolve()
    if not trainer.is_file() or ROOT not in trainer.parents:
        raise FileNotFoundError("Trainer must be a known file inside SwiftVR")
    if not (ROOT/"tools"/"m11b_custom_val.py").is_file():
        raise FileNotFoundError("Missing custom validation implementation")
    run_dir=args.train_output.resolve()
    custom_dir=args.custom_output.resolve()
    if run_dir==custom_dir or run_dir in custom_dir.parents or custom_dir in run_dir.parents:
        raise ValueError("Training and custom-output trees must not overlap")
    plan={
        "kind":"M11B-auto-custom-validation-v1",
        "trainer":str(trainer),
        "train_config":cfg,
        "milestones":args.milestones,
        "reference_bank":str(args.reference_bank.resolve()),
        "e99_root":str(args.e99_root.resolve()),
        "m11b":str(args.m11b.resolve()),
        "base":str(args.base.resolve()),
        "video":args.video,
    }
    # Keep orchestration metadata OUTSIDE the trainer output: trainer strictly
    # requires a fresh, completely absent output on its first invocation.
    state=custom_dir/"custom_validation_plan.json"
    if state.is_file():
        if json.loads(state.read_text())!=plan:
            raise ValueError("Cannot change auto-validation plan for ongoing training")
    elif run_dir.exists() or custom_dir.exists():
        raise ValueError(f"Existing unplanned train/custom output: {run_dir}, {custom_dir}")
    else:
        # Preflight WITHOUT running another model: fail if fixed reference bank absent.
        if not (args.reference_bank/"reference.json").is_file():
            raise FileNotFoundError(f"Missing cached Original SwiftVR/TinyCNN reference bank: {args.reference_bank}")
        for scene in ("Wetland1","Mangrove1"):
            if not (args.e99_root/scene/"e99"/"provenance.json").is_file():
                raise FileNotFoundError(f"Missing cached E99 full-origin reference: {scene}")
        custom_dir.mkdir(parents=True)
        state.write_text(json.dumps(plan,sort_keys=True,indent=2)+"\n")
    previous=0
    for step in args.milestones:
        dst=custom_dir/f"step_{step:08d}"
        train_ckpt=run_dir/"checkpoints"/f"step_{step:08d}"
        if not (train_ckpt/"metadata.json").is_file():
            command=[sys.executable,str(trainer),*config_args,
                     "--output-dir",str(run_dir),"--max-steps",str(step)]
            if previous:
                checkpoint=verified_checkpoint(run_dir,previous)
                command+=["--resume",str(checkpoint)]
            print(f"[auto-val] train to update {step} then release CUDA",flush=True)
            subprocess.run(command,cwd=ROOT,check=True)
        completed=verified_checkpoint(run_dir,step)
        # Child training process has terminated and freed its CUDA allocations.
        command=[
            sys.executable,str(ROOT/"tools"/"m11b_custom_val.py"),
            "--decoder-checkpoint",str(completed/"tiny_decoder"),
            "--output-root",str(dst),
            "--reference-bank",str(args.reference_bank),
            "--e99-root",str(args.e99_root),
            "--m11b",str(args.m11b),
            "--base",str(args.base),
        ]
        if args.video:
            command.append("--video")
        print(f"[auto-val] four-way custom validation of update {step}",flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
        previous=step
    print(f"[auto-val] complete through step={previous}, training and custom outputs preserved")


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-script",type=Path,required=True)
    p.add_argument("--train-config",type=Path,required=True)
    p.add_argument("--train-output",type=Path,required=True)
    p.add_argument("--custom-output",type=Path,required=True)
    p.add_argument("--milestones",type=milestones,required=True)
    p.add_argument("--reference-bank",type=Path,default=Path(DEFAULT_BANK))
    p.add_argument("--e99-root",type=Path,default=Path(DEFAULT_E99_ROOT))
    p.add_argument("--m11b",type=Path,default=Path(DEFAULT_M11B))
    p.add_argument("--base",type=Path,default=Path("checkpoints_prompt_free_no_time"))
    p.add_argument("--video",action="store_true")
    return p


if __name__=="__main__":
    run(parser().parse_args())
