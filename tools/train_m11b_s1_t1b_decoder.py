#!/usr/bin/env python3
"""M11B-S1-T1B: all-phase Original ReAE teacher A/B decoder refinement.

Consumes immutable full M11B-S1-D1 caches. M11-B/encoder NEVER run in this
trainer: only the frozen Original ReAE decoder and the trainable A1 are loaded.
Per-update stateless balanced sample selection provides exact checkpoint resume.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file, load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from swiftvr.models import ReAE
from swiftvr.models.m9_factorized_decoder import M9A1FactorizedReAEDecoder
from swiftvr.training.forward import decode_reae_clip
from swiftvr.training.m11b_decoder_pairs import M11BPairedLatentDataset
from swiftvr.training.m11b_t1b_objective import t1b_objective
from swiftvr.training.stage3 import VideoMetricAccumulator
from tools.m11b_s1_d1_cache import sha256


class BalancedBatchSampler(Sampler[list[int]]):
    """Reproducible 50:50 sampling, stateless in absolute microbatch index.

    Every rank generates distinct stable indices from (seed, microstep, rank).
    No cursor is advanced by DataLoader prefetch; resume begins at the recorded
    next microstep, not the next pre-fetched index.
    """

    def __init__(self, legacy: int, ultra: int, *, start: int, stop: int,
                 batch_size: int, rank: int, world: int, seed: int):
        if min(legacy, ultra, batch_size, world) <= 0 or not 0 <= rank < world:
            raise ValueError("Invalid domain counts/batch/DDP rank")
        if start < 0 or stop < start:
            raise ValueError("Invalid microbatch range")
        self.legacy, self.ultra = int(legacy), int(ultra)
        self.start, self.stop = int(start), int(stop)
        self.batch_size = int(batch_size)
        self.rank, self.world, self.seed = int(rank), int(world), int(seed)

    def __len__(self):
        return self.stop-self.start

    def __iter__(self):
        for micro in range(self.start, self.stop):
            gen = torch.Generator().manual_seed(self.seed + micro * 1000003 + self.rank * 200003)
            indices = []
            for b in range(self.batch_size):
                position = micro * self.batch_size * self.world + self.rank * self.batch_size + b
                if position % 2 == 0:
                    indices.append(int(torch.randint(self.legacy, (1,), generator=gen).item()))
                else:
                    indices.append(self.legacy + int(torch.randint(self.ultra, (1,), generator=gen).item()))
            yield indices


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints_prompt_free_no_time"))
    p.add_argument("--m11b-checkpoint", type=Path, required=True)
    p.add_argument("--a1-init", type=Path, required=True)
    p.add_argument("--legacy-train", type=Path, required=True)
    p.add_argument("--ultra-train", type=Path, required=True)
    p.add_argument("--legacy-val", type=Path, required=True)
    p.add_argument("--path-root", type=Path, default=Path("."))
    p.add_argument("--lpips-weights", type=Path, default=None)
    p.add_argument("--variant", choices=("A", "B"), required=True,
                   help="A: full Original RGB + GT temporal; B: Original HF + GT LF/temporal")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None, help="Own saved step checkpoint")
    p.add_argument("--dtype", choices=("bfloat16","float32"), default="bfloat16")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--schedule-steps", type=int, default=6000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--teacher-weight", type=float, default=1.0)
    p.add_argument("--lowfreq-gt-weight", type=float, default=1.0)
    p.add_argument("--temporal-weight", type=float, default=0.5)
    p.add_argument("--gt-lpips-weight", type=float, default=0.0,
                   help="Off by default in both A/B to isolate supervision")
    p.add_argument("--lpips-frame-batch", type=int, default=2)
    p.add_argument("--highpass-kernel", type=int, default=7)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--smoke-steps", type=int, default=0, help="Separate smoke run; no val images")
    return p


def validate_args(a):
    for k in ("batch_size","gradient_accumulation_steps","max_steps","schedule_steps",
              "save_every","validate_every","log_every","lpips_frame_batch"):
        if getattr(a,k)<=0:
            raise ValueError(f"Invalid positive {k}")
    if a.num_workers<0 or a.smoke_steps<0 or not 0<=a.warmup_steps<a.schedule_steps:
        raise ValueError("Invalid worker/schedule/smoke parameters")
    if a.max_steps>a.schedule_steps or not 0<a.min_lr_ratio<=1:
        raise ValueError("Max steps must fit scheduler and LR ratio must be (0,1]")
    for k in ("learning_rate","max_grad_norm"):
        if not math.isfinite(getattr(a,k)) or getattr(a,k)<=0:
            raise ValueError(f"Invalid {k}")
    for k in ("teacher_weight","lowfreq_gt_weight","temporal_weight",
              "gt_lpips_weight","weight_decay"):
        if not math.isfinite(getattr(a,k)) or getattr(a,k)<0:
            raise ValueError(f"Invalid {k}")
    if a.teacher_weight<=0:
        raise ValueError("Nonzero Original teacher supervision required")
    if a.gt_lpips_weight>0 and a.lpips_weights is None:
        raise ValueError("GT LPIPS requires complete local AlexNet weights")
    if a.highpass_kernel%2!=1 or a.highpass_kernel<3:
        raise ValueError("Highpass kernel must be odd and >=3")
    for name in ("base_checkpoint","m11b_checkpoint","a1_init","legacy_train",
                 "ultra_train","legacy_val","path_root","output_dir","lpips_weights","resume"):
        v=getattr(a,name)
        if v is not None:
            setattr(a,name,v.expanduser().resolve())
    for source in (a.base_checkpoint,a.m11b_checkpoint,a.a1_init,
                   a.legacy_train,a.ultra_train,a.legacy_val):
        if a.output_dir == source or a.output_dir in source.parents or source in a.output_dir.parents:
            raise ValueError("Output overlaps an immutable source tree")
    if a.resume and a.resume.parent.parent!=a.output_dir:
        raise ValueError("Resume checkpoint must belong to its original output dir")
    if a.output_dir.exists() and any(a.output_dir.iterdir()) and not a.resume:
        raise FileExistsError(f"Training output exists: {a.output_dir}")


def lr_for_update(step,a):
    if a.warmup_steps and step<=a.warmup_steps:
        return a.learning_rate*step/a.warmup_steps
    fraction=min(max((step-a.warmup_steps)/max(a.schedule_steps-a.warmup_steps,1),0),1)
    return a.learning_rate*(a.min_lr_ratio+(1-a.min_lr_ratio)*0.5*(1+math.cos(math.pi*fraction)))


def atomic_json(path:Path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def append_jsonl(path:Path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(record,sort_keys=True)+"\n")


def identity_config(a,world):
    paths=[
        a.base_checkpoint/"reae.safetensors",
        a.a1_init/"model.safetensors",a.a1_init/"config.json",
        a.legacy_train/"metadata.json", a.ultra_train/"metadata.json",
        a.legacy_val/"metadata.json",
        Path(__file__).resolve(),
        ROOT/"swiftvr/training/m11b_decoder_pairs.py",
        ROOT/"swiftvr/training/m11b_phase_objective.py",
        ROOT/"swiftvr/training/m11b_t1b_objective.py",
        ROOT/"swiftvr/models/m9_factorized_decoder.py",
    ]
    if a.lpips_weights:
        paths.append(a.lpips_weights)
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    # Exclude max_steps/resume/smoke_steps: continuation can extend stop while
    # preserving everything about the schedule, objective and data.
    runtime={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()
             if k not in ("max_steps","resume","smoke_steps")}
    runtime["world_size"]=world
    runtime["global_batch"]=world*a.batch_size*a.gradient_accumulation_steps
    runtime["fingerprints"]={str(p):sha256(p) for p in paths}
    runtime["kind"]="M11B-S1-T1B-allphase-original-v1"
    return runtime


def t1b_loss_kwargs(a):
    return dict(
        variant=a.variant,
        teacher_weight=a.teacher_weight,
        lowfreq_gt_weight=a.lowfreq_gt_weight,
        temporal_weight=a.temporal_weight,
        highpass_kernel=a.highpass_kernel,
        gt_lpips_weight=a.gt_lpips_weight,
    )


def rng_state(device):
    return {"python":random.getstate(),"numpy":np.random.get_state(),
            "cpu":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state(device)}


def restore_rng(value,device):
    random.setstate(value["python"]); np.random.set_state(value["numpy"])
    torch.set_rng_state(value["cpu"])
    torch.cuda.set_rng_state(value["cuda"],device)


def _encode_png_rows(output, pred, gt, teacher, *, sample_id, step):
    """Full native 384px-pixel panels for phase0 and phase3, not resized previews."""
    from PIL import Image,ImageDraw
    root=output/"validation_visuals"/f"step_{step:08d}"
    root.mkdir(parents=True,exist_ok=True)
    for index in (0,1,5,9,12):
        if index>=pred.shape[1]:
            continue
        images=[]
        for tensor in (gt[0,index],teacher[0,index],pred[0,index]):
            frame=tensor.detach().float().clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy()
            images.append(Image.fromarray(frame))
        width,height=images[0].size
        picture=Image.new("RGB",(width*3,height+28),(0,0,0))
        for col,(im,label) in enumerate(zip(images,("GT","Original decoder","A1 refinement"))):
            picture.paste(im,(col*width,28))
            ImageDraw.Draw(picture).text((col*width+6,8),f"{label} frame{index} phase{(index+3)%4}",fill="white")
        picture.save(root/f"val_{sample_id:02d}_frame_{index:03d}.png")


@torch.no_grad()
def evaluate(decoder,reae,validation,perceptual,*,device,dtype,args,step):
    decoder.eval()
    loader=DataLoader(validation,batch_size=1,shuffle=False,num_workers=0)
    total={};count=0
    a1_quality=VideoMetricAccumulator()
    original_quality=VideoMetricAccumulator()
    for i, batch in enumerate(loader):
        z=batch["z_sr"].to(device=device,dtype=dtype)
        gt=batch["hr"].to(device=device,dtype=torch.float32)
        with torch.autocast("cuda",dtype=dtype,enabled=dtype==torch.bfloat16):
            target=decode_reae_clip(reae,z,output_frames=13,clamp=False)
            pred=decoder(z,output_frames=13,clamp=False)
        metrics=t1b_objective(pred,gt,target,gt_lpips=perceptual,**t1b_loss_kwargs(args))
        # Compare both decoders against the SAME GT; clamp only for pixel
        # metrics, while the training objective remains unclamped.
        a1_quality.update(pred.float(),gt.float(),clamp=True)
        original_quality.update(target.float(),gt.float(),clamp=True)
        for name,value in metrics.items():
            total[name]=total.get(name,0)+float(value.detach().item())
        count+=1
        if i<2:
            _encode_png_rows(args.output_dir,pred,gt,target,sample_id=i,step=step)
    decoder.train()
    result={name:value/count for name,value in total.items()}
    result.update({"step":step,"validation_samples":count,
                   "a1_gt_metrics":a1_quality.compute(),
                   "original_gt_metrics":original_quality.compute(),
                   "validation_kind":"T1B_original_allphase_with_matched_GT", "variant":args.variant,
                   "custom_video_quality_pass":False})
    atomic_json(args.output_dir/"validation"/f"step_{step:08d}.json",result)
    append_jsonl(args.output_dir/"validation_log.jsonl",result)
    print(f"[M11B-S1-T1B-{args.variant} val] "+json.dumps(result,sort_keys=True),flush=True)
    return result


def main():
    a=parser().parse_args()
    validate_args(a)
    rank=int(os.environ.get("RANK",0))
    local_rank=int(os.environ.get("LOCAL_RANK",0))
    world=int(os.environ.get("WORLD_SIZE",1))
    if not torch.cuda.is_available():
        raise RuntimeError("M11B-S1-T1 requires a CUDA GPU")
    torch.cuda.set_device(local_rank)
    device=torch.device("cuda",local_rank)
    dtype=torch.bfloat16 if a.dtype=="bfloat16" else torch.float32
    if dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 not supported by this GPU")
    if world>1:
        from datetime import timedelta
        dist.init_process_group("nccl",timeout=timedelta(hours=4))

    def barrier():
        if world>1:
            dist.barrier()

    try:
        random.seed(a.seed+rank);np.random.seed(a.seed+rank);torch.manual_seed(a.seed+rank)
        legacy=M11BPairedLatentDataset(a.legacy_train,model_root=a.m11b_checkpoint,
            reae_root=a.base_checkpoint,path_root=a.path_root)
        ultra=M11BPairedLatentDataset(a.ultra_train,model_root=a.m11b_checkpoint,
            reae_root=a.base_checkpoint,path_root=a.path_root)
        val=M11BPairedLatentDataset(a.legacy_val,model_root=a.m11b_checkpoint,
            reae_root=a.base_checkpoint,path_root=a.path_root)
        if len(val)!=13:
            raise ValueError("Expected canonical val13")
        if (legacy.meta["domain"],ultra.meta["domain"],val.meta["split"])!=("legacy","ultra","val"):
            raise ValueError("Wrong train/validation cache domains")
        lineage=identity_config(a,world)
        a.output_dir.mkdir(parents=True,exist_ok=True)
        config_path=a.output_dir/"run_config.json"
        if a.resume:
            existing=json.loads(config_path.read_text())
            if existing!=lineage:
                raise ValueError("Resume source/code/loss/world/batch fingerprints differ")
        elif rank==0:
            atomic_json(config_path,lineage)
        barrier()

        student=M9A1FactorizedReAEDecoder.from_pretrained(a.a1_init,device="cpu",
            dtype=torch.float32).to(device).train()
        # Original decoder is the RGB texture reference, NOT the temporal truth.
        reae=ReAE(str(a.base_checkpoint/"reae.safetensors"))
        # The full checkpoint must load strictly, but this stage only decodes.
        # Drop the unused encoder before transferring the teacher to GPU.
        reae.encoder=torch.nn.Identity()
        reae.requires_grad_(False)
        reae.to(device=device,dtype=dtype).eval()
        perceptual=None
        if a.gt_lpips_weight:
            from swiftvr.training.m8_task import LocalLPIPS
            perceptual=LocalLPIPS(a.lpips_weights,frame_batch=a.lpips_frame_batch).to(device).eval()
        optimizer=torch.optim.AdamW(student.parameters(),lr=a.learning_rate,
                                    weight_decay=a.weight_decay,eps=1e-8)
        step=0;micro_cursor=0
        if a.resume:
            state=torch.load(a.resume/"training_state.pt",map_location="cpu",weights_only=False)
            if state["lineage"]!=lineage:
                raise ValueError("Checkpoint lineage differs from current run")
            student.load_state_dict(load_file(str(a.resume/"tiny_decoder/model.safetensors"),
                                              device="cpu"),strict=True)
            optimizer.load_state_dict(state["optimizer"])
            step,micro_cursor=int(state["step"]),int(state["micro_cursor"])
            if micro_cursor != step*a.gradient_accumulation_steps:
                raise ValueError("Inconsistent optimizer/loader resume microstep cursor")
            restore_rng(state["rng_states"][rank],device)
        ddp=DDP(student,device_ids=[local_rank],broadcast_buffers=False,
                find_unused_parameters=False) if world>1 else student
        params=list(student.parameters())
        stop=step+a.smoke_steps if a.smoke_steps else a.max_steps
        if stop<=step or stop>a.schedule_steps:
            raise ValueError("Training stop incompatible with checkpoint or scheduler")
        all_data=ConcatDataset((legacy,ultra))
        start_micro=micro_cursor
        end_micro=stop*a.gradient_accumulation_steps
        sampler=BalancedBatchSampler(len(legacy),len(ultra),start=start_micro,
            stop=end_micro,batch_size=a.batch_size,rank=rank,world=world,seed=a.seed)
        worker_args={"num_workers":a.num_workers}
        if a.num_workers:
            worker_args.update({"persistent_workers":True,"prefetch_factor":2})
        loader=DataLoader(all_data,batch_sampler=sampler,pin_memory=True,**worker_args)
        iterator=iter(loader)

        def save_checkpoint():
            states=[None]*world
            if world>1:
                dist.all_gather_object(states,rng_state(device))
            else:
                states[0]=rng_state(device)
            folder=a.output_dir/"checkpoints"/f"step_{step:08d}"
            if rank==0:
                if folder.exists():
                    raise FileExistsError(f"Refuse to overwrite checkpoint: {folder}")
                folder.mkdir(parents=True)
                student.save_pretrained(folder/"tiny_decoder")
                # Ensure FP32 master weights are written, never bf16 inference weights.
                for tensor in student.state_dict().values():
                    if tensor.is_floating_point() and tensor.dtype!=torch.float32:
                        raise RuntimeError("Trainable decoder weights no longer FP32")
                state={"step":step,"micro_cursor":micro_cursor,"optimizer":optimizer.state_dict(),
                       "rng_states":states,"lineage":lineage}
                temporary=folder/"training_state.tmp"
                torch.save(state,temporary);os.replace(temporary,folder/"training_state.pt")
                atomic_json(folder/"metadata.json",{
                    "step":step,"phase":"M11B-S1-T1B","variant":a.variant,"train_scope":"A1_only",
                    "m11b_checkpoint":str(a.m11b_checkpoint),
                    "a1_initialization":str(a.a1_init),
                    "original_decoder_teacher_role":"all_phase_rgb" if a.variant=="A" else "all_phase_highfreq",
                })
                atomic_json(a.output_dir/"latest.json",{"step":step,"path":str(folder)})
            barrier()
            return folder

        if not a.resume and not a.smoke_steps:
            # Step0 baseline, paired GT; no custom-video success is implied.
            barrier()
            if rank==0:
                evaluate(student,reae,val,perceptual,device=device,dtype=dtype,args=a,step=0)
            barrier()
            save_checkpoint()

        torch.cuda.synchronize(device)
        started=time.monotonic();log_started=started
        reduced_terms={};accum_steps_since_log=0
        optimizer.zero_grad(set_to_none=True)
        while step<stop:
            lr=lr_for_update(step+1,a)
            for group in optimizer.param_groups: group["lr"]=lr
            last_loss=None
            for micro in range(a.gradient_accumulation_steps):
                batch=next(iterator)
                latent=batch["z_sr"].to(device=device,dtype=dtype,non_blocking=True)
                gt=batch["hr"].to(device=device,dtype=torch.float32,non_blocking=True)
                if latent.shape[1:]!=(4,48,24,24) or gt.shape[1:]!=(13,3,384,384):
                    raise ValueError("Formal T1 expects canonical T13/384x384 paired views")
                with torch.no_grad(),torch.autocast("cuda",dtype=dtype,enabled=dtype==torch.bfloat16):
                    teacher=decode_reae_clip(reae,latent,output_frames=13,clamp=False)
                context=ddp.no_sync() if world>1 and micro<a.gradient_accumulation_steps-1 else nullcontext()
                with context:
                    with torch.autocast("cuda",dtype=dtype,enabled=dtype==torch.bfloat16):
                        pred=ddp(latent,output_frames=13,clamp=False)
                    terms=t1b_objective(pred,gt,teacher,gt_lpips=perceptual,**t1b_loss_kwargs(a))
                    (terms["loss"]/a.gradient_accumulation_steps).backward()
                for k,v in terms.items():
                    reduced_terms[k]=reduced_terms.get(k,torch.zeros((),device=device))+v.detach()/a.gradient_accumulation_steps
                micro_cursor+=1
                last_loss=terms["loss"]
            if not any(p.grad is not None for p in params):
                raise RuntimeError("A1 has no gradients")
            if any(p.grad is not None for p in reae.parameters()):
                raise RuntimeError("Frozen Original ReAE teacher unexpectedly has gradients")
            # clip_grad_norm_ checks non-finite total norm across all A1 parameters.
            # Avoid a separate per-parameter CUDA synchronization in Python.
            norm=torch.nn.utils.clip_grad_norm_(
                params,a.max_grad_norm,error_if_nonfinite=True
            )
            if float(norm.detach().item()) <= 0:
                raise RuntimeError("A1 gradient norm is zero; decoder training path inactive")
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            step+=1;accum_steps_since_log+=1
            if step%a.log_every==0 or step==stop:
                names=list(reduced_terms)
                values=torch.stack([reduced_terms[k] for k in names])/accum_steps_since_log
                if world>1:
                    dist.all_reduce(values,op=dist.ReduceOp.SUM)
                    values/=world
                # CUDA launches are asynchronous; synchronize before reporting a
                # wall-clock update rate or elapsed time. Do not include validation
                # and checkpoint I/O in the subsequent train-only interval.
                torch.cuda.synchronize(device)
                if rank==0:
                    log_time=time.monotonic()
                    record={"step":step,"lr":lr,"rank0_grad_norm":float(norm.detach()),
                            "elapsed_s":log_time-started,
                            "updates_per_s":accum_steps_since_log/max(log_time-log_started,1e-9),
                            "global_batch":lineage["global_batch"],
                            **dict(zip(names,[float(v) for v in values.cpu()]))}
                    append_jsonl(a.output_dir/"train_log.jsonl",record)
                    print(f"[M11B-S1-T1B-{a.variant}] "+json.dumps(record,sort_keys=True),flush=True)
                reduced_terms={};accum_steps_since_log=0;log_started=time.monotonic()
            save_due=step%a.save_every==0 or step==stop
            val_due=(not a.smoke_steps) and (step%a.validate_every==0 or step==stop)
            if val_due and rank==0:
                evaluate(student,reae,val,perceptual,device=device,dtype=dtype,args=a,step=step)
            if val_due:
                barrier()
            if save_due or val_due:
                save_checkpoint()
        if rank==0:
            atomic_json(a.output_dir/"summary.json",{
                "status":"smoke_pass" if a.smoke_steps else "training_budget_completed",
                "phase":"M11B-S1-T1B", "variant":a.variant,
                "step":step,"legacy_samples":len(legacy),"ultra_samples":len(ultra),
                "validation_samples":len(val),"visual_quality_pass":False,
                "elapsed_seconds":time.monotonic()-started,
                "m11b_updated":False,"decoder_updated":True})
    finally:
        if world>1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__=="__main__":
    main()
