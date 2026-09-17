# M10-R2: Transformer-only perceptual detail recovery

## Purpose

Recover the detail lost during Stage-A -> M8-A compression. This is NOT a
four-phase flicker fix, a new decoder, or a new MoE architecture. Freeze the
original ReAE and A1 E99. Start again from immutable M8-A30k, not the unsuccessful
M10-V1 step500 checkpoints. Only Transformer parameters are optimized. Deployment
remains approximately 500.8 GFLOPs/frame under the existing 1920x1088 census.

Teacher: cached Stage-A D3072 step200000 velocity, decoded by Original ReAE.
Student: newly recomputed M8-A velocity, decoded by frozen A1 with input autograd.
GT is diagnostic only. No new caches, temporal strides, crops, data, or source
frame-rate assumptions. Both losses and local weight hashes are recorded.

## Objective

```
0.01 * velocity_NMSE
+ 1.0 * RGB_L1(student, StageA_teacher)
+ 0.2 * LPIPS_Alex(student, StageA_teacher)
+ 0.01 * router_balance
```

Velocity cosine, single-scale HF loss and HF temporal loss have training weight
zero in this recipe. HF and HF temporal errors are still measured. The single
velocity anchor prevents discarding representation supervision without duplicating
it with a cosine term. These are initial experimental weights, not proven optimal
or automatically calibrated weights. LR is 1e-5 versus V1's 5e-6. This is a new
recovery recipe, not a one-variable LPIPS ablation.

LPIPS uses the same reference AlexNet v0.1 metric as A1: clamp RGB to [0,1], map
to [-1,1], evaluate all frames, mean the results. Its forward is FP32 and split
into microbatches with activation checkpointing. No phases or difficult frames
are dropped. Both trunk and calibrated heads are frozen; input gradients remain.

The original M10 CLI defaults and `tools/run_m10_gate.sh spatial|temporal` recipe
are preserved. R2 is an opt-in launcher using the same trainer.

## Manual synchronization

Copy these six changed/new files via GitHub Web / VSCode SSH:

```
swiftvr/training/m10_recovery.py                 # updated, opt-in LPIPS
swiftvr/training/m10_perceptual.py               # new
tools/train_m10_backbone_recovery_ddp.py         # updated
tools/run_m10_r2.sh                             # new
tests/test_m10_r2_recovery.py                    # new
docs/m10_r2_recovery.md                         # new
```

Keep all earlier M10 dependencies, particularly `tools/run_m10_gate.sh`.
No server-side git is needed. Baseline model/inference files are unchanged.

## Tests and one-step smoke

```bash
cd /data1/a/SwiftVR
python -m unittest discover -s tests -p 'test_m10*_recovery.py' -v

GPU_IDS=7 MAX_STEPS=1 WARMUP=0 ACCUM=1 \
OUT=outputs/b2b/m10_r2_smoke \
bash tools/run_m10_r2.sh --visual-samples 0
```

In the full environment, `test_real_a1_input_backward` should not skip. The new
`test_real_local_lpips_matches_reference` compares every pretrained parameter and
metric outputs against reference LPIPS, blocking network downloads. It requires
the default torchvision pretrained AlexNet cache to be present; it may skip on
minimal test environments. Core tests use a tiny frozen feature net to verify
normalization, microbatch gradients, loss routing, phase counts, and legacy
regression; they do not establish real pretrained-model quality.

The smoke runs val0, a real optimizer step and val1. It tests LPIPS input gradients
through frozen A1 and the normal MoE DDP forward. GPU/DDP must run on the server.

## Local LPIPS weights

R2 never downloads weights. It first uses
`torch.hub.get_dir()/checkpoints/alexnet-owt-7be5be79.pth`; otherwise exactly one
`alexnet*.pth` in that directory. These are the torchvision weights previously
used by A1. LPIPS calibration comes from the installed lpips package's
`weights/v0.1/alex.pth`.

A different local location can be supplied explicitly:

```bash
export ALEXNET_WEIGHTS=/actual/existing/alexnet.pth
```

Internally LPIPS is constructed without pretrained downloading, then every trunk
tensor and all five calibrated heads are loaded from local files. It is NOT a
random-AlexNet substitute. Missing files produce a local error, not a Hub request.
No LPIPS or font/weight files are included in this code package.

## Recovery gate

```bash
GPU_IDS=0,1,2,3 bash tools/run_m10_r2.sh
```

Defaults: 1000 steps, per-GPU batch1, accumulation4, global batch16 on four GPUs,
LR1e-5, warmup50, cosine LR schedule, BF16 Transformer/A1, FP32 LPIPS/master
parameters, SDPA. Validate/save at 0/250/500/750/1000 (step0 references baseline
weights instead of duplicating them). All 13 validation frames are exported as
PNG panels, plus comparison video at 30 playback FPS. Playback FPS is not a
statement about the source cadence.

Default output: `outputs/b2b/m10_r2_perceptual_gate1000`. Outputs must be new,
nonempty run directories are never overwritten. Use `OUT` to choose another.
The launcher continues to read Stage-A train cache from M3's run_config and
Stage-A val cache from M8-A's run_config, as the successful M10 runs did.
Existing `BASE`, `M8A`, `A1_RUN`, `A1_DECODER`, `STAGEA_TRAIN_CACHE`,
`STAGEA_VAL_CACHE`, `MICRO_BS`, `ACCUM` overrides still work.

This is a weights-only checkpoint workflow. A new run initialized with an R2
checkpoint resets optimizer and LR; it is not exact training resume. This gate
must be visually reviewed before committing to longer training.

## What to read

* `gradient_probe.json`: only first rank0 train microbatch, weighted dLoss/dVelocity
  norms for RGB, LPIPS, and representation, plus representation/appearance ratio
  and cosine. This is a downstream interface check, not gradients of every DiT
  parameter. Router gradients are excluded. Weights are never auto-adjusted.
* `first_update.json`: actual proj_out.weight update after optimizer step1; not a
  global parameter-distance metric. Nonfinite or zero output-head update fails.
* `train_log.jsonl`: changing training batches; inspect trends rather than assume
  every loss should monotonically decrease.
* `val_log.jsonl`: includes `lpips`, existing teacher/GT/RGB/HF metrics, and
  `phase0_lpips` through `phase3_lpips`, plus phase RGB and counts. Phase statistics
  exclude local t0 to balance a 13-frame clip. They remain teacher-relative, NOT
  absolute quality or a test for shared teacher flicker.
* `best.json`: selected by RGB_L1 + 0.2 LPIPS under this recipe, excluding GT,
  router, velocity and HF diagnostics. Scores cannot be compared against V1's
  differently defined score. Step0 may remain best.
* `validation_visuals/step_XXXXXXXX/`: every frame 000..012, with LQ-up, GT,
  Stage-A+Original teacher, current Transformer+A1 student.

Compare step0 with trained outputs for texture shape, thin lines, fabric patterns
and hair. LPIPS improvement alone is not sufficient; avoid trading structure or
extra temporal instability for sharper appearance. Shared four-frame degradation
is explicitly outside this gate's promised scope.

Do NOT use the old cached-M8 val13 component script to evaluate a trained R2
Transformer: it would still display cached M8-A30k latents. R2's own validation
recomputes the current backbone, and the custom runner below loads R2 weights.

## Custom input

```bash
export INPUT=/data1/a/data/yuv_S_10bit/720p_mp4/S4_ChaseDragon2_1920x1080p444_10bit.mp4
export VIS_ROOT=outputs/custom/S4_ChaseDragon2_m10_r2

CUDA_VISIBLE_DEVICES=7 python scripts/inference_custom_components.py \
  --input "$INPUT" --output "$VIS_ROOT/r2_step1000" \
  --base-checkpoint checkpoints_prompt_free_no_time \
  --transformer-checkpoint outputs/b2b/m10_r2_perceptual_gate1000/checkpoints/step_00001000 \
  --transformer-type moe --decoder-type m9a1 \
  --decoder-checkpoint outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder \
  --upscale 3 --clip-len 24 --dit-overlap 0 --dtype bfloat16 \
  --attention-backend sdpa --png
```

Compare against M9-500G (M8-A30k+A1) on the SAME input and ROI, not a different
video. For early review change checkpoint to step_00000250 or step_00000500 and
use a distinct output directory. Do not call a high teacher-matching score proof
of removing the teacher's own four-frame phase issue.
