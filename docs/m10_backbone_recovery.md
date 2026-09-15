# M10 — fixed-budget Transformer HF / temporal recovery

## Scope and lineage

Start from M8-A D1024/L20 MoE `step_00030000`. Freeze the original ReAE encoder
and M9-A1 `epoch_099_step_00024552/tiny_decoder`. Only Transformer parameters
are optimized. Inference topology, active experts and the approximately
500.8 GFLOPs/frame deployment budget are unchanged (1920x1088 padded output,
2 FLOPs/MAC; encoder/Transformer totals use the existing rounded census).

The **direct teacher is Stage-A D3072 step200000**, not the D1536 teaching
assistant used during M8-A architecture training. The canonical Stage-A velocity
cache supplies `v_teacher`; Original ReAE decodes `z_lq - v_teacher` online under
`no_grad` to obtain teacher RGB. The student recomputes `v_student` every step
and decodes `z_lq - v_student` through **frozen A1 with input autograd enabled**.
The old cached M8-A `z_SR` is NOT a valid M10 training target/input replacement.
GT is used only for diagnostics and output geometry, never as a loss target.

Data views come from the Stage-A cache metadata, using the canonical deterministic
view Dataset and identity/hash checks. No crop, stride, augmentation or cache
content is changed. This does not establish the source video's physical FPS;
`--visual-fps 30` controls playback only. Existing 13-frame / 128 LR (384 HR)
views are retained when that is what the cache records.

## Objective and two matched gates

Let `S = A1(z_lq-v_student)`, `T = OriginalD(z_lq-v_StageA)` and
`H(x) = x-Gaussian5x5_sigma1(x)`. Filtering is spatial-only, per RGB channel,
with reflection padding. Differences and reductions are FP32.

```
L = 0.25 velocity_NMSE + 0.25 (1-velocity_cosine)
  + 1 RGB_L1(S,T) + 1 L1(H(S),H(T))
  + w_temporal L1(delta_time H(S), delta_time H(T))
  + 0.01 router_balance
```

`spatial` uses `w_temporal=0`; `temporal` uses `w_temporal=1`. Both start from the
same M8-A30k checkpoint with the same seed/data/LR. Their difference isolates
adding the HF temporal term. Improvement versus the old M8-A includes **both**
changing D1536-TA -> Stage-A supervision and adding RGB/HF losses; do not attribute
it solely to HF weighting without a later same-teacher velocity-only control.

Temporal supervision matches teacher changes, not zero changes. It is an
unwarped adjacent-frame error difference, not a motion-correspondence guarantee
or an independent long-term consistency objective. No GAN, LPIPS or optical-flow
weights are downloaded/required by M10.

## Manual synchronization and tests

Copy these five new files from GitHub Web into the school-server checkout:

```
swiftvr/training/m10_recovery.py
tools/train_m10_backbone_recovery_ddp.py
tools/run_m10_gate.sh
tests/test_m10_backbone_recovery.py
docs/m10_backbone_recovery.md
```

No previous trainer, checkpoint or inference file is changed. Run:

```bash
cd /data1/a/SwiftVR
python -m unittest discover -s tests -p 'test_m10_backbone_recovery.py' -v
```

`test_real_a1_input_backward` must run, not skip, in the full SwiftVR environment.
It checks real A1 input gradients with an active temporal adapter and checkpointed
decoding. Other tests cover teacher/GT gradient isolation, temporal semantics,
layout, high-pass filtering, cache lineage and output-directory protection.

The launcher reads the Stage-A train-cache path from
`outputs/b2a/formal100k_full/run_config.json:teacher_cache` and the Stage-A val-cache
path from `outputs/b2b/m8a_d1024_l20_gate200k/run_config.json:val_teacher_cache`.
It does not guess cache directory names. Explicit overrides are:

```bash
export STAGEA_TRAIN_CACHE=/actual/stagea/train/cache
export STAGEA_VAL_CACHE=/actual/stagea/val/cache
```

Do not point `STAGEA_TRAIN_CACHE` at M8-A's D1536 TA cache. The runner rejects
incorrect kinds, non-200k Stage-A teacher steps, different teacher weights or
ReAE fingerprints, and mismatched deterministic view identities.

## Run

First run one real GPU optimizer step, including validation before and after:

```bash
GPU_IDS=7 MAX_STEPS=1 WARMUP=0 ACCUM=1 \
OUT=outputs/b2b/m10_smoke \
bash tools/run_m10_gate.sh temporal --visual-samples 0
```

Then run the paired gates with identical effective batch sizes:

```bash
GPU_IDS=0,1,2,3 bash tools/run_m10_gate.sh spatial
GPU_IDS=0,1,2,3 bash tools/run_m10_gate.sh temporal
```

Defaults: BF16, SDPA, micro-batch 1/GPU, accumulation 4, global batch 16 on four
GPUs, LR 5e-6, warm-up 50 steps, 500 optimizer steps. Validation/visuals are
exported at 0/250/500. Runs use separate fresh output directories and never
replace the M8-A/A1 baseline. Change `GPU_IDS` to select available GPUs; keep
GPU count and accumulation identical between the two gates.

To use a different A1 run, override `A1_RUN` or `A1_DECODER`. To warm-start an
M10 checkpoint, pass `--student-init /actual/m10/checkpoints/step_00000500` and
use a new `OUT`. This is **weights-only warm start**, not optimizer/LR resume.
Gate snapshots follow the existing compact Transformer format; the fixed A1
checkpoint remains external and is recorded in the lineage. No HF network or
server-side git operation is involved.

## Readouts and visual review

```
run_config.json
train_log.jsonl
val_log.jsonl
best.json
summary.json
checkpoints/step_00000250/transformer/...
checkpoints/step_00000500/transformer/...
validation_visuals/step_00000000/...
validation_visuals/step_00000250/...
validation_visuals/step_00000500/...
```

The canonical exporter produces `comparison_frame_000/006/012.png`,
`comparison.mp4` and error panels for each selected view. Columns are LQ-up,
GT, **Stage-A+Original teacher**, and **current Transformer+A1 student**.
FPS=30 is a playback choice, not a claim about the source cadence. PNGs are the
native-resolution spatial reference; MP4s are viewing conveniences.

`student_teacher_*` measures the deployed path versus Stage-A+Original;
`backbone_original_teacher_*` decodes the current Transformer with OriginalD
at validation only. This helps identify improvement confined to compensating A1
versus broader recovery of Stage-A latent behavior. `student_gt_*` is diagnostic.
`hf_l1` and `hf_temporal_l1` are measured for both gates, even when the temporal
training weight is zero. Log fields `weighted_*` expose each loss contribution;
scalar shares are not gradient shares.

The same selection score (`RGB_L1 + HF_L1 + HF_temporal_L1`) ranks checkpoints
in both gates. It excludes GT and router-balance offsets. Step0 can remain the
best: in that case the baseline checkpoint is referenced rather than duplicated.
Training completion is labeled `COMPLETED_VISUAL_REVIEW_REQUIRED`, not quality
PASS. Do not compare A/B **total losses**, whose weights differ.

At step0 the earlier val13 comparison suggests approximately 30.10 dB for
`student_teacher_psnr`, 31.00 dB for `backbone_original_teacher_psnr`, and 24.90 dB
for `student_gt_psnr`. Small differences are possible because the earlier visual
script recomputed Stage-A while M10 consumes cached velocities. A large discrepancy
should be checked before training beyond the short gate.

Custom-input inference uses the existing runner unchanged:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/inference_custom_components.py \
  --input "$INPUT" --output "$VIS_ROOT/m10_temporal_step500" \
  --base-checkpoint checkpoints_prompt_free_no_time \
  --transformer-checkpoint outputs/b2b/m10_temporal_gate500/checkpoints/step_00000500 \
  --transformer-type moe --decoder-type m9a1 \
  --decoder-checkpoint outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder \
  --upscale 3 --clip-len 24 --dit-overlap 0 --dtype bfloat16 \
  --attention-backend sdpa --png
```

Compare the immutable M9-500G baseline, spatial gate and temporal gate on the
same dense-texture crops. Success means more teacher-like detail without
increased flicker or blur masking instability; an HF loss decrease alone is not
sufficient. Failure of one gate does not establish an architectural capacity limit.
