# Endpoint teacher distillation

This stage starts from the zero-adapter prompt-free/no-time SwiftVR student and
matches the released conditional SwiftVR endpoint:

- empty prompt embedding;
- timestep `1000`;
- the same LQ input and ReAE encoder;
- cached teacher velocity targets.

All development for this project is performed in the user's fork. The upstream
SwiftVR repository is treated as read-only. The server copy remains the final
working copy and does not depend on the GitHub repository remaining available.

## Objective

The core objective is:

```text
normalized velocity MSE + velocity cosine loss
```

All velocity, RGB, temporal, and normalization reductions are performed in
FP32 even when the frozen model forward uses FP16 or BF16 autocast.

### Teacher-relative GT guard

The default GT mode is `guard`. For each sample, the student is penalized only
when its GT pixel or temporal error exceeds the conditional teacher's error:

```text
pixel_guard = relu(student_GT_L1 - teacher_GT_L1) / teacher_GT_L1

temporal_guard =
  relu(student_GT_temporal_MSE - teacher_GT_temporal_MSE)
  / teacher_GT_temporal_MSE
```

This prevents GT supervision from continuously rewarding an increasingly
conservative conditional-mean solution after the student reaches teacher-level
fidelity.

Available modes:

```text
--gt-loss-mode none
--gt-loss-mode guard   # default
--gt-loss-mode direct  # ordinary student-vs-GT regression ablation
```

Default auxiliary weights:

```text
--gt-pixel-weight 0.10
--gt-temporal-weight 0.05
--gt-loss-every 1
```

Set `--gt-loss-every N` to decode RGB and apply the GT term every N optimizer
steps. Velocity distillation still runs at every step.

Optional teacher-output matching remains available:

```text
--output-l1-weight 0.0
--output-temporal-weight 0.0
```

These terms compare student RGB with teacher RGB, not GT.

## Precision policy

The runtime forward can use `float16` or `bfloat16`. The training precision
layout is:

```text
frozen ReAE / DiT forward: FP16 or BF16 autocast
trainable adapters:        FP32
AdamW states:              FP32
teacher cache storage:     FP16 by default
loss and metric reduction: FP32
```

GradScaler is enabled only for FP16. It is disabled for BF16 and FP32. BF16 is
rejected when `torch.cuda.is_bf16_supported()` is false.

A folded FP16 checkpoint may require `--allow-dtype-mismatch` when explicitly
running `--dtype bfloat16`. This is intentional: changing the runtime dtype must
be an explicit experiment rather than an accidental configuration change.

## Fixed validation visuals

At configured validation steps rank 0 writes:

```text
validation_visuals/step_XXXXXXXX/sample_YYY_<record_uid>/
  comparison_frame_000.png
  comparison_frame_006.png
  comparison_frame_012.png
  difference_frame_000.png
  difference_frame_006.png
  difference_frame_012.png
  comparison.mp4
  differences.mp4
```

The comparison order is:

```text
LQ bicubic | GT | Conditional teacher | Prompt-free student
```

Difference panels show:

```text
|Student-Teacher|
|Student-GT|
|Teacher-GT|
```

Controls:

```text
--visual-validation-samples 2
--visual-frame-indices 0,6,12
--visual-video-fps 8
--visual-difference-scale 4
--visualize-every 20
--no-validation-visuals
```

Selected PNG comparison and difference panels are also written to TensorBoard.
MP4 failure does not abort training; PNG outputs and a metadata error record are
kept.

## 1. Build deterministic teacher caches

Cache and trainer settings must match exactly: manifests, split, crop, clip
length, view count, seed, and flip probabilities.

Example validation cache:

```bash
CUDA_VISIBLE_DEVICES=6 \
python tools/build_teacher_velocity_cache.py \
  --reference-checkpoint /data1/a/SwiftVR/checkpoints \
  --manifest manifests/vsr_triplets_plain_val13_newserver.jsonl \
  --output-dir /data1/a/SwiftVR/outputs/teacher_velocity_cache_val \
  --path-root /data1/a/SwiftVR \
  --split val \
  --clip-length 13 \
  --crop-size 128 \
  --scale 3 \
  --views-per-record 1 \
  --view-seed 9000001 \
  --horizontal-flip-probability 0 \
  --vertical-flip-probability 0 \
  --batch-size 1 \
  --device cuda \
  --dtype float16 \
  --attention-backend sdpa \
  --timestep 1000
```

## 2. FP16 guard run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  tools/train_teacher_distillation_ddp.py \
  --checkpoint /data1/a/SwiftVR/checkpoints_prompt_free_no_time \
  --teacher-cache /data1/a/SwiftVR/outputs/teacher_velocity_cache_train \
  --manifest manifests/vsr_triplets_plain_train_newserver.jsonl \
  --manifest manifests/vsr_triplets_text_train_newserver.jsonl \
  --val-teacher-cache /data1/a/SwiftVR/outputs/teacher_velocity_cache_val \
  --val-manifest manifests/vsr_triplets_plain_val13_newserver.jsonl \
  --path-root /data1/a/SwiftVR \
  --split train \
  --val-split val \
  --clip-length 13 \
  --crop-size 128 \
  --val-crop-size 128 \
  --scale 3 \
  --views-per-record 4 \
  --view-seed 0 \
  --val-views-per-record 1 \
  --val-view-seed 9000001 \
  --horizontal-flip-probability 0.5 \
  --vertical-flip-probability 0 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --dtype float16 \
  --attention-backend sdpa \
  --velocity-mse-weight 1 \
  --velocity-cosine-weight 1 \
  --gt-loss-mode guard \
  --gt-pixel-weight 0.10 \
  --gt-temporal-weight 0.05 \
  --gt-loss-every 1 \
  --learning-rate 1e-5 \
  --gradient-accumulation-steps 1 \
  --expected-global-batch-size 4 \
  --max-steps 1000 \
  --save-every 100 \
  --log-every 1 \
  --validate-every 50 \
  --validate-at-start \
  --visual-validation-samples 2 \
  --visual-frame-indices 0,6,12 \
  --visualize-every 50 \
  --output-dir /data1/a/SwiftVR/outputs/distill_guard_fp16
```

## 3. BF16 A/B run

Use the same cache, data order, seed, and step count. Change only:

```text
--dtype bfloat16
--allow-dtype-mismatch
--output-dir /data1/a/SwiftVR/outputs/distill_guard_bf16
```

Compare:

```text
velocity_relative_l2
velocity_cosine
gradient_norm
step_seconds
peak_allocated_gb_per_rank
student-teacher visuals
```

The run config records runtime dtype, cache storage dtype, loss reduction dtype,
and whether GradScaler was enabled.

## Validation and checkpoint selection

Primary selection remains minimum validation `velocity_relative_l2`. Validation
also records:

```text
student_teacher/*
student_gt/*
teacher_gt/*
gt_constraint/*
```

GT metrics diagnose fidelity and guard activation; they do not replace teacher
behaviour as the primary selection criterion.

## Current limitation

The trainer writes rank-0 delta checkpoints but does not yet implement exact
mid-run resume. Use a fresh output directory for each run.
