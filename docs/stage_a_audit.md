# Stage-A distillation audit

The Stage-A freeze uses two deliberately separate evaluation paths:

1. `tools/audit_stage_a_distillation.py` measures quality on the exact deterministic
   val13 views and exports fixed visual comparisons.
2. `tools/profile_stage_a_streaming_macs.py` measures architecture compute on one
   warmed-up, steady-state streaming MIDDLE chunk.
3. `tools/finalize_stage_a_audit_macs.py` merges the streaming compute result into
   the quality audit and rewrites the final Markdown report.

This separation is intentional. Quality must stay tied to the validated teacher
cache/view contract, while method complexity should reflect the real long-video
streaming graph rather than a 128x128 validation crop or hardware-specific timing.

## Primary compute convention

The canonical Stage-A compute table reports:

- canonical serialized parameter count;
- ReAE encoder GMACs per emitted output frame;
- DiT GMACs per emitted output frame;
- ReAE decoder GMACs per emitted output frame;
- total GMACs per emitted output frame;
- GFLOPs per output frame under the explicit convention `1 MAC = 2 FLOPs`;
- prompt-free student / conditional-teacher MAC ratio and percentage reduction.

The runtime counter includes executed `Linear`, `Conv1d/2d/3d`, shifted-window
self-attention QK/AV matrix products, and conditional Wan cross-attention QK/AV
matrix products. It deliberately excludes bias adds, elementwise operations,
normalization, activation, RoPE, indexing/gather/scatter, reshape/concat,
interpolation, pixel shuffle/unshuffle, image/video I/O and CPU preprocessing.

The canonical compute run uses:

- output resolution: `1920x1080`;
- internal padded resolution: `1920x1088`;
- `clip_len=24`;
- a warmed-up `MIDDLE` chunk;
- `dit_overlap=0`;
- SDPA attention backend;
- batch size 1 implicit in the streaming pipeline.

SDPA is required only so shifted-window self-attention is observable to the
runtime counter. The mathematical MAC count is backend-independent.

Wall-clock latency/FPS is retained as supplementary information only; it is not
the primary Stage-A complexity claim.

## Why canonical parameters are counted before inference preparation

SwiftVR `prepare_for_inference()` fuses Wan Q/K/V projections by creating fused
projection modules while retaining the original serialized Q/K/V modules. Directly
summing parameters after preparation therefore counts both canonical parameters
and runtime fusion copies. The streaming MAC profiler counts parameters before
preparation and counts MACs after preparation. This gives the intended pair:

- serialized/canonical architecture size;
- actually executed prepared inference computation.

## Attention counting sanity checks

Prompt-free/no-time SwiftVR shifted-window self-attention calls
`torch.nn.functional.scaled_dot_product_attention`, so the runtime counter patches
that call for QK and AV MACs.

Conditional teacher cross-attention follows `WanAttnProcessor ->
dispatch_attention_fn`. The counter patches that exact SwiftVR module-global call
and suppresses nested SDPA accounting while inside the dispatch wrapper, avoiding
both omission and double counting.

A canonical report is rejected if:

- self-attention is not observed;
- teacher cross-attention is not observed;
- prompt-free student unexpectedly executes cross-attention;
- encoder/DiT/decoder MAC totals are non-positive;
- component MACs do not sum exactly to the reported total;
- any counting diagnostic is non-empty.

## Recommended Stage-A freeze workflow

First run quality/visual audit. Replace the long-run path with the actual best
checkpoint.

```bash
export BASE=/data1/a/SwiftVR/checkpoints_prompt_free_no_time
export VAL_CACHE=/data1/a/SwiftVR/outputs/teacher_velocity_cache_val13
export STEP992=/data1/a/SwiftVR/outputs/distill_formal_v8_bs16_bf16/checkpoints/step_00000992
export LONG_CKPT=/path/to/long_run/best_checkpoint
export AUDIT=/data1/a/SwiftVR/outputs/stage_a_audit

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python tools/audit_stage_a_distillation.py \
  --base-checkpoint "$BASE" \
  --teacher-cache "$VAL_CACHE" \
  --manifest manifests/vsr_triplets_plain_val13_newserver.jsonl \
  --path-root . \
  --split val \
  --clip-length 13 \
  --crop-size 128 \
  --scale 3 \
  --views-per-record 1 \
  --view-seed 9000001 \
  --model init=base \
  --model step992="$STEP992" \
  --model long="$LONG_CKPT" \
  --dtype bfloat16 \
  --allow-dtype-mismatch \
  --attention-backend sdpa \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --visual-samples 2 \
  --visual-frame-indices 0,6,12 \
  --visual-video-fps 8 \
  --difference-scale 4 \
  --output-dir "$AUDIT"
```

Then profile real steady-state streaming compute. `INPUT_VIDEO` only needs to be
long enough to contain FIRST + the requested warm-up MIDDLE + one counted MIDDLE;
model MACs depend on the fixed output geometry, not video content.

```bash
export TEACHER=/data1/a/SwiftVR/checkpoints
export INPUT_VIDEO=/path/to/a/sufficiently_long_lq_video.mp4
export MAC_JSON=/data1/a/SwiftVR/outputs/stage_a_streaming_macs_1080p.json

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python tools/profile_stage_a_streaming_macs.py \
  --input "$INPUT_VIDEO" \
  --teacher-checkpoint "$TEACHER" \
  --student-checkpoint "$BASE" \
  --resolution 1920x1080 \
  --upscale 4 \
  --clip-len 24 \
  --dit-overlap 0 \
  --warmup-middle 1 \
  --dtype bfloat16 \
  --attention-backend sdpa \
  --output-json "$MAC_JSON"
```

Finally merge compute into the same audit report:

```bash
python tools/finalize_stage_a_audit_macs.py \
  --audit-dir "$AUDIT" \
  --streaming-macs-json "$MAC_JSON"
```

The final `stage_a_audit.md` contains the two presentation-facing tables:

- deterministic val13 quality;
- conditional teacher vs prompt-free Stage-A steady-state compute.

Prompt-free init, step992 and long-run intentionally have identical compute rows:
Stage A changes weights through distillation, not the prompt-free/no-time inference
architecture.

## Optional diagnostic profiler

`torch.utils.flop_counter` support in `audit_stage_a_distillation.py` and the older
`finalize_stage_a_audit_flops.py` are retained only as diagnostic cross-checks.
They are not the canonical Stage-A compute result because fused/custom attention
operators can be omitted by generic operator tracing.

## Stage-B handoff

After the final Stage-A report is accepted, preserve both step992 and the long-run
teacher-matching reference. The planned sequence is:

1. B1-A: replace only the heavy ReAE decoder with a Tiny Conditional Decoder;
2. B1-B: replace the ReAE encoder with a causal LR projection while preserving the
   existing latent shape/statistics contract;
3. B1-C: joint velocity/pixel recovery;
4. B2: DiT structural compression;
5. B3: distribution-aware/DMD-style prior recovery.

The same streaming MAC profiler should be extended rather than replaced for
Stage-B components so all compression results use one consistent compute
convention.
