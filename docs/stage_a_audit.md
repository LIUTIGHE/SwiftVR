# Stage-A distillation audit

`tools/audit_stage_a_distillation.py` freezes the quality/compute baseline before
Stage B structural compression. It evaluates all prompt-free/no-time students on
the exact deterministic validation views represented by an immutable teacher
velocity cache.

The cached conditional teacher is the quality reference. Student checkpoints are
loaded one at a time on top of the same folded prompt-free/no-time base, so the
audit does not require multiple 5B models in GPU memory simultaneously.

## What is reported

For every student:

- velocity relative L2 and cosine against the cached conditional teacher;
- student/teacher and student/GT PSNR, SSIM, MAE, RMSE;
- teacher/GT full-reference metrics;
- student/GT and teacher/GT temporal-difference MSE;
- teacher-relative GT pixel/temporal violation rates;
- optional LPIPS when `--lpips` is requested and the external `lpips` package is
  available;
- total/trainable parameters and delta-checkpoint size;
- measured ReAE encoder, prompt-free DiT, ReAE decoder, and end-to-end latency;
- effective clip FPS and peak allocated CUDA memory;
- optional operator-reported FLOPs via `torch.utils.flop_counter`.

FLOP counting is diagnostic only: fused/custom operators can be absent from the
counter. Wall-clock latency is the deployment-facing source of truth.

The output directory contains:

- `stage_a_audit.json`: machine-readable baseline for later Stage-B comparisons;
- `stage_a_audit.md`: compact quality/compute tables;
- `visuals/.../comparison.mp4`: LQ / GT / teacher / all student outputs;
- `visuals/.../differences.mp4`: per-student teacher/GT absolute differences;
- selected comparison/difference PNG frames.

## Recommended Stage-A freeze run

Run from the SwiftVR repository root. Replace `LONG_CKPT` with the actual long-run
best delta-checkpoint directory.

```bash
export BASE=/data1/a/SwiftVR/checkpoints_prompt_free_no_time
export VAL_CACHE=/data1/a/SwiftVR/outputs/teacher_velocity_cache_val13
export STEP992=/data1/a/SwiftVR/outputs/distill_formal_v8_bs16_bf16/checkpoints/step_00000992
export LONG_CKPT=/path/to/long_run/checkpoints/step_00170000
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
  --latency-warmup 3 \
  --latency-repeats 10 \
  --profile-flops \
  --output-dir "$AUDIT"
```

Do not add `--lpips` to the first gate unless the environment already contains the
optional package. LPIPS will become a deliberate Stage-B decoder-training
dependency rather than an implicit Stage-A requirement.

## Stage-B handoff

After the audit is accepted, preserve both the step-992 fidelity reference and the
long-run teacher-matching reference. The planned sequence is:

1. B1-A: replace only the heavy ReAE decoder with a Tiny Conditional Decoder;
2. B1-B: replace the ReAE encoder with a causal LR projection while preserving the
   existing latent shape/statistics contract;
3. B1-C: joint velocity/pixel recovery;
4. B2: DiT structural compression;
5. B3: distribution-aware/DMD-style prior recovery.
