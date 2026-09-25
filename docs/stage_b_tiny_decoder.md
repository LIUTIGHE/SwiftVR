# Stage B1 — Tiny Conditional Decoder

Stage B1 replaces only the heavy ReAE decoder. The ReAE encoder and the long-run
prompt-free/no-time DiT remain frozen. This isolates decoder compression before the
project moves directly to aggressive DiT compression.

## Architecture

The initial decoder adapts the conditional-decoding idea of FlashVSR to SwiftVR's
actual latent contract rather than copying its Wan-VAE geometry.

SwiftVR ReAE uses:

- 48 latent channels;
- temporal compression factor 4;
- spatial compression factor 16;
- three causal warm-up output frames.

For a training clip with `T=4k+1`, the RGB condition is prefix-padded with three
copies of frame 0, packed by `4 x 16 x 16`, projected from 3072 packed RGB channels
to 32 channels with a 1x1 convolution, and concatenated with the 48-channel SR
latent. Prefix padding is deliberate: after 4x temporal growth and removal of the
three decoder warm-up frames, the first retained output aligns with true frame 0.

The default decoder widths are `192,128,64,32` with `2,2,2,1` causal MemBlocks.
It retains the cheap temporal `TGrow` operations used by ReAE and restores spatial
resolution with three 2x upsampling stages plus the final 2x pixel shuffle.

## Objective

For one source latent `z_SR`, the frozen ReAE decoder and the tiny decoder see the
same latent. The initial objective is

```
L = MSE(x_tiny, x_GT)
  + MSE(x_tiny, x_ReAE(z_SR))
  + 2 * LPIPS(x_tiny, x_GT)
  + 2 * LPIPS(x_tiny, x_ReAE(z_SR)).
```

Temporal MSE is logged but is intentionally not another optimization term in the
first gate. This keeps the first experiment focused on the FlashVSR-style dual
pixel/perceptual supervision. LPIPS is an explicit Stage-B dependency in
`requirements-stage-b.txt`; the architecture/MAC tests do not require it.

## Gate 1: CPU/unit tests

Run from the repository root:

```bash
python -m py_compile \
  swiftvr/models/tiny_conditional_decoder.py \
  swiftvr/streaming/tiny_decoder.py \
  swiftvr/training/tiny_decoder.py \
  tools/smoke_tiny_conditional_decoder.py \
  tools/profile_tiny_decoder_streaming_macs.py \
  tests/test_tiny_conditional_decoder.py

python -m unittest tests.test_tiny_conditional_decoder -v
```

The tests check condition packing, `4k+1` whole-clip shape recovery, condition
sensitivity, first/middle streaming frame counts, stream reset, checkpoint round-trip,
and the exact dual-loss coefficients.

## Gate 2: one-sample overfit

Use the actual long-run Stage-A best delta checkpoint, not automatically step 170000
unless it is the best checkpoint.

```bash
export BASE=/data1/a/SwiftVR/checkpoints_prompt_free_no_time
export LONG=/path/to/long_run/checkpoints/step_XXXXXXXX
export OUT=/data1/a/SwiftVR/outputs/stage_b1_tiny_decoder_smoke

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python tools/smoke_tiny_conditional_decoder.py \
  --base-checkpoint "$BASE" \
  --source-checkpoint "$LONG" \
  --manifest /data1/a/SwiftVR/manifests/vsr_triplets_plain_train_newserver.jsonl \
  --path-root /data1/a/SwiftVR \
  --split train \
  --clip-length 13 \
  --crop-size 128 \
  --scale 3 \
  --view-seed 20260811 \
  --sample-index 0 \
  --dtype bfloat16 \
  --allow-dtype-mismatch \
  --attention-backend sdpa \
  --condition-channels 32 \
  --decoder-channels 192,128,64,32 \
  --blocks-per-stage 2,2,2,1 \
  --steps 100 \
  --learning-rate 1e-4 \
  --gt-l2-weight 1 \
  --teacher-l2-weight 1 \
  --lpips-weight 2 \
  --output-dir "$OUT"
```

If the optional LPIPS package is not installed yet, use `--lpips-weight 0` only for
an architecture/backprop smoke. Do not treat that MSE-only run as the B1 quality
result.

The output contains an independently reloadable `tiny_decoder/`, `summary.json`,
and initial/final comparison PNGs for first/middle/last frames.

## Gate 3: 1080p steady-state MACs

The MAC profiler uses the same 1920x1088 internal geometry, 24-frame MIDDLE chunk,
SDPA backend, and `1 MAC = 2 FLOPs` reporting convention as the frozen Stage-A
profile.

```bash
export STAGEA_MAC=/data1/a/SwiftVR/outputs/stage_a_streaming_macs_1080p.json
export B1_MAC=/data1/a/SwiftVR/outputs/stage_b1_tiny_decoder_macs_1080p.json

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python tools/profile_tiny_decoder_streaming_macs.py \
  --input /data1/a/SwiftVR/assets/27_1_lq.mp4 \
  --student-checkpoint "$BASE" \
  --tiny-decoder "$OUT/tiny_decoder" \
  --resolution 1920x1080 \
  --upscale 4 \
  --clip-len 24 \
  --dit-overlap 0 \
  --warmup-middle 1 \
  --dtype bfloat16 \
  --attention-backend sdpa \
  --stage-a-macs-json "$STAGEA_MAC" \
  --output-json "$B1_MAC"
```

Stage-A reference at the same geometry is:

- encoder: 42.278 GMAC/frame;
- prompt-free/no-time DiT: 2182.431 GMAC/frame;
- ReAE decoder: 343.108 GMAC/frame;
- total: 2567.817 GMAC/frame = 5135.635 GFLOPs/frame under 2 FLOPs/MAC.

The B1 gate should show a substantial decoder-MAC reduction. Because the DiT remains
~85% of Stage-A compute, do not spend iterations chasing a small final decoder gain
once the tiny decoder has a reasonable quality/reconstruction result. The next main
stage is aggressive DiT compression toward the 500--800 GFLOPs/frame total target.
