# Stage B1 structured-sparsity Tiny Decoder gate

This gate runs **before** the formal z_SR cache and decoder DDP training.  The dense
Tiny Conditional Decoder already passed one-sample L2/LPIPS overfit and measured
82.002 GMAC/frame at the frozen 1080p protocol.  The purpose here is to reduce that
compute again without deleting causal blocks or TGrow stages.

## What is pruned

Only the hidden channels inside each Tiny Decoder MemBlock are structured-pruned.
The stage/interface widths remain `192,128,64,32`, block counts remain `2,2,2,1`,
and condition projection, transition convolutions, TGrow operations, residual
interfaces, temporal/spatial compression, and output PixelShuffle mapping are kept.

A dense MemBlock uses approximately

```
2C -> C -> C -> C
```

for its three 3x3 convolutions.  The materialized compact block uses

```
2C -> K -> K -> C
```

with the same residual C-dimensional state.  For `r=K/C`, the dominant per-block
channel-product term changes from `4 C^2` to `(3r+r^2) C^2`.

The implementation deliberately does not claim speedup from zero-valued weights.
A gated same-width supernet learns channel importance, then top-k channels are
sliced into ordinary smaller dense Conv2d tensors.  The standard Stage-A MAC
counter therefore measures the actual compact topology.

## Candidate ratios

The first gate exports three candidates, aligning hidden widths to multiples of 8:

- keep 0.75: `144,96,48,24`;
- keep 0.55: rounded hardware-aligned widths derived by the tool;
- keep 0.40: `80,48,24,16`.

No ratio is selected before the measured quality/MAC tradeoff is available.

## Static gate

Run from the repository root:

```bash
python -m py_compile \
  swiftvr/models/tiny_conditional_decoder.py \
  swiftvr/models/tiny_decoder_sparsity.py \
  tools/smoke_structured_sparse_tiny_decoder.py \
  tests/test_tiny_decoder_sparsity.py

python -m unittest \
  tests.test_tiny_conditional_decoder \
  tests.test_tiny_decoder_sparsity -v
```

The tests cover exact dense-to-unit-gate conversion, legacy v1 dense checkpoint
loading, gate gradients, channel alignment, compact materialization, forward/backward,
streaming first/middle semantics, parameter reduction, and compact save/reload.

## One-sample structured-sparsity gate

Use the same deterministic source sample and long-run Stage-A checkpoint that were
used for the successful LPIPS smoke.  `--dense-decoder` should point to that
successful LPIPS smoke's `tiny_decoder/` directory.

```bash
export BASE=/data1/a/SwiftVR/checkpoints_prompt_free_no_time
export LONG=/path/to/long_run/checkpoints/step_XXXXXXXX
export DENSE=/data1/a/SwiftVR/outputs/stage_b1_tiny_decoder_smoke_lpips/tiny_decoder
export SPARSE_OUT=/data1/a/SwiftVR/outputs/stage_b1_tiny_decoder_structured_sparse

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python tools/smoke_structured_sparse_tiny_decoder.py \
  --base-checkpoint "$BASE" \
  --source-checkpoint "$LONG" \
  --dense-decoder "$DENSE" \
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
  --gate-steps 50 \
  --gate-learning-rate 5e-3 \
  --sparsity-weight 0.05 \
  --keep-ratios 0.75,0.55,0.40 \
  --channel-multiple 8 \
  --recovery-steps 50 \
  --recovery-learning-rate 5e-5 \
  --gt-l2-weight 1 \
  --teacher-l2-weight 1 \
  --lpips-weight 2 \
  --max-grad-norm 1 \
  --output-dir "$SPARSE_OUT"
```

The run first checks that dense -> sparse-supernet initialization has
`dense_sparse_max_abs <= 1e-5`.  Only gate parameters are updated during gate
learning.  Each top-k compact candidate is then recovered independently for 50
steps and saved under `keep_075/`, `keep_055/`, and `keep_040/`.

## 1080p MAC comparison

Profile the three materialized checkpoints with the existing frozen protocol:

```bash
export STAGEA_MAC=/data1/a/SwiftVR/outputs/stage_a_streaming_macs_1080p.json

for K in 075 055 040; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  python tools/profile_tiny_decoder_streaming_macs.py \
    --input /data1/a/SwiftVR/assets/27_1_lq.mp4 \
    --student-checkpoint "$BASE" \
    --tiny-decoder "$SPARSE_OUT/keep_${K}/tiny_decoder" \
    --resolution 1920x1080 \
    --upscale 4 \
    --clip-len 24 \
    --dit-overlap 0 \
    --warmup-middle 1 \
    --dtype bfloat16 \
    --attention-backend sdpa \
    --stage-a-macs-json "$STAGEA_MAC" \
    --output-json "/data1/a/SwiftVR/outputs/stage_b1_sparse_keep_${K}_macs_1080p.json"
done
```

The dense Tiny Decoder reference is 82.002 GMAC/frame.  A candidate is interesting
only if the profiler shows a real decoder-MAC reduction and its short recovery does
not collapse GT/teacher L2 or LPIPS.  The intended range is roughly 35--50
GMAC/frame, but the measured quality/MAC Pareto point decides which candidate enters
the one-time formal cache/training stage.

Do not build the formal 3974-view z_SR cache until this gate selects the final
compact topology.
