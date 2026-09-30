# Stage B1 structured-sparsity Tiny Decoder

Stage B1 first replaced the original ReAE decoder with a SwiftVR-specific Tiny
Conditional Decoder, then applied structured hidden-channel sparsity before the
one-time formal decoder training.

## Frozen compact topology

The dense Tiny Decoder used stage widths `192,128,64,32` and block counts
`2,2,2,1`. Structured sparsity is applied only inside causal MemBlocks. The
stage/interface widths, block counts, TGrow stages, condition projection,
transition convolutions, residual interfaces, temporal/spatial compression and
output PixelShuffle mapping remain unchanged.

A dense MemBlock uses

```
2C -> C -> C -> C
```

for its three 3x3 convolutions. The selected compact block uses

```
2C -> K -> K -> C
```

while preserving the same C-dimensional causal residual state. A gated same-width
supernet first learned channel importance; top-k channels were then materialized as
ordinary smaller dense Conv2d tensors. No runtime speedup is claimed from merely
zero-valued weights.

The measured candidates were:

- keep 0.75: internal widths `144,96,48,24`, 66.106 GMAC/frame;
- keep 0.55: internal widths `104,72,32,16`, 52.722 GMAC/frame;
- **keep 0.40: internal widths `80,48,24,16`, 47.945 GMAC/frame**.

The keep-0.40 candidate is frozen for formal Stage-B1 training. It showed no clear
single-sample recovery capacity cliff relative to keep-0.55 while reducing the
original 343.108-GMAC/frame ReAE decoder by about 86%.

## Formal z_SR caches

The formal train cache uses the full plain+text training set with the same v8
coverage used by Stage A:

```
1987 source records x 8 deterministic views = 15896 z_SR samples
```

with `view_seed=20260805`, horizontal flip probability 0.5 and no vertical flip.
The primary validation cache remains the fixed val13 x 1-view protocol with
`view_seed=9000001` and no flips.

The cached target is FP16 `z_SR`. The long-run Stage-A encoder and DiT therefore do
not run during formal Tiny Decoder training.

## Formal DDP trainer

`tools/train_tiny_decoder_formal_ddp.py` trains only the materialized keep-0.40
Tiny Decoder. A frozen ReAE decoder renders the same cached `z_SR` online to form
the decoder-teacher RGB target. The objective remains

```
MSE(Tiny, GT)
+ MSE(Tiny, ReAE(z_SR))
+ 2 * LPIPS(Tiny, GT)
+ 2 * LPIPS(Tiny, ReAE(z_SR)).
```

Temporal MSE remains a validation diagnostic rather than an optimization term.
The trainer performs epoch-boundary checkpointing/resume, validates on val13 after
every epoch, and selects the best checkpoint by minimum validation dual-objective
loss while retaining pixel, perceptual and temporal metrics.

The default formal recipe is BF16, four GPUs, local batch 4, learning rate 5e-5 and
8 epochs. If local batch 4 is too large, restart a fresh run with a smaller local
batch; batch size is part of the resume fingerprint and must not change within a
run.

Static check:

```bash
python -m py_compile \
  tools/train_tiny_decoder_formal_ddp.py \
  tests/test_tiny_decoder_formal.py

python -m unittest tests.test_tiny_decoder_formal -v
```

Formal launch:

```bash
export BASE=/data1/a/SwiftVR/checkpoints_prompt_free_no_time
export INIT_DECODER=/data1/a/SwiftVR/outputs/stage_b1_tiny_decoder_structured_sparse/keep_040/tiny_decoder
export TRAIN_CACHE=/data1/a/SwiftVR/outputs/stage_b1_zsr_cache_train_v8
export VAL_CACHE=/data1/a/SwiftVR/outputs/stage_b1_zsr_cache_val13
export OUT=/data1/a/SwiftVR/outputs/stage_b1_tiny_decoder_formal_keep040_bf16

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=4 tools/train_tiny_decoder_formal_ddp.py \
  --base-checkpoint "$BASE" \
  --init-decoder "$INIT_DECODER" \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  --manifest /data1/a/SwiftVR/manifests/vsr_triplets_plain_train_newserver.jsonl \
  --manifest /data1/a/SwiftVR/manifests/vsr_triplets_text_train_newserver.jsonl \
  --val-manifest /data1/a/SwiftVR/manifests/vsr_triplets_plain_val13_newserver.jsonl \
  --output-dir "$OUT" \
  --path-root /data1/a/SwiftVR \
  --split train \
  --val-split val \
  --clip-length 13 \
  --crop-size 128 \
  --val-crop-size 128 \
  --scale 3 \
  --views-per-record 8 \
  --view-seed 20260805 \
  --val-views-per-record 1 \
  --val-view-seed 9000001 \
  --horizontal-flip-probability 0.5 \
  --vertical-flip-probability 0 \
  --val-horizontal-flip-probability 0 \
  --val-vertical-flip-probability 0 \
  --batch-size 4 \
  --val-batch-size 1 \
  --num-workers 8 \
  --val-num-workers 4 \
  --prefetch-factor 2 \
  --persistent-workers \
  --pin-memory \
  --epochs 8 \
  --learning-rate 5e-5 \
  --gt-l2-weight 1 \
  --teacher-l2-weight 1 \
  --lpips-weight 2 \
  --lpips-microbatch-frames 16 \
  --max-grad-norm 1 \
  --dtype bfloat16 \
  --log-every 20 \
  --ddp-timeout-seconds 1800
```

Resume only from an epoch-boundary checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=4 tools/train_tiny_decoder_formal_ddp.py \
  <same arguments as the original run> \
  --resume latest
```

After formal Stage-B1 converges, freeze its best Tiny Decoder and move to aggressive
DiT compression. Encoder compression is intentionally deferred until after the DiT
budget is reduced, because changing the encoder changes the latent input contract.
