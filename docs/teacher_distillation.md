# Endpoint teacher distillation

This stage trains the zero-adapter prompt-free/no-time SwiftVR student directly
from the released conditional SwiftVR endpoint. It does **not** use GT L1.

The teacher is fixed to:

- empty prompt embedding;
- timestep `1000`;
- the same LQ input and ReAE encoder as the student.

The default objective is:

```text
normalized velocity MSE + velocity cosine loss
```

Optional teacher-output L1 and temporal losses can be enabled after the velocity
path passes the overfit gate.

## 1. Build a deterministic teacher cache

The cache and trainer must use identical manifests, crop settings, view count,
view seed, and flip probabilities. Each `(record, view)` receives a stable RNG
seed, so its temporal window, crop, and flips are independent of DataLoader
ordering and DDP rank.

Two-sample overfit cache:

```bash
CUDA_VISIBLE_DEVICES=6 \
python tools/build_teacher_velocity_cache.py \
  --reference-checkpoint /data1/a/SwiftVR/checkpoints \
  --manifest /data1/a/SwiftVR/manifests/vsr_triplets_plain_train_newserver.jsonl \
  --manifest /data1/a/SwiftVR/manifests/vsr_triplets_text_train_newserver.jsonl \
  --output-dir /data1/a/SwiftVR/outputs/teacher_velocity_cache_gate2 \
  --path-root /data1/a/SwiftVR \
  --split train \
  --clip-length 13 \
  --crop-size 128 \
  --scale 3 \
  --views-per-record 1 \
  --view-seed 0 \
  --horizontal-flip-probability 0 \
  --vertical-flip-probability 0 \
  --batch-size 1 \
  --max-samples 2 \
  --device cuda \
  --dtype float16 \
  --attention-backend sdpa \
  --timestep 1000 \
  --overwrite
```

The cache stores only teacher velocity tensors. RGB teacher outputs are decoded
from the cached velocity only for validation or optional output matching.

## 2. One-GPU overfit gate

`torchrun --nproc_per_node=1` uses the same DDP code path as the later multi-GPU
run.

```bash
CUDA_VISIBLE_DEVICES=6 \
torchrun --standalone --nproc_per_node=1 \
  tools/train_teacher_distillation_ddp.py \
  --checkpoint /data1/a/SwiftVR/checkpoints_prompt_free_no_time \
  --teacher-cache /data1/a/SwiftVR/outputs/teacher_velocity_cache_gate2 \
  --manifest /data1/a/SwiftVR/manifests/vsr_triplets_plain_train_newserver.jsonl \
  --manifest /data1/a/SwiftVR/manifests/vsr_triplets_text_train_newserver.jsonl \
  --val-teacher-cache /data1/a/SwiftVR/outputs/teacher_velocity_cache_gate2 \
  --val-manifest /data1/a/SwiftVR/manifests/vsr_triplets_plain_train_newserver.jsonl \
  --val-manifest /data1/a/SwiftVR/manifests/vsr_triplets_text_train_newserver.jsonl \
  --path-root /data1/a/SwiftVR \
  --split train \
  --val-split train \
  --clip-length 13 \
  --crop-size 128 \
  --val-crop-size 128 \
  --scale 3 \
  --views-per-record 1 \
  --view-seed 0 \
  --val-views-per-record 1 \
  --val-view-seed 0 \
  --horizontal-flip-probability 0 \
  --vertical-flip-probability 0 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --dtype auto \
  --attention-backend sdpa \
  --velocity-mse-weight 1 \
  --velocity-cosine-weight 1 \
  --output-l1-weight 0 \
  --output-temporal-weight 0 \
  --learning-rate 1e-5 \
  --weight-decay 0 \
  --gradient-accumulation-steps 1 \
  --expected-global-batch-size 1 \
  --max-steps 200 \
  --save-every 20 \
  --log-every 1 \
  --validate-every 20 \
  --validate-at-start \
  --output-dir /data1/a/SwiftVR/outputs/distill_gate2
```

Because the same cache is reused for validation, validation settings must match
that cache exactly. This is intentionally an overfit gate, not a generalization
measurement.

## 3. Gate criteria

The important metrics are:

```text
velocity_relative_l2       -> 0
velocity_cosine            -> 1
student_teacher_psnr       increases
student_teacher_ssim       increases
```

`student_gt_psnr/ssim` are diagnostics only and do not select the best
checkpoint. `best.json` selects the minimum validation velocity relative L2.

After the two-sample gate passes, build separate train and validation caches.
For the full train cache use several deterministic views per record, for example
`--views-per-record 4`; the trainer must use the same value and seed.

## 4. Optional output matching

After pure velocity matching is stable, enable small teacher-output terms:

```text
--output-l1-weight 0.1
--output-temporal-weight 0.1
```

These compare the student with the teacher output, not with GT. Do not introduce
GT L1 in this D0 phase.

## 5. Current gate limitation

The initial D0 trainer writes rank-0 delta checkpoints but deliberately does not
yet implement exact mid-run resume. Use it first for the short overfit and
small-data gates. Exact same-world-size resume will be added after the teacher
cache and loss path are validated on the target server.
