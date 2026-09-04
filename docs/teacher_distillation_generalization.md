# D0 teacher-distillation generalization training

This stage follows the validated two-sample gates. It keeps the endpoint teacher
velocity objective, teacher-relative GT guard, validation visuals, and BF16/FP16
behavior, while adding independent train/validation caches and exact
same-world-size resume for multi-GPU training.

## Source-aware cache planning

Before a large teacher-cache build, run `tools/plan_teacher_velocity_cache.py`.
It resolves the manifests on CPU and reports:

- loaded and clip-length-eligible manifest records;
- unique HR sources, using the complete resolved HR frame-path list rather than
  `record_uid` or basename;
- sources represented by several records, such as plain/text degradation variants;
- selected record/source coverage and a histogram of cached views per source;
- the exact DDP optimizer steps per epoch, dropped samples, and the recommended
  `max-steps` for a requested epoch count.

The planner uses the same selector and source identity code as the real cache
builder. Save its JSON output next to the cache command for reproducibility.

## Selected cache builder

Use `tools/build_teacher_velocity_cache_selected.py`. It selects deterministic
full-dataset `(record, view)` indices with one of:

- `--selection-mode all` for every deterministic view;
- `--selection-mode prefix` for compatibility diagnostics;
- `--selection-mode random --selection-seed N --max-samples K` for a reproducible
  random subset;
- `--selection-mode source_balanced --selection-seed N --max-samples K` to cover
  distinct resolved HR sources before taking additional degradation records or
  views from an already covered source.

New caches embed `source_uid`, the first/last HR path, `path_root`, unique-source
counts, selected-source counts, selected indices, and their SHA-256 digest. Tensor
files remain keyed by the original full-dataset `distillation_index`, so the
trainer can strictly validate every cached target. Velocity tensors are stored in
FP16 even when the teacher forward uses BF16.

## Cache audit

Run `tools/inspect_teacher_velocity_caches.py` before training. A repeated
`record_uid` is only a diagnostic warning. Training is blocked only when the
complete resolved HR source identity overlaps, or when a cache tensor is missing,
or when the teacher checkpoint, prompt embedding, ReAE, or endpoint timestep
differ. Do not permit true HR-source overlap in a generalization experiment.

## Validated baseline trainer

`tools/train_teacher_distillation_generalization_ddp.py` is the validated
synchronous-input baseline. It provides:

- `--resume latest` or an explicit checkpoint directory;
- arbitrary selected cache indices rather than a contiguous prefix;
- exact cursor restoration for `num-workers=0`;
- deterministic DDP sampler reconstruction and batch skipping;
- one RNG state per rank;
- strict run fingerprint validation;
- automatic train/validation HR-source overlap rejection.

Exact resume requires the same world size, cache metadata, sample selection,
model, dtype, local batch, gradient accumulation, seed, loss weights, and
optimizer configuration. `max-steps`, log cadence, validation cadence, visual
cadence, and save cadence may be changed. Validation RNG is captured and
restored, so changing validation frequency does not alter the later training
trajectory.

The validated four-GPU baseline uses per-GPU batch 1, accumulation 2, global
effective batch 8, BF16 runtime, velocity MSE/cosine weights 1.0/1.0, and GT guard
weights 0.10/0.05.

## Prefetched throughput trainer

`tools/train_teacher_distillation_throughput_ddp.py` wraps the validated trainer
and changes only the training input path. It does not duplicate or modify the
loss, model forward, optimizer, validation, checkpoint, or DDP reduction logic.
It adds:

- `--num-workers N` with positive worker counts;
- `--prefetch-factor N` for each worker;
- `--persistent-workers` for the active epoch iterator;
- training-time HQ decoding disabled by default, because the current velocity
  objective and GT guard use LR and HR but not HQ;
- `--load-train-hq` for a compatibility diagnostic;
- worker and HQ-loading settings in the strict resume fingerprint.

Validation keeps loading the complete triplet. The deterministic view identity is
still derived from `(view_seed, record_index, view_index)`, so worker scheduling
must not change the sampled temporal window, crop, or flips. Nevertheless, treat
positive-worker exact resume as a new gate: compare a continuous 0-to-75 run with
a 0-to-50 plus 50-to-75 resumed run before formal training.

Start with:

```text
num_workers       = 2 per rank
prefetch_factor    = 2
persistent_workers = true
load_train_hq      = false
```

Use `tools/benchmark_distillation_input_pipeline.py` to compare the synchronous
baseline and candidate worker settings on the same cache. The benchmark reports
triplet decode time separately from teacher-cache loading. Run enough warm-up
samples that worker startup and filesystem page-cache transients do not dominate.

## Formal-scale selection and stopping

The 1987-source baseline uses two deterministic views per source. Increasing to
four or eight views should add unique temporal/spatial views rather than merely
extend the number of epochs over the same fixed views. Select the final view count
after the prefetched input gate, because offline-cache construction should not be
expanded while image decoding leaves the GPUs idle.

Choose training length by the planner's exact optimizer steps per epoch rather
than by copying the 500-step gate. Select checkpoints primarily by validation
velocity relative L2, then compare velocity cosine, student/teacher and student/GT
PSNR/SSIM, GT guard violations, and fixed full-video visual results. Keep both the
distillation-best checkpoint and any nearby GT-favored checkpoint until the final
benchmark is complete.
