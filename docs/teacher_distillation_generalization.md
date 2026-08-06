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

## Resumable DDP trainer

Use `tools/train_teacher_distillation_generalization_ddp.py`. It inherits the
validated arguments from `train_teacher_distillation_ddp.py` and adds:

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

## Formal-scale selection and stopping

Prefer one deterministic view per record initially so large-scale expansion adds
source diversity rather than many crops from a small set of videos. When a cache
must be capped, use `source_balanced`. Choose training length by the planner's
exact optimizer steps per epoch rather than by copying the 500-step gate. Start
with eight effective epochs, validate every quarter to half epoch, and continue
only when the independent validation curve still improves.

Select checkpoints primarily by validation velocity relative L2, then compare
velocity cosine, student/teacher and student/GT PSNR/SSIM, GT guard violations,
and fixed full-video visual results. Keep both the distillation-best checkpoint
and any nearby GT-favored checkpoint until the final benchmark is complete.
