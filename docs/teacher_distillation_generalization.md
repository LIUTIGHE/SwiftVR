# D0 small-data generalization training

This stage follows the validated two-sample gates. It keeps the endpoint teacher
velocity objective, teacher-relative GT guard, validation visuals, and BF16/FP16
behavior, while adding independent train/validation caches and exact
same-world-size resume for multi-GPU training.

## Selected cache builder

Use `tools/build_teacher_velocity_cache_selected.py` for small-data experiments.
It selects deterministic full-dataset `(record, view)` indices with one of:

- `--selection-mode all` for every deterministic view;
- `--selection-mode prefix` for compatibility diagnostics;
- `--selection-mode random --selection-seed N --max-samples K` for an unbiased,
  reproducible subset.

The cache metadata records the selected indices and their SHA-256 digest. Tensor
files remain keyed by the original full-dataset `distillation_index`, so the
trainer can strictly validate every cached target.

Recommended first experiment:

- train cache: 256 random views, `views-per-record=2`, train manifests only;
- validation cache: every record in the independent `val13` manifest,
  `views-per-record=1`, no flips;
- teacher endpoint: empty prompt, timestep 1000;
- teacher forward: BF16 on supported Pro 6000 hardware;
- cache tensor storage: FP16;
- loss reductions and optimizer state: FP32.

## Cache audit

Run `tools/inspect_teacher_velocity_caches.py` before training. It fails when a
cache tensor is missing, when train and validation share a `record_uid`, or when
the teacher checkpoint, prompt embedding, ReAE, or endpoint timestep differ.
Do not permit overlap in a generalization experiment.

## Resumable DDP trainer

Use `tools/train_teacher_distillation_generalization_ddp.py`. It inherits the
validated arguments from `train_teacher_distillation_ddp.py` and adds:

- `--resume latest` or an explicit checkpoint directory;
- arbitrary selected cache indices rather than a contiguous prefix;
- exact cursor restoration for `num-workers=0`;
- deterministic DDP sampler reconstruction and batch skipping;
- one RNG state per rank;
- strict run fingerprint validation;
- automatic train/validation overlap rejection.

Exact resume requires the same world size, cache metadata, sample selection,
model, dtype, local batch, gradient accumulation, seed, loss weights, and
optimizer configuration. `max-steps`, log cadence, validation cadence, visual
cadence, and save cadence may be changed. Validation RNG is captured and
restored, so changing validation frequency does not alter the later training
trajectory.

For the first four-GPU run, use per-GPU batch 1, accumulation 2, global effective
batch 8, BF16 runtime, GT guard weights 0.10/0.05, 500 maximum steps, validation
every 25 steps, and visual export every 50 steps.

## Selection criterion

The two-sample overfit thresholds are not expected on unseen clips. Select by
validation teacher behavior, especially decreasing velocity relative L2 and
increasing velocity cosine, then inspect fixed teacher/student videos. GT guard
violations should remain controlled without allowing the model to become a
pixel-regression solution.
