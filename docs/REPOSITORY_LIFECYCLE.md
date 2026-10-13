# SwiftVR research repository lifecycle (2026-10-13)

This is a **non-destructive, branch-level archival**, not an excuse to delete scientific history or server outputs.

## Snapshot before cleanup

Full pre-cleanup Git tree retained at:

- branch: `archive/pre-repo-cleanup-2026-10-13`
- commit: `3c7187589f25d337c29586660de9aacdcf5ea55d`

All archived files below remain recoverable, e.g.
`git show archive/pre-repo-cleanup-2026-10-13:tools/diagnose_tiny_decoder_layer_phase.py`.
Do not hard-reset a running server checkout; fetch/switch branches after training is stopped.

## Active P2-50K / P1 chain: never rename during run

- `swiftvr/models/`, `swiftvr/streaming/`, `swiftvr/pipeline*.py`, `swiftvr/training/`, `swiftvr/data/`;
- `scripts/inference.py`, `scripts/inference_custom_components.py` (including pre-existing `--stream`);
- `tools/train_m8_p2_gan_continue.py`, `tools/run_m8_p2_gan_50k.sh`,
  `tools/train_m8_p2_video_gan.py`, `tools/visual_check_m8_task.sh`,
  `tools/train_m8_task_posttrain.py` and current training/data/cache entrypoints;
- `tests/test_m8_p2_gan_continue.py`, `tests/test_m8_p2_video_gan.py`,
  `tests/test_m8_task_posttrain.py`, and all current runtime/streaming/data tests.

The ongoing 50K run verifies hashes for several source files and checkpoint sources.
**Do not sync code deletions or change import paths while it runs**. GitHub cleanup
is remote only; the server's existing Python processes and files remain unchanged.
Never run `git clean -fdx`, `git clean -X`, `rm -rf outputs`, or
`git reset --hard` as a repository tidy-up.

## Historical one-off probe archive (19 files)

The following obsolete **standalone Tiny Decoder exploratory diagnostics / visualizations**
and their directly importing tests were moved from the active branch to the
pre-cleanup snapshot. These were used in discarded pre-M9 conditional/resize-conv
explorations, not in current M8/A1/P2 code. They are retained through Git history
and the archival branch; rerun an old experiment from that branch, not the active one.

- `tools/diagnose_tiny_decoder_latent_shuffle.py`
- `tools/diagnose_tiny_decoder_condition_bypass_phase_bias.py`
- `tools/diagnose_tiny_decoder_direct_condition_phase_bias.py`
- `tools/diagnose_tiny_decoder_moderate_capacity_usage.py`
- `tools/diagnose_tiny_decoder_moderate_phase_bias.py`
- `tools/visualize_tiny_decoder_condition_bypass_variants.py`
- `tools/visualize_tiny_decoder_direct_condition_variants.py`
- `tools/visualize_tiny_decoder_moderate_variants.py`
- `tools/visualize_tiny_decoder_samples.py`
- `tools/diagnose_tiny_decoder_layer_phase.py`
- `tools/diagnose_tiny_decoder_phase_bias.py`
- `tools/diagnose_tiny_decoder_resize_conv_layer_phase.py`
- `tools/diagnose_tiny_decoder_variant_phase_bias.py`
- `tools/visualize_tiny_decoder_variants.py`
- `tests/test_tiny_decoder_layer_phase.py`
- `tests/test_tiny_decoder_phase_bias.py`
- `tests/test_tiny_decoder_resize_conv_layer_phase.py`
- `tests/test_tiny_decoder_variant_phase_bias.py`
- `tests/test_visualize_tiny_decoder_variants.py`

Other historical trainers, modules, and tests remain intentionally available
for reproducibility. Do not infer `tools/` entries are disposable merely from
their names. Additional pruning requires dependency search and an archive snapshot.

## Policy for new files

1. Reuse stable entrypoints and add flags/subcommands instead of creating
   `train_v2_v3_fixed_final.py` variants. A new trainer is justified when it
   changes checkpoint state/optimizer semantics and must keep older runs resumable.
2. A single-use observation, grep, profiling query, one-off file rewrite or
   server housekeeping: give a **terminal heredoc command** in chat; do not add
   a tracked repo script unless it is essential for repeated experiments or
   paper-grade reproduction.
3. Persist algorithm/model/data-contract code, validated multi-run entrypoints,
   regression tests, and concise experiment provenance. Prefer one maintained
   experiment log per family over repetitive ad hoc markdown.
4. Experiment checkpoints, teacher/HR caches, videos, PNG comparisons,
   logfiles, conda envs and one-off outputs belong under `outputs/` or
   local external storage and are ignored by Git. Tracked fixtures, if any,
   need explicit review.
5. After a failed completed experiment, record its conclusion and original
   command/config, and archive truly one-off tools before adding more.
6. Keep default upstream SwiftVR README and website assets intact; document
   research additions under `docs/` rather than rewriting original claims.
7. On a running GPU machine, **do not checkout/pull/change shared Python source**
   until the job and its worker processes have exited. Use non-destructive
   `git fetch` or work in a separate clean clone if necessary.

## Server sync: safe after the 50K run

First review local changes and compare without altering any run:
```bash
cd /data1/a/SwiftVR
git status --short
git fetch origin feature/training-forward archive/pre-repo-cleanup-2026-10-13
git diff --name-status HEAD origin/feature/training-forward
```

Only when training has stopped, and only when `git status --porcelain` is empty
(or after making a reviewed commit/stash/backup of local edits):
```bash
git switch feature/training-forward
git pull --ff-only origin feature/training-forward
```

This removes **tracked archived probe files only** in the working tree while
preserving their Git history. It does not touch `outputs/` or checkpoints.
Never replace this with a general `git clean`. If you want to restore any
historical tool:
```bash
git restore --source origin/archive/pre-repo-cleanup-2026-10-13 -- tools/diagnose_tiny_decoder_layer_phase.py
```
