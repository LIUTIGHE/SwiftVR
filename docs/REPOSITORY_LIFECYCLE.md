# SwiftVR research repository: file management policy

Updated 2026-10-13.

## Status

The October 13 attempted cleanup was **reversed**: all 19 previously removed
historical Tiny Decoder diagnostic/test files have been restored to
`feature/training-forward` with their original content. We keep the file-count
reduction off the agenda while the P2-50K experiment is in progress.

The earlier archive branch (`archive/pre-repo-cleanup-2026-10-13`, original
commit `3c7187589f25d337c29586660de9aacdcf5ea55d`) remains as a snapshot,
not as a required source for the restored files.

The `.gitignore` introduced during the cleanup **is retained** to avoid
accidentally committing new caches, weights, logs, and experiment outputs.
It does not remove or untrack existing files.

## Default rules for future contributions

1. **One-off operation → terminal command.** For checking logs, inspecting a
   checkpoint, plotting a quick statistic, repairing a one-time environment
   issue, or performing a single experiment audit, prefer a copyable shell
   command or a `python - <<'PY'` heredoc in chat. Do not create a tracked
   `.sh`/`.py` file unless reuse or reproducibility actually requires it.
2. **Reuse established entrypoints.** Prefer adding a parameter or subcommand
   to an existing maintained trainer/analysis tool. Avoid a succession of
   `v2`, `v3`, `fixed`, `final` duplicates for simple parameter changes.
3. **Create new code when responsibilities differ.** An independently trained
   model component, a materially different optimizer/checkpoint protocol,
   an important dataset contract, or a reusable integration/regression test
   justifies a new source file.
4. **Preserve experimental provenance.** Record the exact training settings,
   source commit, checkpoint lineage, metrics, and negative conclusions; do
   not delete or silently relabel experiments just to reduce file count.
5. **Keep large artifacts out of Git.** Store HR/LR/teacher caches, outputs,
   video comparisons, model weights, optimizer checkpoints, and logs under
   ignored `outputs/` or external storage. Never run broad cleanup against
   these directories simply to reorganize the repository.
6. **Review before historical archival.** Delete/archive only after checking
   dependencies, ensuring recoverability, and determining that simplification
   warrants the synchronization burden. Historical tools may remain when
   practical benefit is small.
7. **Avoid new documentation for trivial runs.** Maintain an existing runbook
   or experiment diary where possible instead of one Markdown file per run.
8. **Respect running jobs and offline servers.** Do not rename, move, remove,
   or replace code that a GPU training process may have imported or whose
   hashes are checked by resume logic. Do not require GitHub connectivity from
   the training machine for housekeeping. A remote GitHub update does not
   require immediate local changes.

## Active P1/P2 run compatibility

While P2-50K is running, do not alter the machine's local code, checkpoint
directories, data caches, or working-tree state for organization alone.
Current active entrypoints include:

- `tools/train_m8_p2_gan_continue.py`
- `tools/run_m8_p2_gan_50k.sh`
- `tools/train_m8_p2_video_gan.py`
- `tools/visual_check_m8_task.sh`
- `swiftvr/models/`, `swiftvr/streaming/`, `swiftvr/training/`,
  `swiftvr/data/`, and `scripts/inference*.py`

Never use `git reset --hard`, `git clean -fdx`, or blanket deletion of
`outputs/` as a file-management shortcut.

The server may be unable to access GitHub. No local update is required by
this policy change. When synchronization becomes possible and desirable,
review local modifications and the exact remote diff first; preserve running
experiments and any local-only changes.
