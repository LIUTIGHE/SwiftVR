# M10-LS confirmation: two fixed masks, expanded data, equal recovery budgets

## Purpose and scope

Confirm whether the short-run advantage of `swap_bd14890ed632` over `heuristic`
persists with more training. Do not run another layer search. Both start again
from the parent-derived `init_checkpoint` recorded in the locked `selection.json`,
not from the 250-step search-adapted checkpoints and not from mature M8-A30k.
The selected ID/masks are read from that file; the implementation does not
hardcode the chosen source-layer indices.

Only Transformer weights train. Original ReAE Encoder, A1 E99, layer masks,
D1024/L20 structure, experts and active top-k remain fixed. The direct target is
the existing Stage-A D3072 step200k velocity cache. Loss remains exactly:

```
velocity_NMSE + 0.01 * router_balance
```

No RGB, LPIPS, HF, temporal, GT or decoder loss enters backward. LPIPS and
RGB/phase metrics are evaluation only. Final deployment uses the unchanged
approximately 500.8 GFLOPs/frame system census. This experiment is not designed
to repair the teacher-shared four-frame phase artifact.

## Larger training data, without contaminating heldout sources

Use ALL available Stage-A TRAIN-cache views after excluding every source named
in the earlier probe and rank sets and in the official validation cache. Group
by `sample_id` across views/variants, not by individual crop/record index. The
128 adaptation views remain eligible, but no longer define the training set.
Actual retained source/view counts are printed and recorded; source aliases
outside metadata cannot be detected. This uses only existing cached views and
does not introduce new augmentations, longer clips or different frame cadence.

Encoder latents are generated lazily, once per visited view, under no_grad and
saved to the NEW confirmation directory. The two candidates share these packets.
The existing teacher cache is never edited. The first candidate can spend more
wall time encoding than the second; optimizer steps and sample exposures are
still matched. Cache hits/misses preserve CPU/CUDA RNG state. Missing local
weights/cache paths fail locally; no server git or new downloads are used.

Data schedule is a deterministic shuffled sequence with full epoch coverage
before repeating. Both masks see the same exact view order and effective batch.
Rank16 and val13 reuse the completed search's teacher/input packets. These are
already monitored validation sets, not newly untouched test sets. They are not
used for parameter gradients, new mask selection, or per-candidate early stopping.

## Run (one GPU, candidates sequentially; not torchrun)

Manually sync these five files from the fork:

```
tools/select_m10_layers.py               # small opt-in command dispatch change
tools/run_m10_layer_selection.sh         # accepts confirm
tools/m10_layer_confirmation.py          # new confirmation implementation
tests/test_m10_layer_confirmation.py     # new CPU contracts
docs/m10_layer_confirmation.md
```

Keep all earlier M10-LS dependencies and the completed work directory, including
`selection.json`, `packets/rank`, `formal_val13/packets`, `formal_val13/report.json`
and the two `recovery/*/init` checkpoints. There is no need to rerun A/B/C.

```bash
cd /data1/a/SwiftVR
python -m unittest discover -s tests -p 'test_m10_layer*.py' -v

export WORK=outputs/b2b/m10_layer_selection_v1
export CONFIRM_OUT=outputs/b2b/m10_layer_confirmation_v1

CUDA_VISIBLE_DEVICES=7 bash tools/run_m10_layer_selection.sh confirm \
  --output-dir "$CONFIRM_OUT" \
  --steps 2000 --accumulation 4 --learning-rate 1e-5 \
  --warmup-steps 25 --eval-every 500 --visual-samples 13
```

Default recipe deliberately keeps the short recovery's microbatch1,
accumulation4 (global batch4), LR1e-5, warmup25 then CONSTANT LR, grad clip1.0,
BF16 forward and FP32 master parameters/AdamW moments. Increasing the data pool
and update budget are the only experimental changes shared by both candidates.
Validation/deployment snapshots occur at 0/500/1000/1500/2000 (step0 references
existing init instead of duplicating it). The heuristic finishes first, then
selected starts from its own init; compare matching saved steps only.

2000 steps means 8000 sample-view presentations PER candidate, not necessarily
one full epoch or convergence. `data_passes_equivalent` divides presentations
by actual eligible views. It is a decision point, not an assertion of sufficient
optimization. Longer matched training can be requested with --resume.

For a separate GPU smoke (tiny optimizer budget, full validation infrastructure):

```bash
CUDA_VISIBLE_DEVICES=7 bash tools/run_m10_layer_selection.sh confirm \
  --output-dir outputs/b2b/m10_layer_confirmation_smoke \
  --steps 1 --accumulation 1 --learning-rate 1e-5 \
  --warmup-steps 0 --eval-every 1 --visual-samples 0
```

Use a fresh output for the real experiment; smoke is not a quality result.

## Resume and extend without resetting the optimizer

Unlike the earlier short-search weights-only snapshots, confirmation ALSO saves
one rolling `resume.pt` per candidate. It contains FP32 master model weights,
optimizer moments/step, CPU/CUDA RNG and the data position implied by global
step. The original recipe/inputs/selection/checkpoint fingerprints are locked.
Only the total --steps can change on resume; GPU ID can change. The shuffled
data schedule is prefix-stable and LR does not depend on total training length.
For identical hardware/software deterministic execution the CPU test confirms
uninterrupted/resumed weights match exactly. Bitwise GPU determinism across
hardware or nondeterministic kernels is not promised.

```bash
# Interrupted run: restore the last saved step for each candidate.
CUDA_VISIBLE_DEVICES=7 bash tools/run_m10_layer_selection.sh confirm \
  --output-dir "$CONFIRM_OUT" --steps 2000 --accumulation 4 \
  --learning-rate 1e-5 --warmup-steps 25 --eval-every 500 \
  --visual-samples 13 --resume

# Subsequent matched extension, when justified by the curves/visuals:
CUDA_VISIBLE_DEVICES=7 bash tools/run_m10_layer_selection.sh confirm \
  --output-dir "$CONFIRM_OUT" --steps 5000 --accumulation 4 \
  --learning-rate 1e-5 --warmup-steps 25 --eval-every 500 \
  --visual-samples 13 --resume
```

Do not change accumulation, LR, warmup, evaluation interval or visual count while
resuming; a mismatch is rejected. Save/evaluation work preserves training RNG.
Progress after the latest saved checkpoint is replayed following an interruption.
The rolling state is training-only and larger than the inference weights;
retain it to continue. Never load it as an inference checkpoint.

Intermediate snapshots copy live weights to BF16 on CPU without converting the
live FP32 parameters. Evaluation loads that snapshot as a separate model, so the
reported metrics match the exported deployment-precision checkpoint. Saving or
evaluation does not repeatedly round the running training weights to BF16.

## Outputs and comparison

```
confirmation_plan.json                 # locked pair, exclusions, view IDs, recipe
latent_packets/<original_view_id>.pt   # derived encoder/teacher-latent packets
heuristic/
  resume.pt                           # rolling full training state
  train_log.jsonl
  checkpoints/step_00000500/transformer/...
  checkpoints/step_00001000/transformer/...
  checkpoints/step_00002000/transformer/...
  evaluations/step_00000000/{rank16,val13}/...
  evaluations/step_00000500/{rank16,val13}/...
  evaluations/step_00001000/{rank16,val13}/...
  evaluations/step_00002000/{rank16,val13}/...
selected/                             # same layout, ID recorded in plan
comparison.json                       # intersection of evaluated steps only
```

Both masks export all 13 frames for each chosen val view via the canonical
visual exporter: LQ-up | GT | Stage-A+Original | candidate+A1. Playback30fps is
not a source-cadence claim. Rank16 visuals are disabled to avoid duplicate
panels; all rank16 metrics are still computed. Comparison records both absolute
results and selected-minus-heuristic LPIPS, RGB MSE and PSNR at the SAME step.
No independent best-checkpoint cherry-picking or automatic new milestone.
Parent_L30 and mature_M8A30k reference results remain in the earlier stage-C
`formal_val13/report.json`, whose fingerprint is locked by the new plan.

Custom inference uses the unchanged runner. Compare:

```
$CONFIRM_OUT/heuristic/checkpoints/step_00002000
$CONFIRM_OUT/selected/checkpoints/step_00002000
```

Use the same A1 E99, input video, clip_len24, dit_overlap0, BF16, SDPA, crop and
frame IDs. Judge texture/edge fidelity and temporal behavior, not GT PSNR alone.
Shared teacher flicker and short val13 clips still limit temporal conclusions.
Read `comparison.json` plus both `train_log.jsonl` files before extending the run.
Selection results, model references and old output directories remain immutable.
