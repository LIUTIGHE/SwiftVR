# M10-LS: fixed-block L30 -> L20 layer selection

## Scope

Keep Original ReAE Encoder and M9-A1 E99 fixed. Only change which 20 source
Transformer blocks are retained from the trained M5 D1024/L30 MoE. Width, expert
configuration, active top-k, attention window size, adapters, and decoder
architecture are unchanged. All final candidates have the same major inference
MAC budget as M8-A: approximately 500.8 GFLOPs/frame for the complete A1 system
under the existing padded 1920x1088, 2 FLOPs/MAC convention. L19/L21 probes and
L30 parent evaluation are SEARCH work, not claimed to be 500G inference.

This implementation is a bounded discrete cross-region swap search plus matched
short-recovery reranking. It borrows the recovery-aware/less rigid group-budget
principles discussed in the project. It is NOT a TinyFusion/TinySR reproduction,
a differentiable search, or proof of globally optimal/lossless pruning. First and
last source layers remain protected. Region quotas and max-consecutive-prune
constraints are not imposed on NEW candidates.

M9-500G checkpoints, the old initializer, model code, inference code, and existing
trainers are not edited. R2 stays reverted. All five files in this change are new.

## Fixed lineage

The launcher reads the historical baseline from:

```
outputs/b2b/m8a_d1024_l20_init_from_m5/moe_depth_init_report.json
```

The recorded `source_checkpoint` supplies the trained M5 L30. Its configuration
and weight SHA256 must match that report; no random layers are inserted. A
relocated source can be passed with `--source-checkpoint /actual/M5/checkpoint`.
The historical project source is
`outputs/b2b/b2b_d1024_moe_ta_100k/checkpoints/step_00017500`, but the script uses
the report, not this hardcoded directory. Kept layers also come from its
`selection.kept_source_blocks`, not from a remembered list.

All candidates, including the old heuristic mask, reset to the SAME parent
weights before recovery. Routers and retained experts are copied, not randomly
reinitialized. The mature M8-A30k is evaluated only as an immutable deployment
reference; its much larger training budget is NOT an equal-budget mask control.

Frozen decoder:
`outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder`.
The teacher remains cached Stage-A D3072 step200k. Original ReAE decodes teacher
RGB; every candidate and parent uses the same frozen A1. Stage-A and M5 have
different roles: M5 supplies candidate parameters, Stage-A supplies the final
behavioral target. GT is diagnostic only.

## Data separation

Use existing Stage-A cache metadata and canonical deterministic Dataset checks.
The launcher reads Stage-A TRAIN cache from M3's run config `teacher_cache`, and
Stage-A VAL cache from M8-A's `val_teacher_cache`, exactly as in M10-V1. Do not
use the M8-A training TA cache or cached M8 restored latents.

Select three disjoint TRAIN-source groups, grouping by `sample_id` across all
views/variants and excluding every sample_id appearing in the official val cache:

* 8 probe views (8 sources): endpoint-only proposal screening;
* 128 adaptation views: round-robin source coverage, disjoint from both other sets;
* 16 rank views (16 sources): perceptual ranking after equal recovery.

Counts are defaults, not a dataset sufficiency claim. Names/indices/seed are
saved. Grouping cannot identify external aliases of the same physical source.
The original formal val13 is not rendered or scored until AFTER `selection.json`
is fixed. It is not used to choose masks or recovery steps. Do not repeatedly
tune the search against final val13 results.

Encoder input latents are encoded once for these views into `packets/`. Pixel
teacher/GT/LQ is stored only for rank/val views. These are derived working files,
not modifications to the teacher cache. The source/caches/A1 fingerprints are
checked before recovery and validation.

## Search and recovery semantics

1. Score the historical L20 by full endpoint NMSE versus Stage-A on probe views.
2. Probe removing each eligible kept layer (L19) and restoring each absent layer
   (L21). These scores ONLY propose promising drops/additions. Keep up to three
   of each, form up to nine paired swaps, and measure each actual L20 forward.
   No additive independence assumption is used for its final screen score.
3. Move to the better mask for at most two rounds. Even an immediately worse
   proposal can enter the final shortlist if it is among the evaluated best
   nonbaseline masks. Retain the historical mask plus up to three alternatives.
   This explores a small local neighborhood; good distant masks can be missed.
4. Recover each of the four candidates separately from original M5 weights with
   exactly the same shuffled batch schedule and optimization settings.
5. Rank the recovered deployment-precision models by rank-set LPIPS, with RGB
   MSE as a tie-breaker. If the old heuristic wins, keep that outcome; do not
   force a new mask. A winning scalar is not a visual-quality PASS.

Recovery loss is ONLY:

```
velocity_NMSE(v_student, v_StageA) + 0.01 * router_balance
```

There is NO RGB, LPIPS, HF or temporal training loss. Decoder is not in the
backward graph. LPIPS is evaluation-only, reusing the same pretrained AlexNet
loss as A1. Its existing torchvision cache is checked before construction, so
missing AlexNet weights fail locally rather than initiate a download.

Default recovery: single GPU, all Transformer parameters trainable, microbatch1,
accumulation4, 250 optimizer steps/candidate, LR1e-5 with 25-step warmup then
constant LR, FP32 master parameters/optimizer, BF16 autocast/checkpointed DiT,
SDPA, grad clipping1.0. 128 views and 250 steps are a cheap recoverability proxy,
not full training. Different masks may converge at different rates; longer
matched confirmation may still reorder them. Checkpoints are weights-only,
not exact optimizer/LR resume files. Formal retraining can start from the
chosen `init/` to avoid comparing unequal search-time adaptation histories.

### Shifted-window contract

The existing canonical runner assigns shifted/unshifted windows by COMPACT
layer index, not original source index. `compact_block_view` therefore physically
selects only the retained blocks and sets `_do_shift` by compact order during
screening, then restores the full parent on exit. Recovery materializes a real
L20 via the existing depth-copy helpers. Save/load/inference retain that same
compact-index convention. No zero-multiplied L30 is reported as an L20.

## Manual synchronization and tests

Copy via GitHub Web / VSCode SSH; no server git operation:

```
tools/m10_layer_search_core.py
tools/select_m10_layers.py
tools/run_m10_layer_selection.sh
tests/test_m10_layer_selection.py
docs/m10_layer_selection.md
```

```
cd /data1/a/SwiftVR
python -m unittest discover -s tests -p 'test_m10_layer_selection.py' -v
```

Core CPU tests exercise source-disjoint grouping, cross-region swaps, parent
restoration, compact parity, fixed recovery order, and synthetic recovery. The
real small-MoE view/materialization/save-load test needs full SwiftVR dependencies
and should NOT skip on the server. Local minimal-environment tests do not certify
real checkpoint/GPU training. No weights or new packages are bundled.

## Run

All three stages are sequential on one selected GPU. Do not use torchrun.

```bash
cd /data1/a/SwiftVR
export WORK=outputs/b2b/m10_layer_selection_v1

CUDA_VISIBLE_DEVICES=7 bash tools/run_m10_layer_selection.sh screen
```

This trains nothing. Read `screen_reference/{parent_L30,heuristic_init_L20,
mature_M8A30k}/metrics.json` and corresponding full-frame visual panels. These
are TRAIN rank views, not val13. Check whether the L30 parent actually contains
additional detail worth retaining. Early endpoint screening scores are not final
perceptual rankings.

Then run the same recovery budget for every shortlisted candidate:

```bash
CUDA_VISIBLE_DEVICES=7 bash tools/run_m10_layer_selection.sh recover \
  --steps 250 --accumulation 4 --learning-rate 1e-5 --warmup-steps 25
```

After ranking is locked:

```bash
CUDA_VISIBLE_DEVICES=7 bash tools/run_m10_layer_selection.sh validate
```

For a separate end-to-end smoke before the formal job:

```bash
WORK=outputs/b2b/m10_layer_selection_smoke CUDA_VISIBLE_DEVICES=7 \
  bash tools/run_m10_layer_selection.sh screen \
  --probe-views 2 --rank-views 2 --adapt-views 8 \
  --proposal-width 1 --swap-rounds 1 --candidates 2
WORK=outputs/b2b/m10_layer_selection_smoke CUDA_VISIBLE_DEVICES=7 \
  bash tools/run_m10_layer_selection.sh recover --steps 1 --accumulation 1 --warmup-steps 0
```

The smoke is not a selection result. Use a new WORK for formal search. Fresh
stage directories are required; interrupted stages do not silently resume or
overwrite partial checkpoints. Existing BASE/A1_DECODER/M8A/DEPTH_INIT and
STAGEA_TRAIN_CACHE/STAGEA_VAL_CACHE environment overrides remain available.

## Outputs and acceptance

```
run_config.json
screen_trace.jsonl
candidates.json
packets/{probe,adapt,rank}/*.pt
screen_reference/.../metrics.json
recovery/recovery_config.json
recovery/<candidate>/init/transformer/...
recovery/<candidate>/train_log.jsonl
recovery/<candidate>/step_00000250/transformer/...
recovery/<candidate>/{before,after}/metrics.json
selection.json
formal_val13/report.json
formal_val13/<model>/validation_visuals/step_00000000/...
```

Visual panels use every 13-frame view's frame0..12, not only 0/6/12. Playback FPS30
is not a source-cadence statement. LPIPS ranks candidates relative to Stage-A;
it cannot certify correct hallucinated detail or repair teacher-shared flicker.
Phase teacher MSE excludes local frame0; it is diagnostic, not phase correction.

Final val13 compares L30 parent, mature M8-A30k, historical-mask matched recovery,
and selected-mask matched recovery (omit duplicate if historical wins), always
through the SAME A1. Look first at new-mask vs historical-mask after the SAME
250-step recovery. Compare with mature M8-A only as a deployment-quality target.

A useful result retains more texture structure at matched compute/recovery and
does not materially worsen temporal behavior. Endpoint/LPIPS changes alone do
not establish success. An unsuccessful local shortlist does not prove block
architecture or all possible L20 choices have no remaining headroom.

Custom inference reuses `scripts/inference_custom_components.py` unchanged:
use `selection.json:selected.checkpoint` as `--transformer-checkpoint`,
`--transformer-type moe --decoder-type m9a1`, and the same A1 E99. Never run the
old cached-M8 component visualizer to evaluate a new selected Transformer.
