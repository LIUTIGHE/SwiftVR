# M12 Roadmap — 500–600G System Milestone, Flicker, and Final Quality Recovery

**Status:** M12-1 native-video **qualitative results reported by user on 2026-11-03** (see below); detailed server test logs and integrated compute report are not available here. Decoder training/weights and existing model source are unchanged. Next: **M12-2 temporal stability first, texture fidelity in parallel**.
**Repository:** `LIUTIGHE/SwiftVR` fork ONLY, branch `research/m12-transformer-quality-recovery`, PR to the fork's own `main`. NEVER edit, push or PR upstream `H-oliday/SwiftVR`.
**Previous milestone:** `docs/M11B_TRANSFORMER_V1_MILESTONE.md` and merged PR #4.

## 1. The actual end goal and order of work

Our goal is **acceptable visual detail and temporal stability while reaching a WHOLE-SYSTEM 500–600 GFLOPs per emitted RGB frame at the canonical 1080p-class output geometry**. FLOPs are the present primary engineering budget; throughput and CUDA latency are secondary measurements, not a requirement to prematurely discard a good candidate. Use **1 MAC = 2 FLOPs** consistently; explicitly state the 1920×1088 internal compute geometry. The 720p→2160p full-origin visual tests are a separate, more expensive workload and must NEVER inherit the 1080p-class GFLOPs label.

**Stage order is not "stop decoder compression, improve Transformer first".** Instead:
1. **Freeze M11-B bootstrap30k as the current compressed Transformer anchor.**
2. **Investigate the existing two-frame luminance flicker early** to locate its likely source/interaction and avoid repeating it in decoder distillation; do not assume a particular layer caused it.
3. **Actively compress / adapt the ReAE decoder** while M11-B stays frozen, prioritizing visually faithful high-frequency detail and avoiding any *additional* temporal artifacts. First test the ALREADY TRAINED Slim100 before building a new network.
4. **Reach and record a stable 500–600G full-system candidate** with an explicit quality gate. Flicker can be corrected either during decoder adaptation or via a small targeted correction, depending on the diagnostic; do not force a large detour for exhaustive root-cause proofs.
5. **Only then refine remaining Transformer fine detail and, if needed, jointly adapt Transformer and compressed Decoder** within the same deployment budget.

The user-observed controlled comparison was: M11-B30k + Original ReAE is only slightly worse than full Original SwiftVR, and severe phase-dependent texture collapse is introduced by M9-A1 compression. Meanwhile **both full Original SwiftVR and M11-B+Original show a two-frame luminance oscillation**. These are separate observations: the fact that both exhibit flicker does NOT prove the Decoder alone caused it; encoder/latent, temporal upsampling, streaming state and interaction may matter.

### Strict baseline labels

| Label | Transformer | Decoder | Meaning |
|---|---|---|---|
| Full Original SwiftVR | original pretrained | original ReAE from full Original model | quality / existing flicker reference |
| M11-B30k + Original | frozen M11-B30k | original ReAE from `checkpoints_prompt_free_no_time` | best quality seen under Transformer compression; **not** final compute target |
| M11-B30k + E99 | frozen M11-B30k | M9-A1 E99 | proven lower-compute decoder path with severe four-phase texture loss |
| M11-B30k + Slim100 | frozen M11-B30k | existing Slim100 teacher-distilled ReAE-family decoder | **active ~572G candidate: user visually reports slight texture loss, visible luma/detail temporal jitter; not yet stable or integrated-profile certified** |
| TinyCNN | external AISR | external | external quality/compute reference, not an A1 variant |

## 2. Budget opportunity overlooked in earlier discussions

The historical repository already contains the **structurally slimmed ReAE decoder** (`swiftvr/models/reae_slim_decoder.py`) and a `Slim100` checkpoint with four stage widths `[256,128,64,64]`. This retains ReAE-family temporal TGrow and causal layout while reducing stage channels, unlike the more aggressive M9-A1 topology. Source: `docs/shared_video_autoencoder.md`, `tools/train_reae_slim_teacher_distill_ddp.py`.

Same-profile component costs at 1920×1088 (GMAC/output frame):

| Component | GMAC/frame | Evidence |
|---|---:|---|
| M11-B30k Transformer (activated) | 145.5955968 | M11 runtime MAC gate |
| Frozen ReAE Encoder | ~42.278 | fixed previous encoder profile |
| Original ReAE Decoder | ~343.108 | historical decoder MAC audit |
| M9-A1 compact Decoder | ~75.9504 | previous M11 full-system budget |
| **Slim100 Decoder** | **98.2228992** | `profile_reae_slim_macs.py` analytical |
| Aggressive Slim Decoder | 86.79211008 | historical analytical option |

Derived end-to-end **MAC sum** (not new integrated profiler proof):
- M11-B + Encoder + **Original Decoder** ≈ **1061.96 GFLOPs/frame**.
- M11-B + Encoder + **A1 Decoder** ≈ **527.65 GFLOPs/frame**, but unacceptable texture loss.
- M11-B + Encoder + **Slim100 Decoder** ≈ **572.19 GFLOPs/frame**, **already INSIDE target budget by estimate**.
- M11-B + Encoder + **Aggressive Slim Decoder** ≈ **549.33 GFLOPs/frame**, an alternative only if needed after Slim100.

For the target 500–600G, a decoder ideally fits **~62.13–112.13 GMAC/frame** (given the fixed M11-B and Encoder costs). Slim100 fits without further Transformer changes.

**Do not infer Slim100 quality from historical validation alone**: `docs/shared_video_autoencoder.md` reports Slim100 ~36.65 dB / .9715 SSIM *versus Original ReAE teacher* on historical restoration latents. User now reports the fixed full-origin M11-B30k+Slim100 visual comparison: texture is slightly softer than M11-B+Original, and both brightness and fine-detail jitter remain visible. These are qualitative findings, **not a passing flicker benchmark nor measured GT PSNR/SSIM on M11-B clips**.

**Immediate low-cost decisive experiment:** evaluate pre-existing Slim100 as a drop-in M11-B decoder, not another training sweep:

```bash
cd /data1/a/SwiftVR
conda activate swiftvr
export CUDA_VISIBLE_DEVICES=6

BASE=checkpoints_prompt_free_no_time
M11B=outputs/b2b/m11b_stagea_ultra_bootstrap30k_v1/checkpoints/step_00030000
SLIM100=outputs/stage_b1_reae_slim100_teacher_distill/checkpoints/epoch_050_step_00012400/tiny_decoder
test -f "$SLIM100/model.safetensors"
test -f "$SLIM100/config.json"

# One scene first; full original clip from frame zero, matched C24 parameters.
python scripts/inference_custom_components.py \
  --input ../data/yuv_S_10bit/720p_mp4/S5_Wetland1_1920x1080p444_10bit.mp4 \
  --output outputs/m11b/M12_Slim100_C24/Wetland1/png \
  --base-checkpoint "$BASE" \
  --transformer-checkpoint "$M11B" --transformer-type moe \
  --decoder-type slim --decoder-checkpoint "$SLIM100" \
  --upscale 3 --clip-len 24 --dit-overlap 0 \
  --dtype bfloat16 --attention-backend sdpa --png
```

Only if the first scene looks competitive, run **Mangrove1** using the same command with scene/input/output substitutions. Preserve all existing reference PNGs. The existing Original SwiftVR / M11-B+Original / TinyCNN bank can be reused for native ROI comparisons; focus Wetland1 245/P0 and 248/P3 plus playback for luminance jumps. Do not resize ROI for detail judgment. Use `tools/profile_reae_slim_macs.py` and the existing full-system profiling conventions to **validate the combined compute estimate before claiming the budget milestone**.

### M12-1 executable four-way observation gate (2026-11-03)

Implementation is now staged in PR #5, with **no training/model modifications**:
- `tools/m12_slim100_fourway.py`: render Slim100 with **frozen M11-B** only if it is not already available under the exact recorded provenance; verify the reused M11-B+Original full-origin PNG reference; create a lossless native-ROI 2×2 FFV1 four-way video and selected native PNGs.
- `tests/test_m12_slim100_fourway.py`: CPU functional check for 4-way order, real crop geometry, AISR ordinal frame map, and no overwrite.

**Specified four-way order (do not substitute the old LQ four-way):**

```text
┌────────────────────────────────┬────────────────────────────────┐
│ Original SwiftVR               │ AISR baseline / TinyCNN         │
├────────────────────────────────┼────────────────────────────────┤
│ M11-B 30k + Original ReAE      │ M11-B 30k + Slim100             │
└────────────────────────────────┴────────────────────────────────┘
```

This uses the **existing reference bank** and the **already generated M11-B + Original Decoder outputs** from the prior three-way test. The previously provided output layout was `outputs/m11b/M11B30K_OriginalDecoder_Comparison/<scene>/M11B_Original/png`; if the actual saved reference is elsewhere, point `--m11b-original-root` to a directory with that same scene layout. Never regenerate Original SwiftVR or TinyCNN. If the bank or M11-B+Original reference is missing, the program fails before GPU inference rather than guessing.

```bash
cd /data1/a/SwiftVR
conda activate swiftvr
export CUDA_VISIBLE_DEVICES=6

python -m py_compile tools/m12_slim100_fourway.py
python -m unittest discover -s tests -p 'test_m12_slim100_fourway.py' -v

# First observe Wetland1 alone; if good, remove "--scenes Wetland1"
# (or repeat with both scenes: completed Wetland1 will be reused).
python tools/m12_slim100_fourway.py \
  --reference-bank outputs/custom/p1_reference_v1 \
  --m11b-original-root outputs/m11b/M11B30K_OriginalDecoder_Comparison \
  --slim-root outputs/m11b/M12_Slim100_C24 \
  --output-root outputs/m11b/M12_Slim100_C24/fourway \
  --slim-checkpoint outputs/stage_b1_reae_slim100_teacher_distill/checkpoints/epoch_050_step_00012400/tiny_decoder \
  --transformer-checkpoint outputs/b2b/m11b_stagea_ultra_bootstrap30k_v1/checkpoints/step_00030000 \
  --base-checkpoint checkpoints_prompt_free_no_time \
  --scenes Wetland1
```

Then run the same full command WITHOUT `--scenes Wetland1` to include Mangrove1. Outputs, never replacing prior E99/T1/T1B:

```text
outputs/m11b/M12_Slim100_C24/
  Wetland1/{png/,inference.log,provenance.json}
  Mangrove1/{png/,inference.log,provenance.json}
  fourway/
    Wetland1/{native_fourway.mkv,native_fourway_000245_p0.png,
              native_fourway_000248_p3.png,summary.json,ffmpeg.log}
    Mangrove1/{native_fourway.mkv,native_fourway_*.png,summary.json,ffmpeg.log}
```

Four videos share a **common full input origin**, 249 output frames, exact native output geometry and fixed `(960,540,960,960)` ROI from the bank. The ROI panels are never spatially resized; FFV1 is lossless for the cropped **8-bit RGB** displays. AISR/TinyCNN uses the established length-normalized decoded ordinal index (not independently verified timestamps). This is a **visual/temporal quality check, not GT PSNR/SSIM**. Judge P0/245 texture vs P3/248, the spatial detail of Slim100 relative to M11-B+Original, and the inherited two-frame brightness oscillation at natural FPS. Original SwiftVR is the complete original model; M11-B+Original is a different Transformer with original-family ReAE.

**M12-1 user report:** Slim100 shows **subtle but visible texture loss** versus M11-B+Original; the combination with M11-B's own small detail gap is noticeable. More important are persistent **brightness oscillation and mild detail flicker**. Unlike severe A1 four-phase collapse, this is close enough to pursue Slim100-specific stabilization and detail retention. **M12-1 is visually inspected but NOT qualified as temporally stable**, and 572G remains a component-summed compute estimate until integrated profiling. Continue to M12-2; do not widen the decoder architecture search yet.

---

## 2A. M12-2: temporal stability FIRST, slight detail recovery in parallel (2026-11-03 decision)

### Observations and hypothesis boundaries

User's direct four-way review: **M11-B 30k + Slim100** is modestly softer than **M11-B 30k + Original ReAE**, while the combined M11-B and Slim100 small texture differences are perceptible; **two-frame brightness fluctuation and mild temporal detail jitter** are the dominant current defects. The earlier Original SwiftVR and M11-B+Original comparisons already showed brightness fluctuations. Distinguish:
1. **Inherited brightness behavior** present in Original ReAE paths, irrespective of the newer decoder compression.
2. **New Slim100-specific component** (brightness/texture deviations with the SAME M11-B latent), if present.
3. **Moving detail instability**, potentially involving occlusion, geometric alignment and temporal decoder behavior. It is not the same as the earlier A1 P0/P3 four-phase texture loss.

### Confirmed Wetland1 high-motion failure regime (user visual clarification, 2026-11-03)

The 249-frame Wetland1 source has a **real shot cut near output frame 163**: a relatively steady, distant aerial view changes to a close, fast fly-over. The brightness drop around 162→163 is attributable at least in substantial part to the changed *scene contents*, so **exclude that cross-shot transition** from flicker loss, derivative statistics, and claims of decoder malfunction. Do not interpret a scene cut as a brightness-fluctuation failure by itself.

In the subsequent **frame173–248 high-motion shot**, the already examined per-frame `frames.csv` shows persistent phase-correlated ROI luma variations in **both** M11-B+Original and M11-B+Slim100, while Slim100 has extra phase-dependent detail loss. The two are not identical errors; dynamic texture/ROI movement can alias into uncompensated frame luminance and HF statistics. Label this a **high-motion/scene-content-conditioned failure case**, not proof of decoder-only root cause or of a universal 2-frame oscillation.

**Actionable M12-2 acceptance gate:** use stable pre-cut frames as a non-regression reference and the sustained post-cut fly-over as the *primary* hard case. Ignore the shot boundary itself when comparing consecutive frames or computing GT-relative temporal loss; ensure training pairs spanning independently detected cuts are masked, not smoothed. Motion-aware detail consistency (warped GT / occlusion masking) is only a follow-on if the first Slim100 adaptation preserves static texture but moving features still shimmer. Keep the initial implementation small: one focused short-section evaluation and one bounded training pilot, not a cut-detection research subproject.

The phrase "two-frame brightness flicker" is an *observation*, not a proved **mathematical period-2** signal. Determine whether it is 2-period alternating frames or a 4-period two-bright/two-dark pattern, and inspect C24 chunk-boundary effects. Neither a phase spectrum nor a local TGrow weight statistic alone establishes causality.

### First gate: reuse the existing paired-output diagnostic, zero GPU inference

No new model or cache is needed. `tools/m11b_s1_d0.py frames` operates on **any** two identical full-origin PNG streams despite the legacy CLI label `--a1`. Compare **Slim100 vs Original Decoder on the SAME M11-B 30k latent source**, preserving frame0 and native ROI. Do not compare against complete Original SwiftVR here because its Transformer differs.

```bash
cd /data1/a/SwiftVR
conda activate swiftvr
SLIM=outputs/m11b/M12_Slim100_C24
REF=outputs/m11b/M11B30K_OriginalDecoder_Comparison
OUT=outputs/m11b/M12_Slim100_Temporal_Audit

for SCENE in Wetland1 Mangrove1; do
  python tools/m11b_s1_d0.py frames \
    --a1 "$SLIM/$SCENE/png" \
    --original "$REF/$SCENE/M11B_Original/png" \
    --roi 960,540,960,960 --start 0 --end 248 \
    --phase-offset 3 --snapshots 245,248 \
    --output-dir "$OUT/$SCENE"
done
```

Outputs: `frames.csv`, `summary.json`, native snapshots. Here `a1_minus_original_luma` and `lf_delta_mae` quantify **additional Slim100-versus-Original difference**, not the inherited common flicker. `hf_delta_mae` tracks high-frequency disagreement but not objective detail correctness. The CSV also contains `luma_mean_a1` and `luma_mean_original` for plotting the two independent brightness traces. Interpret any 2- or 4-phase signal with continuous video: motion and actual scene lighting can alias into a phase average. Do not claim to have motion-compensated temporal accuracy without optical flow / GT.

**One high-value control if inherited flicker remains ambiguous:** reconstruct a short, paired clean GT clip with the original ReAE **Encoder→Original Decoder**, at the same frame-origin/streaming convention. If the clean round-trip already flickers, the ReAE/streaming temporal path is implicated; if not, pay more attention to restoration latent distribution / Transformer–Decoder interaction. This is not a request to instrument every layer or repeat the whole model on many datasets.

### Main M12-2 experiment: one compact-decoder adaptation, not an A1 loss sweep

**Freeze M11-B30k and the ReAE Encoder; initialize Decoder with the existing Slim100 checkpoint**. Reuse `M11B-S1-D1` **15,896 Legacy + 20,088 UltraVideo M11-B `z_SR` training views** and their paired GT, validation13, plus the Original ReAE Decoder output on exactly the same latent. Validate the actual cache/model identity from `plan.json`/`metadata.json` before training. The D1 data were built for M11-B + A1 but the latent format is decoder-independent.

**Existing mature baseline, important caveat:** `tools/train_reae_slim_teacher_distill_ddp.py` already trains Slim100 using **teacher RGB L2 + teacher LPIPS + teacher temporal difference matching**. When Original ReAE itself has brightness oscillation, the last term can *copy* the defect into Slim100. Blindly increasing teacher temporal loss or repeating T1B-B's highpass-teacher/lowpass-GT objective is **not** our primary solution: T1B-B already reduced E99 high-frequency response and failed the actual visual goal.

Use a **small, conceptually simpler adaptation objective**:
- **Detail retention (primary anchor):** full-frame teacher RGB / robust reconstruction and **perceptual/feature supervision to the same-latent Original Decoder output**, plus paired GT image-quality observations. Avoid GT-only pixel reconstruction dominating and blurring small textures.
- **Brightness stability (separate anchor):** a *small*, **GT-relative low-frequency luminance residual temporal loss**, e.g. `mean(abs(diff_time(LP(Y(Slim - GT))))))`. This removes genuine GT scene motion/illumination before evaluating flicker, unlike penalizing `|Slim_t-Slim_{t-1}|`.
- **No unnecessary temporal-detail smoothing:** if mild detail flicker remains after this first run, examine **occlusion-aware motion-compensated feature/high-frequency residual consistency** using GT-aligned pairs rather than adding naive raw-frame temporal TV. Treat that as a subsequent targeted extension, not a required second loss cocktail on day one.

Possible schematic (not asserted implementation or fixed hyperparameters):

```text
z = frozen M11B(LR)       # D1 cache, no Transformer backward
T = frozen OriginalReAE(z)
S = trainable Slim100(z)
G = paired clean GT

L = lambda_rgb * robust(S - T)
  + lambda_perc * perceptual(S, T)
  + lambda_temp * |delta_t LP(Y(S - G))|
```

**One smoke + one bounded pilot** on real D1 batches, fresh training output initialized from Slim100; preserve original Adam/RNG fingerprints and never pretend that it is an exact resume from historical Stage-B Slim training. Validate numerical finiteness and trainable Slim100 gradients, then evaluate on the same untouched full-origin two-video gate and existing quality references. Modest learning-rate/step budget informed by the old Slim trainer, not 500k updates. **Do not optimize an ungrounded 2-frame brightness correction by assuming a known fixed phase or forcing adjacent frames equal.**

### M12-2 runnable implementation (new, not yet GPU-validated)

Three new files in fork PR #5:
- `swiftvr/training/m12_slim_objective.py`: teacher RGB Charbonnier + high-frequency teacher L1 + optional **local** AlexNet LPIPS + paired-GT **low-frequency luminance residual temporal L1**. The temporal term compares **prediction minus GT**, not prediction to the teacher's brightness oscillation. Adjacent temporal pairs are selectively masked when a conservative GT color-histogram cut gate detects a discontinuity. It is NOT a proven cut detector; check masked-pair fraction and genuine high-motion samples.
- `tools/train_m12_slim_temporal.py`: **single CUDA GPU only**, immutable D1 paired-cache loader, balanced Legacy/Ultra sampling, frozen Original ReAE teacher, FP32 trainable Slim100, BF16 inference/autocast, exact source/data/optimizer/RNG checkpoint resume, 13-frame paired val13 and teacher/GT metrics.
- `tests/test_m12_slim_temporal.py`: CPU checks for natural motion not being penalized by the GT-relative temporal term, teacher-only gradient direction, cut masking, historical data contract, and source safety.

**No new Transformer inference/cache is required.** The first run is an A/B *before/after* quality gate, not a search over losses/weights. Run tests and a 2-update smoke, using separate never-reused output roots:

```bash
cd /data1/a/SwiftVR
conda activate swiftvr
python -m py_compile \
  swiftvr/training/m12_slim_objective.py \
  tools/train_m12_slim_temporal.py \
  tests/test_m12_slim_temporal.py
python -m unittest discover -s tests -p 'test_m12_slim_temporal.py' -v

export CUDA_VISIBLE_DEVICES=6
M11B=outputs/b2b/m11b_stagea_ultra_bootstrap30k_v1/checkpoints/step_00030000
SLIM=outputs/stage_b1_reae_slim100_teacher_distill/checkpoints/epoch_050_step_00012400/tiny_decoder
D1=outputs/m11b/M11B-S1-D1
BASE=checkpoints_prompt_free_no_time
LPIPS=outputs/local_weights/lpips_alex_complete.pt

COMMON=(
  --m11b-checkpoint "$M11B" --slim-init "$SLIM"
  --base-checkpoint "$BASE"
  --legacy-train "$D1/legacy_train_full"
  --ultra-train "$D1/ultra_train_full"
  --legacy-val "$D1/legacy_val13"
  --lpips-weights "$LPIPS" --path-root .
  --dtype bfloat16 --batch-size 1 --gradient-accumulation-steps 8
  --num-workers 2 --learning-rate 1e-5 --warmup-steps 50
  --schedule-steps 2000 --save-every 125 --validate-every 125
)

python tools/train_m12_slim_temporal.py "${COMMON[@]}" \
  --output-dir outputs/m11b/M12-S2-SlimTemporal/smoke2_v1 \
  --smoke-steps 2
# ONLY after the smoke's summary.json reports status=smoke_pass:
python tools/train_m12_slim_temporal.py "${COMMON[@]}" \
  --output-dir outputs/m11b/M12-S2-SlimTemporal/pilot500_v1 \
  --max-steps 500
```

For an interrupted pilot, resume **only** from its own step checkpoint with **identical original arguments** plus `--resume`. Changing learning rate schedule, loss weights or source files invalidates resume; `max_steps`, `resume`, and `smoke_steps` are the only intentionally excluded lineage fields. Never force-bypass provenance.

GPU acceptance: smoke must yield finite/nonzero Slim100 gradients and `m11b_updated=false`. Inspect `validation/step_00000000.json`, `step_00000125.json`, `step_00000250.json`, `step_00000375.json`, `step_00000500.json`. Val13 does NOT authorize a quality pass. The new checkpoint at `pilot500_v1/checkpoints/step_00000500/tiny_decoder` follows standard `SlimReAEDecoder.save_pretrained` config + weights and can be passed to `scripts/inference_custom_components.py --decoder-type slim` without modifying inference code.

Then use **existing M12-1 four-way utility**, pointing at a fresh result directory and the new trained decoder checkpoint:

```bash
RUN=outputs/m11b/M12-S2-SlimTemporal/pilot500_v1
OUT=outputs/m11b/M12_Slim100_Temporal500_C24
python tools/m12_slim100_fourway.py \
  --reference-bank outputs/custom/p1_reference_v1 \
  --m11b-original-root outputs/m11b/M11B30K_OriginalDecoder_Comparison \
  --slim-root "$OUT" --output-root "$OUT/fourway" \
  --slim-checkpoint "$RUN/checkpoints/step_00000500/tiny_decoder" \
  --transformer-checkpoint "$M11B" --base-checkpoint "$BASE"
```

The fourth panel represents **fine-tuned Slim100**, not the original Slim100 initial checkpoint. Compare its `native_fourway.mkv` with previously rendered pretraining M12-1 video, **especially Wetland1 frame173–248 high motion**, stable pre-cut and Mangrove1. Do NOT measure a flicker penalty across the real Wetland1 frame162→163 scene cut. If no tangible visual improvement, stop this loss rather than automatically extending to 500k.

### Stop criteria and downstream decisions

- **Accept the M12-2 checkpoint** only if it decreases visible brightness jumping and detail jitter in actual-speed Wetland1/Mangrove1 playback **without perceptible new texture flattening**, with paired-val GT quality not collapsing, and the underlying decoder stays ~98.223 GMAC/frame (architecture unchanged).
- If the slim teacher error falls but inherited brightness flicker remains unchanged, do **not** lengthen the same teacher imitation indefinitely. Focus one targeted low-frequency ReAE/streaming correction or latent-distribution-aware joint adaptation. The *teacher* may contain the flicker.
- If GT-relative temporal residual improves numerically but motion detail becomes smeared, reject the checkpoint; change the temporal reference/occlusion treatment rather than weight sweeping.
- After a stable ~572G candidate and integrated compute report, perform M12-3 Transformer endpoint/feature quality restoration or minimal Transformer–Decoder co-adaptation only if the residual texture gap justifies it.

**Engineering constraints:** an independent minimal M12-2 trainer is now implemented in PR #5 because the legacy teacher-only Slim trainer cannot consume D1 paired M11-B latents + GT-relative temporal targets as-is. No edits were made to historical `tools/train_reae_slim_teacher_distill_ddp.py`, T1/T1B trainers, or to active source-hash resume code. PR #5 remains the M12 work branch, fork only.

---

## 3. Brightness flicker: early diagnosis, not a detached late-stage project

**Observed:** Original SwiftVR and M11-B+Original each show two-frame luma fluctuations; the A1 four-phase texture collapse is another failure.

Make one targeted, non-exhaustive attribution pass:
- Use the SAME full-origin input, phase/time-index convention and native ROI. Measure low-frequency luminance magnitude and two-frame periodicity, particularly across C24 boundaries. Do not call raw consecutive-frame difference a flicker score when real illumination/motion changes.
- Where an available clean paired video exists, compare **GT-relative low-frequency luma residual**. A small static or smoothly evolving ROI can complement this; do not infer causality from raw high-frequency energy.
- If feasible, decode the SAME clean-video ReAE encoder latents with Original Decoder once. Flicker in this path would implicate the shared ReAE encoding/decoding temporal path; absent flicker shifts attention toward restored-latent distribution / Transformer–Decoder interaction. It does not individually prove that a single TGrow layer is guilty.
- Inspect temporal growth, causal warm-up, state reset and phase alignment **only to the extent required by the observation**. We already have TGrow/phase diagnostic code; reuse it, do not invent dozens of new tests.

Decision: if flicker can be removed without reducing texture, prefer a narrowly targeted **low-frequency/color gain correction** or a **small decoder temporal fine-tune**. For learned correction, use GT-relative temporal luma residual where paired GT exists and strong texture-preserving anchor; do not directly smooth RGB across moving frames or impose equal output brightness on naturally changing content. If brightness fix requires disrupting high-frequency quality, postpone deployment of the fix and continue decoder fidelity work; do not block the entire system budget milestone with a long root-cause detour.

## 4. Decoder compression: next model decisions

**Priority 1: reuse Slim100** (Section 2). It is a mature, already-trained structure with original-family temporal path. Perform one full-origin matched visual run. If good enough, report the **M11-B30k + Slim100 candidate** and move to targeted flicker repair / validation. Do not replace it with an A1 phase-specific loss cocktail merely because A1 was the latest branch.

**Priority 2: adapt Slim100 to M11-B's actual latent distribution** if it is close but slightly softer. Prior Slim100 teacher distillation was on historical Stage-A `z_SR`, not proven on M11-B `z_SR`. Reuse cached M11-B train/val pairs and teacher Original Decoder outputs if their provenance is verified. Fine-tune Slim100 on M11-B restoration latents, preserving original-family temporal structure and high-frequency response. Use **one established teacher-distillation objective** (e.g. RGB/feature/perceptual teacher anchor with GT-relative low-frequency temporal correction), no arbitrary energy maximization. Include any flicker correction only after measured temporal behavior. A meaningful bounded pilot, one native Wetland/Mangrove gate.

**Priority 3: only if Slim100 fails materially**, consider the earlier 86.79-GMAC aggressive Slim variant or controlled structured reduction/fine-tuning of expensive original ReAE stages. Prefer architecture-preserving structured slimming + distillation / recovery over a new temporal decoder topology. One new decoder candidate, not a grid.

**Stop condition for budget stage**: a 500–600G candidate with acceptable native fine detail and manageable temporal artifacts on Wetland1/Mangrove1, and a profiler report with explicit model identities and 1080p-class geometry. If neither existing Slim nor one adapted variant can preserve quality, record a clear capacity/decoder-training limitation before revisiting architecture. Do not require beating Original SwiftVR in both quality and FPS before recording a valid compute milestone.

## 5. Final quality optimization after the whole-system budget gate

With **compact Decoder and compute budget locked**, assess remaining Transformer detail gap (compared fairly with Original Decoder and the compressed decoder variants). Then explore the proven standard recovery tools:
- **Output/endpoint or feature distillation** in addition to existing TA/Stage-A velocity supervision; first verify teacher/student latent-coordinate and conditioning alignment; otherwise distill in an aligned decoded RGB/feature domain. Do not blindly copy teacher flicker.
- One bounded fine-tune of M11-B from 30k to correct remaining detail, then selectively expand image/video data (textures, motion, realistic degradations) **only if** actual native-video failure motivates it.
- **Selective joint Transformer + compressed decoder adaptation** if fixing one block in isolation cannot recover details; maintain same inference architecture and 500–600G profile.
- Only then consider budget reallocation or quantization as an alternative; quantization usually reduces precision/storage/bandwidth, **not MAC-count FLOPs under this profiling convention**. It may aid deployment speed later, but does not count as a 500–600G MAC budget reduction by itself.

This retains the valuable mature endpoint/feature KD ideas from the previous M12 draft but **corrects their order**. A long Transformer-only quality sweep is no longer the immediate next experiment.

## 6. Compact experiment sequence and gates

| Gate | Action | Success / next decision |
|---|---|---|
| M12-0: flicker & compute | One focused two-frame luma/source check with Original and M11-B+Original; verify 1080p-class MAC component budget | Record likely artifact locus and safe loss target, without claiming a unique root cause |
| M12-1: Slim100 drop-in | Frozen M11-B30k + existing Slim100 checkpoint; full Wetland1 then Mangrove1; matched bank comparisons | If fidelity acceptable, candidate ~572G → confirm integrated compute |
| M12-2: preserve detail & correct flicker | If necessary, short Slim100 fine-tune on exact M11-B restored latents, small temporal/luma correction or one architecture-preserving slimmer decoder | Stable 500–600G candidate; no A1 four-phase texture collapse; record residual two-frame flicker |
| M12-3: final detail recovery | Once system compute gate is reached, output-aware M11-B recovery + optional selective joint decoder adaptation / data expansion | Move toward Original SwiftVR detail without leaving budget |

Do not turn the gate table into mandatory weeks of experiments: if pre-existing Slim100 already meets quality and cost, record it and skip optional decoder sweeps. Visual failures drive next work, not fear of hypothetical reviewer objections.

## 7. Implementation, server and new-chat handoff constraints

1. Read `docs/M11B_TRANSFORMER_V1_MILESTONE.md`, `docs/m11_transformer_budget_gate.md`, `docs/shared_video_autoencoder.md`, `swiftvr/models/reae_slim_decoder.py`, then this document. Keep checkpoint and baseline names exact. Check actual trained checkpoint presence; docs only give a historical path.
2. Preserve `M11-B 30k` and **Original ReAE** reference; create fresh outputs only. All decoder training is from a fresh run if optimizer, source hashes or objectives change; no spoofing checkpoint resume lineages.
3. If running on the shared school GPU server, DO NOT replace files participating in active source-hash checkpoint validation or change git branches during live training. Remote GitHub doc edits do not update the offline server.
4. Reuse `scripts/inference_custom_components.py --decoder-type slim`, existing ReAE Slim profiler, source bank, prior A1 / original references. Do not create duplicate new scripts for one-off comparisons.
5. Keep the distinction between M11-B+Original vs complete Original SwiftVR, especially when assigning a Teacher.
6. Evaluate actual native frames and temporal playback; where GT is available report GT-aligned PSNR/SSIM/LPIPS. Native clips without GT are **not** quantitative image-quality proof. The external TinyCNN uses historical length-normalized ordinal alignment, not verified PTS alignment.
7. In M12, the **target is total ~500–600 GFLOPs/frame** at the canonical 1080p-class output. End-to-end latency/throughput measured alongside but **secondary** to matching the compute band and usable quality.
8. Keep PR #5 document edits and future commits **only on LIUTIGHE/SwiftVR**; no actions against `H-oliday/SwiftVR`, no upstream PR. Avoid deleting legacy trials or force-pushing.

## 8. What remains unproven

- Where the two-frame flicker originates (Decoder temporal topology, encoder / streaming state, latent distribution or interaction). The current controlled comparison does not identify a unique cause.
- Whether the historical Slim100 checkpoint exists in the active runtime and generalizes to M11-B30k full-origin clips.
- Whether a good 500–600G system can also remove the inherited two-frame brightness oscillation without harming detail; preserving/fixing it is a measurable quality goal, not an assumed property.
- Whether output-aware Transformer fine-tuning / joint adaptation will yield visible quality improvement once the compact decoder is in use.
- The final end-to-end latency and full-system profiler result for M11-B+Slim100, though component-level arithmetic places it within budget.
