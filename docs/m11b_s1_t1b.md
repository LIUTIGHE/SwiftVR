# M11B-S1-T1B — Original ReAE all-phase distillation (A/B)

**Status:** implementation prepared on PR #4; first CPU/GPU smoke and pilot250 **not yet executed**. This work is an ablation of the post-T1 over-smoothing failure, not a proven quality fix.

## Motivation and fixed evidence

The full T1 pipeline has completed full D1 caches: 15,896 Legacy train + 20,088 UltraVideo train + val13 (13). T1 GT reconstruction improved from 24.78 to 26.24dB on val13; however the user's full-origin Wetland1 inspection found both T1/250 and T1/500 too smooth versus E99, particularly at frame248/P3, while frame245/P0 had partially recovered at 500. T1/500 highpass energy retained ~51.5% of E99 in P0 but only ~40.9% in P3 on Wetland1, and ~50% across all Mangrove1 phases. High-frequency energy is NOT a GT quality metric. **E99 remains the deployed/visual reference.**

Rather than treating P0/P3 as requiring different teachers, the next controlled hypothesis is that **frozen Original ReAE can supply all phases**, provided flickering low-frequency luminance is corrected with GT supervision. It is not established that Original's high-frequency content is completely correct or that A1 can represent all teacher detail at current compute.

## Frozen identities

- **M11-B**: `outputs/b2b/m11b_stagea_ultra_bootstrap30k_v1/checkpoints/step_00030000`
- **A1 E99 initial**: `outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder`
- **Original ReAE**: `checkpoints_prompt_free_no_time/reae.safetensors`
- **D1 source**: `outputs/m11b/M11B-S1-D1/{legacy_train_full,ultra_train_full,legacy_val13}`
- **Trainable**: A1 decoder ONLY (FP32 master). DiT not loaded; frozen Original decoder strictly loaded then encoder removed before GPU transfer. No HR duplication, no renewed D1 cache.
- **Validation**: all 13 fixed GT paired views at steps 0,125,250; Original and student decoded from EXACT same cached `z_sr` sample.
- **Two separate output directories**: `outputs/m11b/M11B-S1-T1B/pilot250_a_v1` and `pilot250_b_v1`.

## The two mutually exclusive objective variants

Let (S=D_{A1}(z_{SR})), (T=D_{Original}(z_{SR})), and (G) the paired real HR.

The `LP` operator is a per-frame replicated-edge 7×7 mean filter, `HP(X)=X-LP(X)`. Temporal `\Delta` operates on the GT-relative BT.709 luminance **error** after low-pass filtering (not on the source motion itself).

**T1B-A (Original all-phase RGB supervision)**:

```text
L_A = 1.0 * mean_abs(S - T)
    + 0.5 * mean_abs(temporal_diff( LP(Y(S)-Y(G)) ))
```

No GT RGB/LPIPS reconstruction term. This diagnostic asks whether A1 can fit the whole teacher and whether GT-relative temporal correction can prevent inheriting its brightness flicker. Low-frequency brightness biases constant over time can remain; judge under paired GT metrics.

**T1B-B (Original all-phase HF, GT LF and temporal)**:

```text
L_B = 1.0 * mean_abs(HP(S) - HP(T))
    + 1.0 * mean_abs(LP(S) - LP(G))
    + 0.5 * mean_abs(temporal_diff( LP(Y(S)-Y(G)) ))
```

Both variants use identical Original teacher for *all phases*, without phase-dependent targets. Loss excludes GT LPIPS at default weight 0 in BOTH variants; this keeps the ablation focused on teacher/GT supervision and prevents a repeat of the GT-LPIPS-driven over-smoothing confound. For quality measurement, locally stored AlexNet LPIPS is loaded **for val13 only**, producing `a1_gt_lpips_eval` and `original_gt_lpips_eval`. It is **not** backpropagated when `--gt-lpips-weight=0`.

Neither objective maximizes raw high-frequency magnitude. With teacher and student decoded from the same frozen latent, high-pass alignment is meaningful but still cannot guarantee teacher texture accuracy.

## Files

- `swiftvr/training/m11b_t1b_objective.py` — A/B objectives and phase-specific diagnostics
- `tools/train_m11b_s1_t1b_decoder.py` — independent T1B trainer (does not alter T1)
- `tools/run_m11b_s1_t1b.sh` — fail-closed pilot/smoke launcher
- `tests/test_m11b_s1_t1b_objective.py`, `tests/test_m11b_s1_t1b_decoder.py`
- `docs/m11b_s1_t1b.md`

Existing `swiftvr/training/m11b_phase_objective.py`, `m11b_decoder_pairs.py`, E99 and all D1 caches stay unchanged. `t1b_objective` shares the spatial lowpass/phase convention with the existing T1 objective but the T1 code is not otherwise edited.

## Stage 0 — CPU syntax and contract gates

Sync those five new files plus this document, then:

```bash
cd /data1/a/SwiftVR
conda activate swiftvr

python -m py_compile \
  swiftvr/training/m11b_t1b_objective.py \
  tools/train_m11b_s1_t1b_decoder.py

bash -n tools/run_m11b_s1_t1b.sh

python -m unittest discover -s tests -p 'test_m11b_s1_t1b_objective.py' -v
python -m unittest discover -s tests -p 'test_m11b_s1_t1b_decoder.py' -v
```

## Stage 1 — independent 2-update CUDA smoke for A and B

```bash
cd /data1/a/SwiftVR
export CUDA_VISIBLE_DEVICES=7

bash tools/run_m11b_s1_t1b.sh smoke A
bash tools/run_m11b_s1_t1b.sh smoke B
```

Both must have `summary.json` with `status=smoke_pass`, `variant`, `step=2`, finite nonzero gradients, frozen Teacher and no changed M11-B weights. Both must write `checkpoints/step_00000002/tiny_decoder/model.safetensors` and `training_state.pt`. Do not overwrite older smoke directories. New runs begin at **E99**.

## Stage 2 — paired 250-update controlled experiments

**Only after BOTH smokes pass**:

```bash
bash tools/run_m11b_s1_t1b.sh pilot A
bash tools/run_m11b_s1_t1b.sh pilot B
```

The launcher checks its own variant's successful smoke before starting a new pilot. Both use one GPU, BF16 autocast with FP32 A1 weights, batch1×accum8, 250 optimizer updates, AdamW 1e-5, warmup100, schedule horizon6000, 50:50 alternating Legacy/Ultra microbatches. Neither pilot resumes the 2-step smoke or the old T1/500 checkpoint. Run under separate output folders. All source identities and trainer/objective hashes are frozen in `run_config.json`, with strict checkpoint resume within a given variant.

Each pilot writes validation at **step 0, 125, 250**:

```text
outputs/m11b/M11B-S1-T1B/
  pilot250_a_v1/
    validation/step_00000000.json
    validation/step_00000125.json
    validation/step_00000250.json
    validation_visuals/step_*/
    checkpoints/step_00000000/tiny_decoder/
    checkpoints/step_00000125/tiny_decoder/
    checkpoints/step_00000250/tiny_decoder/
    train_log.jsonl, summary.json
  pilot250_b_v1/
    (same structure)
```

Metrics: paired GT PSNR/SSIM/MAE/RMSE for BOTH A1 and Original, validation-only GT LPIPS for BOTH, Teacher RGB L1, teacher HF L1, GT LF L1, GT RGB/HF L1, GT-relative LF-temporal error, per-phase student/Original highpass energies and HF teacher/GT errors, per-phase GT MSE.

**Interpretation priority:** does A substantially lower teacher RGB/HF error? Does B preserve/recover teacher highpass energy on P0 **without flattening P3**? Are brightness and temporal errors smaller for B? Does GT PSNR/SSIM/LPIPS stay tolerable? The validation highpass is a 7×7 average-filter difference and therefore **not numerically interchangeable** with custom-video D0's Gaussian highpass (sigma1.5).


## Stage 3 — COMPLETED pilots, current full-origin C24 comparison

**User report (2026-10-22):** both 250-update pilots A and B finished. Their val13 logs and custom-video results have not yet been provided; this is **training completion only**, NOT A/B image-quality PASS.

The new evaluator uses the already-existing fixed Original SwiftVR + TinyCNN reference bank (made by `tools/visual_check_m8_task.sh prepare`) and the *already rendered* M11-B+E99 full-origin PNGs from T1. Only T1B-A and T1B-B require new M11-B inference.

**Four compared methods and exact meaning:**
- `Original SwiftVR` = complete original conditional SwiftVR from `checkpoints`, pre-rendered in `reference.json`, NOT the frozen Original ReAE decoder paired with M11-B.
- `E99` = M11-B bootstrap30k + M9-A1 E99 decoder, reference `outputs/m11b/M11B-S1-T1/custom_pilot500_v1/{Wetland1,Mangrove1}/e99/png`.
- `Current` = fixed same M11-B bootstrap30k + T1B-A/B decoder, native C24 inference.
- `TinyCNN` = external precomputed AISR baseline video `../data/yuv_S_10bit/AISR_pred/S5_{Wetland1,Mangrove1}_1920x1080p444_10bit_SR_YUV444_RC1.0.mp4`, reusing earlier bank frame-ordinal mapping; this is **not** the A1 tiny decoder.

First, sync new files `tools/m11b_custom_val.py`, `tools/compare_m11b_t1b_ab.py`, `tests/test_m11b_custom_val.py` (and `tools/train_with_custom_val.py` for future runs). Locate the bank on the offline server:

```bash
cd /data1/a/SwiftVR
conda activate swiftvr
find outputs/custom -name reference.json -type f -print
```

The default is `outputs/custom/p1_reference_v1/reference.json`, but override the root if the bank was saved under a different name. Do **not** create a fresh bank or run Original SwiftVR if a valid bank already exists. The evaluator checks original bank file sizes/mtimes, immutable E99/source movie/transformer/ReAE SHA256, matching C24 inference and exact full-frame geometry; it fails closed on misaligned assets.

```bash
python -m py_compile \
  tools/m11b_custom_val.py tools/compare_m11b_t1b_ab.py \
  tools/train_with_custom_val.py
python -m unittest discover -s tests -p 'test_m11b_custom_val.py' -v

export CUDA_VISIBLE_DEVICES=7
BANK=outputs/custom/p1_reference_v1  # replace with ACTUAL reference.json parent
ROOT=outputs/m11b/M11B-S1-T1B

python tools/m11b_custom_val.py \
  --decoder-checkpoint "$ROOT/pilot250_a_v1/checkpoints/step_00000250/tiny_decoder" \
  --reference-bank "$BANK" \
  --output-root "$ROOT/custom_pilot250/a"

python tools/m11b_custom_val.py \
  --decoder-checkpoint "$ROOT/pilot250_b_v1/checkpoints/step_00000250/tiny_decoder" \
  --reference-bank "$BANK" \
  --output-root "$ROOT/custom_pilot250/b"

python tools/compare_m11b_t1b_ab.py \
  --root-a "$ROOT/custom_pilot250/a" \
  --root-b "$ROOT/custom_pilot250/b" \
  --output "$ROOT/custom_pilot250/ab_comparison"
```

The evaluator runs each source video from **frame0** to the full 249 frames and holds `--upscale 3 --clip-len 24 --dit-overlap 0 --dtype bfloat16 --attention-backend sdpa --png` fixed. The two runs need only new T1B-A/B decode; they never invoke `scripts/inference.py` to reconstruct Original SwiftVR or regenerate E99/TinyCNN. It saves native full-size PNG sequences and four-way native 960×960 ROI panels for selected frames, plus `frames.csv` and `summary.json`. A/B aggregation produces five-column native ROI frames for Wetland1 global245/P0 and 248/P3, plus Mangrove1 global12/124/245/248, with a per-phase comparison JSON. No GT is available, so raw HF energy and HF deviation are diagnostics, **not objective image quality**.

Optional: `--video` on each `m11b_custom_val.py` command saves a full-resolution lossless FFV1 `native_fourway.mkv` per scene, greatly increasing disk usage; off by default. Full PNG output is always retained.

Review outputs:

```text
outputs/m11b/M11B-S1-T1B/custom_pilot250/
  a/Wetland1/{png/,comparison/{frames.csv,summary.json,native_fourway_000245_p0.png,...}}
  a/Mangrove1/{png/,comparison/...}
  b/Wetland1/{png/,comparison/...}
  b/Mangrove1/{png/,comparison/...}
  ab_comparison/{summary.json,Wetland1_global_000245_p0_fiveway.png,
                 Wetland1_global_000248_p3_fiveway.png,...}
```

**Decision gate**: compare texture preservation at P3/248, recovery at P0/245, and any four-phase flicker/brightness shift; inspect continuous original-speed videos where possible. Variant A may copy Original decoder luminance variation; B uses matched GT for low-frequency correction. A/B GT val13 and custom highpass energy quantify different properties; never promote from PSNR or HF energy alone.

## Stage 4 — future training: automatically run two custom clips on milestone validation

Do **not** run M11-B full-video inference while the optimizer model is still resident in GPU memory. The reusable driver `tools/train_with_custom_val.py` runs **train to milestone, clean process exit, render Current for both videos, strict exact resume**. Original SwiftVR, E99 and TinyCNN are cached and reused. The two full custom videos are a costly secondary validation; default val13 stays cheap and can run every 125 steps, while custom clips can be scheduled less frequently by choosing e.g. milestones 250,500.

The new train run must use its own fresh directory, not retrofit an already-completed pilot250 run (which has no driver plan). Example variant B training configuration, stored as `outputs/m11b/t1b_future_config_b.json` (manually create if/when starting a *future* run):

```json
{
  "variant": "B",
  "base_checkpoint": "checkpoints_prompt_free_no_time",
  "m11b_checkpoint": "outputs/b2b/m11b_stagea_ultra_bootstrap30k_v1/checkpoints/step_00030000",
  "a1_init": "outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder",
  "legacy_train": "outputs/m11b/M11B-S1-D1/legacy_train_full",
  "ultra_train": "outputs/m11b/M11B-S1-D1/ultra_train_full",
  "legacy_val": "outputs/m11b/M11B-S1-D1/legacy_val13",
  "lpips_weights": "outputs/local_weights/lpips_alex_complete.pt",
  "teacher_weight": 1.0,
  "lowfreq_gt_weight": 1.0,
  "temporal_weight": 0.5,
  "highpass_kernel": 7,
  "gt_lpips_weight": 0.0,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "num_workers": 2,
  "dtype": "bfloat16",
  "learning_rate": 1e-5,
  "schedule_steps": 6000,
  "warmup_steps": 100,
  "save_every": 125,
  "validate_every": 125,
  "log_every": 10
}
```

Launch a FRESH run (illustrative, not needed for completed pilot A/B):

```bash
python tools/train_with_custom_val.py \
  --train-script tools/train_m11b_s1_t1b_decoder.py \
  --train-config outputs/m11b/t1b_future_config_b.json \
  --train-output outputs/m11b/M11B-S1-T1B/future_b_500_auto \
  --custom-output outputs/m11b/M11B-S1-T1B/future_b_custom \
  --reference-bank "$BANK" \
  --milestones 250,500
```

Stages run at 250 then 500 and the current decoder is evaluated on both scenes after each stage. Trainer and code/config hashes, optimizer, per-rank RNG and step counters enforce exact resume within the same run; no new TinyCNN or Original SwiftVR inference is required. The driver stores its plan outside the fresh trainer output tree, rejects changing the plan mid-run, and refuses to overwrite existing custom frames.

**Limitations:** the driver orchestrates consecutive child processes in one foreground command, not background/asynchronous training. Native full video custom validation still must run current M11-B forward for each checkpoint and does not eliminate this cost. A future optimized implementation might cache M11-B **inference video latents** separately, but this would require proving C24 equivalence and temporal alignment before changing semantics.


Do not promote a variant based on val13 alone. Use the prior unchanged `scripts/inference_custom_components.py`, M11-B bootstrap30k, full-origin source videos, upscale3, C24 overlap0 BF16 SDPA, and A/B final decoder paths. Compare to exact E99 with full PNG sequences and native 960×960 ROI at Wetland1 frame245/P0 and frame248/P3, plus Mangrove1 249 frames. Preserve all existing E99/T1 and A/B outputs. A/B comparison/render commands will be added after their pilot validation gate, to avoid re-rendering until the source of learning is demonstrated.

**Reject** a variant if it reduces per-phase deviation only by suppressing good-phase texture. Original ReAE's own temporal artifacts are NOT GT and must not be blindly copied. Evaluate subjective temporal brightness flicker, especially P0/P3 and chunk boundaries. No model, quality or runtime pass should be assumed before these full-origin tests.
