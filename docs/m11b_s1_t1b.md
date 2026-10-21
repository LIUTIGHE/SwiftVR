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

## Stage 3 — independent C24 Wetland1/Mangrove1 gate

Do not promote a variant based on val13 alone. Use the prior unchanged `scripts/inference_custom_components.py`, M11-B bootstrap30k, full-origin source videos, upscale3, C24 overlap0 BF16 SDPA, and A/B final decoder paths. Compare to exact E99 with full PNG sequences and native 960×960 ROI at Wetland1 frame245/P0 and frame248/P3, plus Mangrove1 249 frames. Preserve all existing E99/T1 and A/B outputs. A/B comparison/render commands will be added after their pilot validation gate, to avoid re-rendering until the source of learning is demonstrated.

**Reject** a variant if it reduces per-phase deviation only by suppressing good-phase texture. Original ReAE's own temporal artifacts are NOT GT and must not be blindly copied. Evaluate subjective temporal brightness flicker, especially P0/P3 and chunk boundaries. No model, quality or runtime pass should be assumed before these full-origin tests.
