# M11-B Transformer Compression V1 — Research Milestone & Handoff

Status: **completed research milestone on 2026-11-02**, not a production-quality or comprehensive-CI certification.
Ownership: **LIUTIGHE/SwiftVR fork only**; upstream **H-oliday/SwiftVR is read-only and must never be modified by this project**.

## Git lineage and immutability

- Completed historical research PR: https://github.com/LIUTIGHE/SwiftVR/pull/4
- Merge commit into fork main: `4a38a50e71fa0fff1c6a276f2cda02a4d227d15b`
- PR #4 historical head snapshot: `milestone/m11b30k-transformer-v1` at `8c13f1466918aee1a24d84a6f3bc8548f03f1df8`
- Forward branch for the next work: `research/m12-transformer-quality-recovery`
- PR #4 includes 614 commits / 323 files of iterative research; tests on school GPU are not exhaustively represented by GitHub CI. Preserve its history and experimental scripts; do not rewrite as a production-tested release.
- Do not delete or overwrite any M11-B, E99, T1/T1B or custom reference checkpoint/cache. GitHub merge does **not** update the offline server; do not casually checkout or pull a new branch during running training.

## What V1 has actually established

**Main observation (reported by the user after a matched, full-origin two-scene visual comparison):**
- M11-B bootstrap30k **+ Original ReAE Decoder** gives *slightly lower visual quality than full Original SwiftVR* but **does not show the severe four-phase texture degradation** seen with A1.
- This M11-B + Original system instead shows the **same two-frame luminance fluctuation** already visible in full Original SwiftVR.
- M11-B + compressed M9-A1 Decoder **E99** loses much more detail in a phase-dependent manner (Wetland1 global245/P0 weak, global248/P3 relatively strong).
- Therefore, the **new severe phase-dependent quality loss is primarily attributable to replacing Original ReAE Decoder with compressed A1**, rather than being an intrinsic failure of the M11-B compressed Transformer. This is strong controlled comparative evidence, not an exact mathematical decomposition of every error.

**Three separable observed effects (not proven independent causal mechanisms):**
1. Transformer compression quality gap: Original full SwiftVR vs M11-B + Original decoder.
2. Decoder replacement quality gap: M11-B + Original decoder vs SAME M11-B + E99/A1 decoder.
3. Shared temporal **system** symptom: two-frame luminance fluctuation present in both Original SwiftVR and M11-B + Original decoder. Its exact source remains undetermined; the Encoder, Decoder, latent distribution, and streaming protocol may all be involved. Do not equate this with A1 four-phase detail collapse.

V1 **does not** prove that further training will close all transformer gaps, eliminate ReAE flicker, or deliver original throughput. Any parameter/MAC ratio is **not** an end-to-end speedup.

## Reference models: keep identities distinct

| Label | Transformer | Decoder | Use |
|---|---|---|---|
| Original SwiftVR | Original pretrained (checkpoint root `checkpoints`) | Original ReAE | visual/quality ceiling reference |
| M11-B + Original | `outputs/b2b/m11b_stagea_ultra_bootstrap30k_v1/checkpoints/step_00030000` | original `checkpoints_prompt_free_no_time/reae.safetensors` | **V1 Transformer-only compression baseline** |
| M11-B + E99 | same M11-B 30k | `outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder` | Decoder compression diagnosis |
| M11-B + T1B-A/B | same M11-B 30k | pilot250_a_v1 / pilot250_b_v1 | negative/limited progress on Decoder |
| TinyCNN | external AISR model | external | deployment/comparison reference, not A1 Decoder |

Checkpoint `checkpoints_prompt_free_no_time` is **not** the same named root as original complete `checkpoints`. Do not call an Original SwiftVR output the same-latent Original ReAE Teacher. Explicitly verify teacher/student encoder identity before any latent regression.

## Fixed evaluation contracts

- Test originals: `../data/yuv_S_10bit/720p_mp4/S5_Wetland1_1920x1080p444_10bit.mp4` and `S5_Mangrove1_1920x1080p444_10bit.mp4`.
- Use full video from frame 0, restore 249 output frames with `--upscale 3 --clip-len 24 --dit-overlap 0 --dtype bfloat16 --attention-backend sdpa --png`. Native output is ~2160p; never confuse this with the reported **1080p-class ~527G/frame MAC** budget.
- Phase index on these full-origin outputs: `phase=(global_frame+3)%4`. Wetland1 frame245=P0, frame248=P3.
- Native ROI on output: `(960,540,960,960)`. Compare unresized ROI and playback at actual FPS to assess flicker.
- Reuse fixed bank `outputs/custom/p1_reference_v1/reference.json` for Original SwiftVR and TinyCNN, and existing E99 PNG/provenance; reconstruct only a missing candidate.
- The external TinyCNN sequence has historically been aligned by **length-normalized decoded ordinal index**, not independently verified PTS-to-PTS time synchronization. State this limitation; do not use it as GT.
- Do **not** assert PSNR/SSIM/LPIPS improvements on these custom videos without true paired HR GT; high-frequency energy is merely a diagnostic.

A direct, read-only visual comparison of Original SwiftVR / M11-B+Original / TinyCNN can reuse `tools/compose_4way_video.py`, with only the M11-B+Original full-origin outputs generated once from `scripts/inference_custom_components.py --decoder-type original`.

## Quantitative/structural anchor (reported in existing M11 budget audit)

- M11-B DiT: hidden1024, depth20, 8 attention heads, shared expert width1024, 12 routed experts of width576, top2 active; 415,564,480 total parameters, 179,314,880 active parameters.
- Runtime audited **M11-B+frozen encoder+A1 decoder** total ~527.648 GFLOPs/output frame at canonical 1920x1080-class output. This number is not automatically valid for M11-B+Original Decoder (different decoder compute).
- Exact arch/report derivation: `docs/m11_transformer_budget_gate.md`. For any claims of speedup versus Original SwiftVR, obtain independent matched-resolution *end-to-end* inference runtime/memory; MAC accounting and active params alone are insufficient.

## Decoder studies: recorded negative/limited findings

- Stage M11B-S1-D0: same-M11-B Wetland1 ROI phase-dependent E99 HF energy (P0 ~.02318 vs P3 ~.03547) versus Original ReAE (~.03797 vs ~.03827); **not** a GT quality metric.
- Stage T1 500 updates: val13 GT PSNR improved 24.78→26.24 dB, GT LPIPS .3138→.2777; yet visible custom-video **oversmoothing**. Mangrove1 E99-frequency retention roughly 50% on T1. Do not infer better visual quality merely from val13 PSNR.
- Stage T1B-A/B 250 updates: all-phase Original ReAE supervision greatly reduces the prior extra over-smoothing but **does not repair P0 weakness**. A HF-energy/E99 ratio: Wetland P0/P3 .895/.938, Mangrove P0/P3 .931/.950. B: Wetland .819/.854, Mangrove .857/.879. The user reports very little visible difference from E99. Neither is accepted as a decoder fix.
- A post-250 further training attempt has been discussed; its latest endpoint/results are **not established here**. Do not invent final learning curves.

## Resulting decision

**Freeze M11-B bootstrap30k Transformer weights as the baseline V1. Decoder compression remains the IMMEDIATE active goal, not a deferred follow-up.** Our next system-level target is **500–600 GFLOPs/output frame at the canonical 1080p-class profiling geometry**, using frozen M11-B + frozen ReAE Encoder + a quality-preserving compact decoder. Throughput and latency are important secondary measurements, not the first selection criterion. Analyze the existing **two-frame luminance flicker first**, without assuming it originates entirely in the Decoder or entirely in the Transformer. Decoder compression and temporal correction may then be developed together or sequentially as the evidence supports. Once the entire system fits the compute band with acceptable visual detail/temporal quality, move to final Transformer detail recovery and, where necessary, joint Transformer–Decoder adaptation. Preserve the separate identities of the severe four-phase A1 texture loss and the pre-existing two-frame brightness oscillation.

**Previously underused candidate: Slim100.** The historical `SlimReAEDecoder` preserves the Original ReAE temporal topology and changes spatial-stage channel widths; checkpoint recorded in `docs/shared_video_autoencoder.md` as `outputs/stage_b1_reae_slim100_teacher_distill/checkpoints/epoch_050_step_00012400/tiny_decoder`. Its documented estimated decoder cost at 1920×1088 is **98.2228992 GMAC/frame**. Combining the separately reported M11-B DiT **145.5955968 GMAC/frame** and ReAE Encoder **42.278 GMAC/frame**, a provisional same-convention estimate is **572.19 GFLOPs/frame** (`1 MAC = 2 FLOPs`). This is a **candidate estimate, not a newly measured integrated-profile pass**. The previous Slim100 val13 **36.65 dB / 0.9715 SSIM** against Original ReAE was reported on historical restoration-latent data, **not proven on M11-B30k latents or full native Wetland/Mangrove streams**. Test this checkpoint directly with `scripts/inference_custom_components.py --decoder-type slim` before redesigning/long-training another decoder.

The old M11-B+E99/A1 ~527.648 GFLOPs/frame figure establishes the *cost feasibility* of aggressive decoder compression, **not acceptable decoder quality**. The heavier M11-B+Original path is the texture and temporal reference, not the expected final 500–600G deployment configuration.

### M12-1 Slim100 outcome and M12-2 priority (user-reported 2026-11-03)

Native four-way review of **M11-B30k + Slim100** against complete Original SwiftVR, external AISR/TinyCNN and **the same M11-B30k + Original ReAE**: **subjective visual-quality impressions only** are approximately Original SwiftVR 100, M11-B+Original ReAE 85, and M11-B+Slim100 80 on an informal relative scale. These are NOT PSNR, LPIPS, a formal perceptual study, or additive metric points; the small Transformer and Decoder detail losses become visible in combination. Slim100 shows only modest per-frame texture loss, but the small Transformer and Decoder detail deficits **visibly accumulate**, and the most pressing defects are **brightness oscillation and mild temporal detail jitter**. This is a visual observation, **not evidence that Slim100 has passed a temporal-stability or paired-GT quality threshold**. Preserve M11-B30k + Slim100 as the primary ~572G (component-sum estimate) system candidate; do not restart A1 or widen the decoder grid prematurely. **First diagnose inherited vs Slim-added temporal error; then run one Slim100-only, teacher-detail-anchored and GT-relative temporal-luma adaptation** using already completed D1 M11-B latent/GT caches. The historic Slim100 teacher-temporal imitation objective can copy the Original ReAE's brightness oscillation; teacher temporal matching alone is insufficient as a flicker-removal target. Keep final Transformer detail recovery after the compact system's temporal/compute gate. Details: `docs/M12_TRANSFORMER_QUALITY_ROADMAP.md`.


References inside repo: `docs/m11_transformer_budget_gate.md`, `docs/m11b_s1_d0.md`, `docs/m11b_s1_d1.md`, `docs/m11b_s1_t1.md`, `docs/m11b_s1_t1b.md`, `docs/REPOSITORY_LIFECYCLE.md`.

## Handoff to a new chat / coding agent

Read **this file first**, then `docs/M12_TRANSFORMER_QUALITY_ROADMAP.md`, then relevant implementation. Do not ask the user to re-run historical checks already answered by full-origin comparison. Preserve data/model origin and the three different quality gaps. **Next priority is to diagnose the two-frame luminance flicker and preserve fine detail while compressing the Decoder with M11-B30k frozen to reach 500–600 GFLOPs/frame.** Check the existing Slim100 checkpoint before designing a new decoder. After the system-level compute milestone, pursue Transformer detail recovery and selective joint adaptation. Do **not** lock Original Decoder as the only acceptable next-phase model, and do **not** assume the flicker is decoder-caused until a controlled diagnostic supports it. Work exclusively on `LIUTIGHE/SwiftVR` branches; never push, PR, edit or merge `H-oliday/SwiftVR`. Prefer one meaningful improvement test to many tiny defensive ablations; use precise changes and source-hash/optimizer resume contracts.
