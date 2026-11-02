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

**Three independent sources of quality:**
1. Transformer capacity / distillation gap: Original full SwiftVR vs M11-B + Original decoder.
2. Decoder compression gap: M11-B + Original decoder vs SAME M11-B + E99/A1 decoder.
3. Temporal decoder/system artifact: two-frame luminance variation present even with Original decoder and Original full SwiftVR. Do not equate this with four-phase detail collapse.

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

**Freeze M11-B bootstrap30k + Original ReAE as the baseline V1.** Prioritize Transformer quality recovery, not more A1 minor loss sweeps. Keep decoder compression as deferred follow-up with a fresh, explicit target budget and no false equivalence to the Original ReAE training teacher.

References inside repo: `docs/m11_transformer_budget_gate.md`, `docs/m11b_s1_d0.md`, `docs/m11b_s1_d1.md`, `docs/m11b_s1_t1.md`, `docs/m11b_s1_t1b.md`, `docs/REPOSITORY_LIFECYCLE.md`.

## Handoff to a new chat / coding agent

Read **this file first**, then `docs/M12_TRANSFORMER_QUALITY_ROADMAP.md`, then relevant implementation. Do not ask the user to re-run historical checks already answered by full-origin comparison. Preserve data/model origin and the three independent quality gaps. Evaluate progress with Original Decoder held fixed and new Transformer candidate only. Work exclusively on `LIUTIGHE/SwiftVR` branches; never push, PR, edit or merge `H-oliday/SwiftVR`. Prefer one meaningful improvement test to many tiny defensive ablations; use precise changes and source-hash/optimizer resume contracts.
