# M12 Roadmap — Transformer Quality Recovery With Original ReAE Kept Intact

**Status:** plan only (2026-11-02). No new GPU results, code changes or checkpoints are implied by this document.
**Scope:** work **only** in the `LIUTIGHE/SwiftVR` fork on `research/m12-transformer-quality-recovery`. Do not touch upstream `H-oliday/SwiftVR`. M11-B 30k stays immutable.

## 1. Research question and end condition

**Can we improve M11-B's remaining visual gap to full Original SwiftVR, at an explicit inference-compute budget, while preserving its recovered fine-detail consistency?**

User-provided strong visual result: same full-origin Wetland1/Mangrove1, **M11-B+Original ReAE** is only slightly worse than Original SwiftVR; severe four-phase texture collapse comes from subsequent A1 compression, not the M11-B transformer. Two-frame luminance flicker exists even with Original ReAE and is an **independent issue**.

**Stop condition** for this track: either (a) M11-B quality improves visibly on both fixed custom clips, including previously weak detail frames, without a major motion/temporal regression, at a declared GFLOPs and actual latency; or (b) controlled longer training shows a plateau and a single teacher-target recovery experiment fails, so deeper capacity change is warranted. Do not continually invent defensive micro-ablations.

### Fixed axes

- Reference full system: Original SwiftVR, separate original DiT checkpoint and decoder.
- V1 compressed Transformer: M11-B bootstrap30k (`outputs/b2b/m11b_stagea_ultra_bootstrap30k_v1/checkpoints/step_00030000`).
- **Decoder always Original ReAE**, specifically the one in `checkpoints_prompt_free_no_time/reae.safetensors`, for every M12 Transformer candidate and its M11-B baseline.
- The model being optimized is **Transformer only**; avoid reintroducing A1 decoder ambiguities.
- Maintain single-frame source origin and C24 zero-overlap streaming. Only trained model changes, not inference alignment/protocol or the external TinyCNN baseline.
- Record **both** all-parameters/active-parameters, theoretical MAC and *measured* CUDA end-to-end latency. Sparse MoE total width is not automatically a speed improvement due to routing/gather/scatter.

## 2. Underused mature methods: ranked, not an unbounded survey

### Priority 1 — Direct teacher endpoint / feature distillation and restoration-target fine-tuning

Existing M11-B training largely distills teacher **velocity** from Stage-A/TA caches; velocity/cosine loss is a surrogate for the final restored video. Such a surrogate can plateau while perceptual quality differs. More training of exactly the old surrogate need not close the remaining gap.

First inspect what *actual* M11-B and Original SwiftVR teacher targets are already cached. **Do not rebuild all caches before checking**. If equivalent teacher/student ReAE latent coordinates are verified, cache aligned Original SwiftVR **restored latent / x0** targets for a representative fixed training subset; train M11-B with its existing velocity target and a small endpoint latent term, with optional frozen Original ReAE-decoded feature/pixel supervision on matched video frames. If latent coordinates or conditioning differ, do not force an invalid `L1(z_student,z_teacher)`; use aligned decoded RGB/feature targets instead, and state conditioning differences.

A minimal first objective (illustrative only; choose 2-3 established terms, not a 12-loss cocktail):

```text
L_M12 = lambda_velocity * L_velocity(student, existing teacher cache)
      + lambda_endpoint * L_endpoint(z_student, z_original_teacher)
      + lambda_quality * L_quality(OriginalDecoder(z_student), reference target)
```

Use direct GT/degraded-input anchors where available to avoid copying teacher artifacts blindly. Consider standard pre-trained feature/perceptual (LPIPS) only if needed for detail; avoid pure pixel L1 dominating again as in T1. Regressing the original teacher's **decoded** temporal brightness pattern may preserve its flicker, so do not frame this as a flicker fix.

**Why this is the main next experiment:** it directly optimizes the missed *actual* target while retaining well-initialized M11-B weights, avoids architectural proliferation and reuses established data/provenance infrastructure.

### Priority 2 — Matched continuation / curriculum / data distribution

- Inspect actual M11-B 0→30k learning curve, optimizer/schedule and the *existing* output/velocity training teacher, rather than assuming 30k has converged. Compare 15k/30k if still available, then extend **one** controlled quality-recovery run for a modest next budget.
- Do not turn 30k → 500k without intermediate inspection. Maintain exact-resume hashes; if optimizer/loss/schedule changes, start a **new run initialized from 30k**, identify it as fine-tuning, and do NOT fake checkpoint exact resume.
- Current paired-data caches: Legacy 15,896 views, UltraVideo 20,088 views (35,984 total); earlier materialization/velocity caches may cover different domains, so inspect rather than assume a drop-in new subset. A 50:50 sampler mixes domains by design, not necessarily the best distribution for Wetland1/Mangrove1.
- Data expansion is a subsequent lever if a concrete failure mode is visible: add high-quality complex natural textures, camera motion, exposure changes and real-degradation pairs; assess whether synthetic-to-real mismatch, frame clip scale or color processing is driving loss. A targeted subset beats indiscriminate enlargement.
- Keep evaluation cheap: val13 and training trajectories on a modest cadence, Wetland1/Mangrove1 at meaningful checkpoints only (e.g. before/after a main 2–5k recovery experiment); reuse the Original/E99/TinyCNN fixed bank. Avoid GPU-heavy complete videos at every 125 updates.

### Priority 3 — Capacity expansion only if the training targets plateau

M11-B V1 depth20, hidden1024, 12 routed experts×576, top2 plus shared. It has a demonstrated quality gap despite ~527.65 GFLOPs/frame measured for **M11-B+encoder+A1** under canonical 1080p-class setup, not yet the exact M11-B+Original runtime. If output-based recovery fails and the loss floor suggests undercapacity:
- Consider **single controlled wider/deeper student** with transparent budget (e.g. compare fixed ≤530G vs deliberately relaxed higher-cost branch); avoid a large architecture grid.
- Wider FFN/expert or network depth changes need new initialization/transfer and are not exact optimizer resume.
- Measure actual latency, peak VRAM and effective MoE utilization; macro FLOPs alone is insufficient.
- As an orthogonal established compression path, evaluate **weight/activation quantization of a high-quality reference** before another destructive pruning/decoder rebuild. Quantization could reduce bandwidth/storage, but its throughput gains depend on available hardware kernels and are not guaranteed.

### Separate track — Two-frame luminance flicker

The unchanged Original ReAE decoder / streaming protocol also shows a two-frame luminance oscillation. Longer Transformer training may compensate indirectly, but because the artifact survives with the full Original SwiftVR Transformer, **there is no evidence Transformer capacity alone fixes it**.

Treat after M12 visual quality recovery as a **small ReAE temporal-decoder correction**:
- Compare per-frame low-frequency luminance residual relative to paired GT and causal state (not raw interframe frame differences, which punish real lighting/motion).
- Prefer a tiny phase-aware luminance gain/bias/low-frequency correction or bounded temporal decoder fine-tuning over shrinking/replacing the full decoder.
- Preserve the Original ReAE high-frequency path. Verify causal inference chunk boundaries and true motion/brightness changes.
- Do not train A1/T1B to imitate the Original ReAE's brightness oscillation and then claim that this eliminates flicker.

Literature that motivates *categories*, **not automatically a drop-in implementation**:
- DiffVSR (ICCV 2025): staged progressive learning separates degradation, content and temporal learning burden. https://openaccess.thecvf.com/content/ICCV2025/html/Li_DiffVSR_Revealing_an_Effective_Recipe_for_Taming_Robust_Video_Super-Resolution_ICCV_2025_paper.html
- Improved Video VAE (CVPR 2025): group causal convolutions address temporal compression phase imbalances and flicker in video decoders; the ReAE implementation and temporal topology must be checked first. https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Improved_Video_VAE_for_Latent_Video_Diffusion_Model_CVPR_2025_paper.pdf
- Q-VDiT (ICML 2025) and QuantSparse (ICLR 2026): established quantization / distillation alternatives for video DiT; not evidence of free speedups in SwiftVR. https://proceedings.mlr.press/v267/feng25q.html and https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html

## 3. Concrete M12 experimental sequence — four meaningful decisions

| Stage | Work | Check / decision |
|---|---|---|
| M12-0: package baseline | Inspect trained M11-B30k run config, teacher cache provenance, current inference scripts; if necessary record current matched M11-B+Original vs full Original SwiftVR native samples and actual latency | **No new architecture/training.** Lock exact baseline inputs and compute. |
| M12-1: one quality-recovery run | Fine-tune M11-B30k with output/endpoint-aligned Original teacher guidance plus conservative existing velocity objective; stage as **fresh run initialized from 30k** if any settings differ | Compare a small but meaningful budget (around 2–5k updates, not 500k); look at paired loss trend and same two custom videos. |
| M12-2: if learning continues | Extend only the best version and selectively diversify texture/motion/degradation data | Main quality gain visible on Wetland1 **and** non-regression on Mangrove1; no major new temporal artifacts. |
| M12-3: if plateau | One budget-relaxed Transformer or quantization of a high-quality checkpoint, chosen based on whether capacity or deployment FLOPs/VRAM is the bottleneck | State target GFLOPs/latency and compare to M12-1, avoiding dozens of grid runs. |

Then, and only then, pursue the separate ReAE **two-frame flicker** correction track, unless temporal defect actively prevents quality selection.

## 4. Training and validation implementation rules

1. **Do not modify a source file participating in an active checkpoint fingerprint**, especially `tools/train_b2b_moe_ta_distill_ddp_cache_safe.py` / relevant objective/teacher cache modules; a GitHub PR does not require immediate sync to an occupied school GPU server.
2. Reuse existing trainer/cache contracts where practical; only add a new trainer when the actual objective/gradient/data interfaces materially differ. No arbitrary file renaming or deleting old experiments.
3. For the M12-1 run, *first* prove teacher/output alignment and end-to-end gradient reaches only the intended Transformer. Then run one short GPU smoke, then proceed to the bounded quality recovery. This is a correctness gate, not a request for hundreds of adversarial microtests.
4. Save at moderate cadence and keep enough checkpoints to compare the actual learning trend. Use fresh run directories; preserve M11-B baseline and prior T1 data.
5. Do **not** place two full restored 249-frame 2160p videos inside every val13 step. Run custom full-origin comparisons less frequently and outside the training GPU process when needed; reuse bank identity and exact frame origins.
6. Any result table must distinguish frozen teacher (Original complete SwiftVR vs M11-B+Original ReAE), latent targets, decoder version, source GT availability, MAC convention, measured runtime and source code hash.
7. A simple, measurable success on the two representative videos is more useful than a long speculative sweep.

## 5. First instructions for a new chat / Codex agent

- Read `docs/M11B_TRANSFORMER_V1_MILESTONE.md` plus this document and `docs/m11_transformer_budget_gate.md`.
- Summarize existing M11-B training objective and teacher-cache coverage from **real code/config**, not assumed from this planning document.
- Decide exactly **one** supervised teacher endpoint route with matching latent/decoder provenance; if in doubt show the specific mismatch rather than guessing.
- Make a compact plan: (1) aligned-target check, (2) one smoke/gradient gate, (3) bounded fine-tuning on M11-B+Original, (4) full-origin Wetland1/Mangrove1 visual gate.
- Avoid premature continuation of A1/T1/T1B, long 500k schedules or numerous phase-conditioned texture losses.
- All commits/PRs go to **LIUTIGHE/SwiftVR only**, branch `research/m12-transformer-quality-recovery`, targeting **fork main**, never upstream.

## 6. What remains unknown

- Actual Original vs M11-B end-to-end matched CUDA latency and FFN router overhead at each resolution.
- Whether the Original SwiftVR *teacher latents* share exact encoder, condition, timestamp and scale coordinates with M11-B restored latents. Must check before direct latent KD.
- What the recent long T1B-A continuation finally produced (reported to show little numerical effect, no terminal val record here).
- Whether M11-B+Original brightness flicker is fully decoder-caused versus a decoder/latent interaction. Current comparison supports keeping the issues separate but does not isolate root cause.
- Whether any new objective will improve real-custom video quality without GT; success must be confirmed in real native-resolution visual playback.
