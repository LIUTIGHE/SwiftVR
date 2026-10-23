# M10 next gate: four-phase / same-physical-frame diagnosis

This commit does **not** change any model, checkpoint, training objective,
streaming semantics, or inference FLOPs. M9-500G remains the baseline.
The two checks decide whether Stage-A target construction and the final decoder
phases need attention before another recovery-training run.

## Files to synchronize manually from GitHub Web

```
tools/visualize_m9_val13_components.py   # existing tool, opt-in additions
tools/m10_phase_metrics.py
tools/diagnose_m10_phase_offsets.py
tests/test_m10_phase_diagnostics.py
docs/m10_phase_diagnostics.md
```

Run in the existing SwiftVR environment; no server-side git operation or new
model download is needed. The original M10 files must already be present.

```bash
cd /data1/a/SwiftVR
python -m unittest discover -s tests -p 'test_m10_phase_diagnostics.py' -v
```

CPU tests cover canonical 25/24/24/8 emission accounting, physical frame identity,
phase offsets, boundary exclusion, HF-filter agreement with M10, report writing,
shared teacher flicker, and synthetic PNG comparison end-to-end. They do not run
real GPU model inference. Full val13/custom inference must run on the server.

## 1. Exact val13, now including Stage-A latent decoded by A1

```bash
export BASE=checkpoints_prompt_free_no_time
export STAGEA=outputs/stage_a_distill_formal_v8_bs16_bf16/materialized_step_00200000
export A1_RUN=outputs/b2b/m9a1_factorized_m8a30k
export A1="$A1_RUN/checkpoints/epoch_099_step_00024552/tiny_decoder"
export D76=outputs/b2b/m8b_decoder76_m8a30k_warmup/checkpoints/epoch_008_step_00001984/tiny_decoder

CUDA_VISIBLE_DEVICES=7 python tools/visualize_m9_val13_components.py \
  --a1-run "$A1_RUN" --a1-checkpoint "$A1" \
  --stagea-checkpoint "$STAGEA" --base-checkpoint "$BASE" \
  --decoder76-checkpoint "$D76" \
  --include-stagea-a1 --phase-report \
  --device cuda --dtype bfloat16 --attention-backend sdpa \
  --frame-indices 0,1,2,3,4,5,6,7,8,9,10,11,12 \
  --panel-size 256 --fps 30 \
  --output-dir outputs/custom/m10_val13_phase
```

The Stage-A forward runs once per view; its exact z_SR is retained on CPU and
later decoded by the already-loaded A1. M8-A still uses the exact existing val
cache. No replacement teacher cache, new crop, new data views, or training occurs.
The original command's behavior/defaults are kept unless the new flags are used.

New column: `StageA+M9A1`, exported as `stagea_m9a1/`. New overall metrics:

```
stagea_m9a1_vs_gt
stagea_m9a1_vs_stagea_original_same_latent
```

Root and each sample directory also contain `phase_per_frame.csv` and
`phase_report.json`. Errors use float RGB outputs BEFORE PNG quantization;
HF is sigma1 5x5 Gaussian residual. GT errors are diagnostic only.

Phase is `(local_output_frame + 3) % 4`, based on the reset decoder and initial
3-frame trim. It is NOT the manifest/source frame number modulo 4. In 13-frame
views, phase3 occurs four times because of the first frame; other phases occur
three times. Report `exclude_first_frame` removes local t0 to balance counts and
excludes the temporal pair crossing t0. This is **not** a claim that such short
clips have reached streaming steady state. Temporal errors are unwarped and are
assigned to the destination phase; no cross-sample temporal differences are taken.

Read `StageA+M9A1 vs StageA+Orig` to test decoder compatibility with Stage-A
latents. Compare it to `M8A+M9A1 vs M8A+Orig`, but do not interpret RGB error alone
as detail quality. Inspect all four phases and GT/LQ independently: if both
student and teacher fail on the same phase, teacher matching can still be good.
The new contact sheets label local frame t and phase p. FPS30 is playback only.

## 2. Same source frame, two reset origins, using canonical streaming

First use Stage-A+Original to ask whether a teacher frame that is poor under
one origin becomes useful under the other. This is Stage-A, **not upstream's
conditional/prompted model**. Both origins use exactly the same checkpoint.

```bash
export INPUT=/data1/a/data/yuv_S_10bit/720p_mp4/S6_Elephant_1920x1080p444_10bit.mp4

CUDA_VISIBLE_DEVICES=7 python tools/diagnose_m10_phase_offsets.py run \
  --input "$INPUT" \
  --base-checkpoint "$BASE" --transformer-checkpoint "$STAGEA" \
  --transformer-type dense --decoder-type original \
  --start-frame 0 --frames 81 --clip-len 24 --upscale 3 \
  --dtype bfloat16 --attention-backend sdpa \
  --crop 960,540,960,960 --panel-width 640 --fps 30 \
  --output-dir outputs/custom/m10_stagea_offset01
```

The crop is the earlier custom visual ROI in UNPADDED OUTPUT pixels; set it to
the actual diamond-shirt ROI if different, or omit it to compare full frames.
It is applied **after** full-frame model inference, never to the model's input.
`start-frame` is an absolute zero-based SOURCE position; both windows must have
81 frames. Source positions 0..81 (82 frames total) are required for the defaults.
Using a later start is supported; filenames/labels retain absolute source IDs.

Preparation reads each source frame once with the existing decode dependency
(or reads RGB images), saves lossless PNG, and hardlinks/copies the same bytes:

```
input_offset0: source 0..80
input_offset1: source 1..81
```

No interpolation, FPS conversion, temporal resampling, or latent slicing is used.
Each 81-frame folder is passed in a separate subprocess to the UNMODIFIED
`scripts/inference_custom_components.py`. It independently resets and recomputes
E/T/D. Canonical input-folder frame names are preserved in output. `run.json`
records commands, source IDs, PNG hashes, phases, and actual chunk output mapping.

```
output_offset0/00000029.png   # physical source 29, local t29, phase0
output_offset1/00000029.png   # SAME physical source29, local t28, phase3
```

`comparison/frames/` contains all 80 common physical frames, with LQ-up, start0,
start1 and explicit t/p labels. `contact_sheet.png` selects the first eight valid
interior common frames; `comparison.mp4` plays ALL common frames (including
boundary frames for inspection). `per_frame.csv` flags interior frames.
Main report `middle_interiors` excludes FIRST and LAST chunks and two frames at
each MIDDLE edge in BOTH runs: 38 pairs for the default protocol. Phase counts
are reported and need not be equal. Each phase mean is computed separately.

The RGB/HF error here measures **origin sensitivity**, not distance to GT or
visual quality. HF RMS fields are explicitly marked `not_quality`. This tool
never selects the sharper-looking output as a training target automatically.
Changing the origin changes phase AND context/chunk composition; even a clear
improvement is not proof that TGrow alone caused the problem.

To inspect another crop without rerunning any model:

```bash
python tools/diagnose_m10_phase_offsets.py compare \
  --root outputs/custom/m10_stagea_offset01 \
  --comparison-name comparison_full --panel-width 640 --fps 30
```

The compare-only action requires NumPy/Pillow and optionally FFmpeg for MP4;
PNG analysis works without Diffusers, Decord, CUDA or Hugging Face access.
All output roots/comparison names must be fresh; nothing is silently overwritten.
If inference fails, `run.json` retains the exact commands for that run. Finish
the missing canonical outputs before invoking `compare`; missing/extra frames
are rejected rather than guessed/aligned by nearest index.

## Decisions after these checks

If Stage-A+A1 already preserves teacher detail, backbone recovery need not be
held back by a decoder-domain mismatch. If it fails, adjust teacher/decoder
alignment before insisting on matching both latent and deployed RGB targets.

If the SAME physical frame recovers its diamond texture under another origin,
origin-diverse teacher construction is worth investigating; inspect actual
structure, not just HF energy. If it does not recover, a different temporal
training signal may be needed. Neither outcome automatically launches training.
The next model experiments remain separate: backbone detail recovery and
worst-phase repair, each starting from the immutable M9-500G milestone.
