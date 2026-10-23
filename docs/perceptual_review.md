# Stage-3 perceptual and visual review

`tools/review_stage3_perceptual.py` evaluates several prompt-free Stage-3
checkpoints on the exact deterministic validation samples used by the
conditional-reference cache.

The tool deliberately runs outside the trainer. It:

1. loads the 5B student once and exports each requested checkpoint sequentially;
2. releases the 5B model and CUDA window caches;
3. computes LPIPS, DISTS, and MUSIQ with `pyiqa`;
4. saves selected comparison PNGs and side-by-side MP4 clips;
5. writes aggregate CSV/JSON, per-sample JSONL, TensorBoard curves, and an HTML report.

The conditional reference is read from the existing cache, so a second 5B
conditional model is never resident together with the student.

## Optional environment

Install the isolated evaluation dependencies after training:

```bash
python -m pip install -r requirements-perceptual.txt
```

The first metric run may download IQA weights into the PyTorch cache. Override
the location with `--metric-cache-dir`.

## Example: hard-removal base, step 300, and step 2000

```bash
cd /data1/a/SwiftVR

CUDA_VISIBLE_DEVICES=6 \
python tools/review_stage3_perceptual.py \
  --student-base-checkpoint \
    /data1/a/SwiftVR/checkpoints_prompt_free_no_time \
  --include-base \
  --student-checkpoint \
    step300=/data1/a/SwiftVR/outputs/stage3_recon_ddp_gate_from100/checkpoints/step_00000300 \
  --student-checkpoint \
    step2000=/data1/a/SwiftVR/outputs/stage3_recon_ddp_gate_from100/checkpoints/step_00002000 \
  --reference-cache \
    /data1/a/SwiftVR/outputs/reference_cache_plain_val10 \
  --val-manifest \
    /data1/a/SwiftVR/manifests/vsr_triplets_plain_val13_newserver.jsonl \
  --path-root /data1/a/SwiftVR \
  --val-split val \
  --clip-length 13 \
  --crop-size 128 \
  --scale 3 \
  --max-samples 10 \
  --device cuda \
  --metric-device cuda \
  --dtype auto \
  --attention-backend sdpa \
  --metrics lpips,dists,musiq \
  --metric-batch-size 1 \
  --visual-frame-indices 0,6,12 \
  --video-fps 8 \
  --difference-scale 4 \
  --output-dir \
    /data1/a/SwiftVR/outputs/perceptual_review_0_300_2000
```

If metric initialization or weight download fails after student predictions were
already exported, rerun the same command with:

```bash
--reuse-predictions
```

Do not combine `--reuse-predictions` with `--overwrite`.

## Outputs

```text
perceptual_review_0_300_2000/
├── metadata.json
├── summary.json
├── summary.csv
├── per_sample_metrics.jsonl
├── samples.json
├── report.html
├── tensorboard/
├── prediction_cache/
│   ├── step0/
│   ├── step300/
│   └── step2000/
└── visuals/
    └── <sample>/
        ├── comparison.mp4
        ├── difference_to_gt.mp4
        ├── frame_000.png
        ├── diff_000.png
        └── ...
```

Open `report.html` locally or through a static HTTP server. The comparison video
uses this column order:

```text
GT | LQ bicubic | conditional reference | step0 | step300 | step2000
```

The difference video visualizes amplified absolute RGB error against GT. It is a
diagnostic aid rather than a perceptual-quality metric.

## Metric interpretation

- `gt_lpips` and `gt_dists`: full-reference perceptual distance to GT; lower is better.
- `reference_lpips` and `reference_dists`: distance to the original conditional reference; lower means closer to its behavior, not necessarily better quality.
- `nr_musiq`: no-reference perceptual-quality score; higher is better.
- `gt_psnr`, `gt_ssim`, `gt_mae`, and `gt_rmse`: retained as distortion diagnostics.

TensorBoard records student checkpoints at their real global steps and repeats
the LQ/reference baselines at those steps for direct comparison.

## Reproducibility

The tool records:

- checkpoint paths and inferred global steps;
- validation-manifest SHA256;
- reference-cache path and dataset geometry;
- pyiqa version and metric directions;
- selected sample count and visual frame indices.

Keep `summary.csv`, `per_sample_metrics.jsonl`, and `metadata.json` with any
subjective conclusions drawn from the videos.
