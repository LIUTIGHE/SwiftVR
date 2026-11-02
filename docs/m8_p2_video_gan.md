# SwiftVR P2: video-adversarial post-training on P1-P5000

This implementation **does not** modify P1 trainer, inference, original checkpoints, A1 weights, or model FLOPs. Independent run and explicit warm start.

## What is new, and what is not

P1-R1500/P1-P5000: user visual inspection on Wetland1 found improved temporal consistency but insufficient detail; P1-P5000 a little better. This is a **visual observation**, not evidence that GT LPIPS alone caused the delta: training durations differed. B1 found fixed weak-frame positions for C24/C48 and mild Original brightness flicker. No new result is assumed.

P2 starts at `outputs/b2b/p1_gt_perceptual_v1/checkpoints/step_00005000`.
Frozen Original Encoder and M9-A1 E99, train **M8 only** through actual A1 reconstruction. GT L1 weight 1, GT LPIPS weight 0.1, GT-relative temporal MSE weight 1, router balance 0.01, matching P1-P.

A *training-only* video discriminator is new:
- 13 consecutive 384×384 output frames, identical paired random HR 128×128 crops for real/fake.
- 3D spectral-normalized PatchGAN gets 6 channels: **RGB and adjacent-frame RGB differences**. First difference frame is zero, no temporal reorder; each four-frame phase remains represented.
- Hinge D: mean[ReLU(1-D(GT))]+mean[ReLU(1+D(fake.detach()))].
- G: old P1-P objective + `lambda_adv * (-mean(D(fake)))`.
- D updates once per G update; D grads never pass to M8. During G update, D is in eval mode with weights frozen, but input gradients pass through D into M8. No D features / GAN are deployed.
- Initial `lambda_adv=0.01`, linear warmup 100 P2 steps, G LR 1e-6, D Adam LR 1e-4, betas (0,0.99), spectral normalization. First principled test, **not an optimized hyperparameter**.

The discriminator can still invent plausible-looking texture or destabilize true video motion. It does not have ground-truth object correspondences and cannot *guarantee* recovery of original fine detail. The entire 13-frame crop is used, but the causal streaming long-video protocol is not changed and not made equivalent by P2.

## Files

```text
swiftvr/training/m8_p2_gan.py       discriminator, paired crops, GAN losses
tools/train_m8_p2_video_gan.py      new P2 DDP training/checkpoint/validation
tools/run_m8_p2_gan.sh             explicit foreground launcher
tests/test_m8_p2_video_gan.py      CPU assertions
docs/m8_p2_video_gan.md            this note
```

## Before full GPU run

Run on server with existing SwiftVR environment (the four HR shards and original custom reference from P1 are reused, not regenerated):

```bash
cd /data1/a/SwiftVR
python -m unittest discover -s tests -p 'test_m8_p2_video_gan.py'
bash -n tools/run_m8_p2_gan.sh
```

Recommended **real checkpoint** two-update smoke (new directory, not the final run; four HR shards already complete):

```bash
CUDA_VISIBLE_DEVICES=7 python tools/train_m8_p2_video_gan.py \
  --base-checkpoint checkpoints_prompt_free_no_time \
  --student-init outputs/b2b/p1_gt_perceptual_v1/checkpoints/step_00005000 \
  --decoder-checkpoint outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder \
  --legacy-view-cache outputs/b2a/cache_stage_a200k_train_full \
  --val-view-cache outputs/b2a/cache_stage_a200k_val13 \
  --ultra-pairs outputs/data_audit/ultravideo_hr_train_v1/shard_0/paired_views.jsonl \
  --ultra-pairs outputs/data_audit/ultravideo_hr_train_v1/shard_1/paired_views.jsonl \
  --ultra-pairs outputs/data_audit/ultravideo_hr_train_v1/shard_2/paired_views.jsonl \
  --ultra-pairs outputs/data_audit/ultravideo_hr_train_v1/shard_3/paired_views.jsonl \
  --lpips-weights outputs/local_weights/lpips_alex_complete.pt \
  --output-dir outputs/b2b/p2_video_gan_smoke2_v1 \
  --path-root . --batch-size 1 --gradient-accumulation-steps 1 --num-workers 0 \
  --smoke-steps 2 --log-every 1
```

This smoke checks M8+discriminator nonzero grads, frozen Encoder/A1 grad absence, NaN/inf, and checkpoint serialization. It **does not** verify visual quality or distributed/snapshot-resume correctness. A full four-GPU run should begin only after the real GPU smoke succeeds.

## Formal run (1,000 new updates)

```bash
cd /data1/a/SwiftVR
bash tools/run_m8_p2_gan.sh
```

Defaults (all explicit in launcher):

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
local batch=1, accum=16, GPUs=4 -> effective batch64
generator LR=1e-6, scheduler horizon1000, warmup50
discriminator LR=1e-4
GAN lambda=0.01 ramp 100 steps
HR crop128, discriminator width32
snapshot+val=0,250,500,750,1000
run=outputs/b2b/p2_video_gan_p1p5000_v1
```

Default `--student-init` is P1-P step5000. The P2 checkpoint step0 is **equal to that P1-P5000 model**, but P2 step250 means 250 new GAN steps, **not step5250 in the metadata**. All model paths, checkpoint source hashes, and settings recorded in `run_config.json`.

Loss & critic logs: `train_log.jsonl` contains `pixel_l1`, `lpips`, `temporal_mse`, `d_loss`, `g_adv`, `gan_strength`, `critic_real/fake/margin`, generator and D gradient norms. Compare these together, not only critic score; a larger critic margin does not mean a better video.

Existing **full** Wetland1/Mangrove1 and val13 are reused without new Original or TinyCNN inference: P2's custom hook only computes Current at each checkpoint. ROI outputs stay native-pixel lossless FFV1 and phase-labeled PNG; no resize of ROI or FPS-based frame remapping.

## Stopping / rollback

After 250/500/1000 new updates, review **native ROI**, especially:
- moving wetland tree/grass texture vs P1-P5000, Original, LQ, TinyCNN;
- former phase-0 weak frames: quality **and** physically correct texture motion;
- camera motion, luminance flicker, repeated texture crawling, hallucinated lines;
- GT val13 PSNR/SSIM/LPIPS and phase MSE (GT correctness veto).

If texture sharpens but temporal popping/flicker noticeably returns, or false patterns appear, do not promote that step. Preserve all checkpoints. Do not auto-increase GAN weight or unfreeze A1. Earliest good step can be selected by visual review. No training-speed or deployment-FLOPs claim from additional offline discriminator compute.

### Resume

Only from an **explicit P2** checkpoint with generator, discriminator, both optimizers, RNG, cursor; never resume P1-P optimizer state as P2:

```bash
RESUME=outputs/b2b/p2_video_gan_p1p5000_v1/checkpoints/step_00000500 \
bash tools/run_m8_p2_gan.sh
```

Keep run path, four ranks, local batch1/accum16, optimizer/loss/scheduler, source paths, and current code/weights identical; no auto checkpoint selection. For changes use a **new run** with new lineage.

## Verification boundary

CPU unit tests cover 3D critic shape, temporal-difference signal, paired crop identity, hinge sign, detached D step, frozen D/G input gradients, safetensors discriminator round trip, GAN weight ramp, and launcher CLI/syntax. They do not simulate full DDP/GPU restoration. Real two-step CUDA smoke and full visual results must be reported separately.
