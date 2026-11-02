# SharedVideoAutoencoder

`SharedVideoAutoencoder` is a reusable deterministic causal video latent codec assembled from:

- the original SwiftVR ReAE encoder; and
- a ReAE-family decoder, normally the Stage-B1 Slim100 decoder `[256,128,64,64]`.

It is deliberately **not** called a VAE. The encoder emits one deterministic 48-channel latent tensor; there is no `mu/logvar`, stochastic reparameterization, or KL prior.

## Current intended use

The intended multi-task pattern is:

```text
input video
   -> frozen shared encoder
   -> task-specific Transformer / UNet
   -> frozen shared decoder
   -> restored RGB video
```

The codec itself should normally stay frozen while a task network learns a mapping inside the common latent space.

## Export the current Stage-B1 Slim100 codec

```bash
python tools/export_shared_video_autoencoder.py \
  --base-checkpoint /data1/a/SwiftVR/checkpoints_prompt_free_no_time \
  --slim-decoder /data1/a/SwiftVR/outputs/stage_b1_reae_slim100_teacher_distill/checkpoints/epoch_050_step_00012400/tiny_decoder \
  --output-dir /data1/a/SwiftVR/outputs/shared_video_codec_slim100
```

The exported folder contains:

```text
shared_video_codec_slim100/
  config.json
  model.safetensors
```

It can then be copied to another project and loaded with:

```python
import torch
from swiftvr.models import SharedVideoAutoencoder

codec = SharedVideoAutoencoder.from_pretrained(
    "/path/to/shared_video_codec_slim100",
    device="cuda",
    dtype=torch.bfloat16,
).freeze()
```

You can also assemble it directly from the current two component checkpoints:

```python
codec = SharedVideoAutoencoder.from_component_checkpoints(
    "/path/to/checkpoints_prompt_free_no_time/reae.safetensors",
    "/path/to/slim100/checkpoint/tiny_decoder",
    device="cuda",
    dtype=torch.bfloat16,
).freeze()
```

Omit the Slim decoder path to wrap the original full ReAE encoder + decoder.

## Tensor contract

Whole-clip helper methods intentionally follow the SwiftVR training-clip convention:

```text
RGB input:  [B,T,3,H,W], T = 4k+1, H/W divisible by 16, normally in [0,1]
latent:     [B,(T+3)/4,48,H/16,W/16]
RGB output: [B,T,3,H,W]
```

For a `4k+1` clip, `encode()` repeats the final RGB frame three times before the two temporal x2 pooling stages. `decode()` performs x4 temporal growth and removes the first three causal decoder warm-up frames.

There is **no latent scaling to N(0,1)** and no stochastic sampling API. Do not treat this latent as a Stable-Diffusion-style Gaussian VAE latent.

## Training a task network with the codec frozen

```python
codec.freeze()

# Encoder is frozen; no graph is needed through the input encoding path.
with torch.no_grad():
    z_in = codec.encode(degraded_video)
    z_gt = codec.encode(clean_video)   # optional latent target

z_pred = task_network(z_in)

# Do NOT put this decode in torch.no_grad(). Codec parameters are frozen, but
# autograd must pass through the decoder to train task_network from RGB losses.
pred_rgb = codec.decode(z_pred, output_frames=degraded_video.shape[1])

loss_rgb = charbonnier(pred_rgb, clean_video)
loss_latent = (z_pred.float() - z_gt.float()).square().mean()
loss = loss_rgb + 0.1 * loss_latent
loss.backward()
```

## Why it may be a useful shared codec

1. **Common output domain.** Video SR, denoising, deblurring, compression-artifact removal, and related low-level tasks all aim at natural clean RGB video, so sharing a decoder is structurally reasonable.
2. **Relatively rich latent.** The 48-channel latent is much wider than a typical 4-channel image-generation VAE latent, which may help preserve low-level detail, phase, texture, and temporal evidence.
3. **Video-native causal structure.** ReAE uses temporal pooling/growth and MemBlocks rather than independent per-frame image coding.
4. **Light shared endpoint.** At the current 1920x1088 profiling geometry, the existing ReAE encoder is about 42.278 GMAC/output-frame and Slim100 is about 98.223 GMAC/output-frame, for about 140.5 GMAC/output-frame of shared codec compute before the task network.
5. **Decoder fidelity is already encouraging.** On the current 13-view B1 validation protocol, Slim100 reaches about 36.65 dB / 0.9715 SSIM against the original ReAE decoder on restoration latents and passed the visual artifact audit.

## Current limitations / what is NOT yet proven

1. **Standalone AE reconstruction has not yet been formally audited.** Slim100 was distilled mainly on Stage-A restoration latents `z_SR`, not on the raw encoder latent distribution `E(clean RGB)`. Shape compatibility is proven; universal reconstruction quality is not.
2. **Cross-degradation encoder robustness is unproven.** The ReAE encoder was trained in the SwiftVR restoration setting. Noise, blur, low light, compression errors, and other domains may shift the latent distribution differently.
3. **The decoder is not an unconditional generator.** Arbitrary Gaussian latents are not expected to decode to realistic video. A task network should keep its outputs near the latent support seen by ReAE/SwiftVR.
4. **No task-invariant latent prior is enforced.** There is no KL regularization or explicit domain alignment, so different task datasets may occupy different regions of the 48-channel latent space.
5. **SR uses target-resolution coding in SwiftVR.** SwiftVR first resizes the low-resolution RGB to the target spatial size and then encodes it. The codec is therefore not itself a learned x3/x4 SR decoder; spatial resolution changes must be handled outside or before the codec unless the architecture is changed.
6. **Whole-clip helper is intentionally strict.** `encode()/forward()` use `T=4k+1` and spatial multiples of 16. For arbitrary streaming chunk sizes, use SwiftVR's existing `StreamingTAE` path or add a task-specific streaming wrapper.
7. **Slim100 is a restoration-latent decoder compression result, not yet a universal codec result.** If a new task produces substantially out-of-distribution latents, the original full ReAE decoder should be kept as a reference during early experiments.

## Suggested minimum evaluation before claiming a universal shared codec

For each new task/domain, first test:

1. `clean -> encoder -> decoder -> clean` reconstruction with both original ReAE decoder and Slim100;
2. latent statistics of clean and degraded inputs (mean/std/channel RMS and feature distance);
3. frozen codec + small task-specific network on at least two degradation domains;
4. visual/temporal inspection, not only PSNR/SSIM.

A successful result on two or more tasks would support the stronger claim that this codec provides a reusable low-level video latent space. Until then, treat it as a promising SwiftVR-derived shared codec candidate rather than a validated universal VAE.
