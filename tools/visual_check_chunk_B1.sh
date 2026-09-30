#!/usr/bin/env bash
# Same input bytes and model, different outer chunk lengths. No training or TinyCNN.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-7}"
PROBE_ROOT="${PROBE_ROOT:-outputs/diagnostics/chunk_B1_97f_v1}"
ROI="${ROI:-960,540,960,960}"  # x,y,w,h in unpadded SR output pixels; no resize.
INPUT1="${INPUT1:-../data/yuv_S_10bit/720p_mp4/S5_Wetland1_1920x1080p444_10bit.mp4}"
INPUT2="${INPUT2:-../data/yuv_S_10bit/720p_mp4/S5_Mangrove1_1920x1080p444_10bit.mp4}"
CURRENT_TRANSFORMER="${CURRENT_TRANSFORMER:-outputs/b2b/m8a_fullboost_ta_5k_v1/checkpoints/step_00001500}"
BASE="checkpoints_prompt_free_no_time"
A1="outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder"
ORIGINAL="checkpoints"

for file in "$INPUT1" "$INPUT2" "$BASE/reae.safetensors" \
  "$CURRENT_TRANSFORMER/transformer/config.json" "$ORIGINAL/reae.safetensors" \
  "$ORIGINAL/prompt_embedding.safetensors"; do
  test -f "$file" || { echo "Missing required file: $file" >&2; exit 1; }
done
test -d "$A1" || { echo "Missing A1 checkpoint: $A1" >&2; exit 1; }
if [[ -e "$PROBE_ROOT" ]]; then
  echo "Use a fresh PROBE_ROOT; existing results are not deleted: $PROBE_ROOT" >&2
  exit 1
fi
mkdir -p "$PROBE_ROOT"
exec > >(tee "$PROBE_ROOT/run.log") 2>&1

python -m unittest discover -s tests -p 'test_streaming_chunk_diagnostics.py'

INPUTS=("$INPUT1" "$INPUT2")
NAMES=("Wetland1" "Mangrove1")
for i in 0 1; do
  ROOT_VIDEO="$PROBE_ROOT/${NAMES[$i]}"
  FRAMES="$ROOT_VIDEO/input_png"
  mkdir -p "$FRAMES"
  # Decode once into lossless PNGs. Never change FPS, sample every other frame,
  # normalize duration, resize, or reset the origin differently between runs.
  ffmpeg -nostdin -v error -n -i "${INPUTS[$i]}" -map 0:v:0 \
    -frames:v 97 -vsync 0 -start_number 0 "$FRAMES/%08d.png"
  python - "$FRAMES" <<'PY'
from pathlib import Path
import sys
paths = sorted(Path(sys.argv[1]).glob('*.png'))
expected = [f'{i:08d}.png' for i in range(97)]
if [p.name for p in paths] != expected:
    raise SystemExit(f'Need exactly source frames 0..96; got {len(paths)} PNGs')
print(f'Input frame prefix: {len(paths)} frames, no temporal resampling')
PY

  # Current candidate: change only --clip-len. A1 E99 stays fixed.
  for C in 24 48; do
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/inference_custom_components.py \
      --input "$FRAMES" --output "$ROOT_VIDEO/current_c${C}/png" \
      --base-checkpoint "$BASE" --transformer-checkpoint "$CURRENT_TRANSFORMER" \
      --transformer-type moe --decoder-type m9a1 --decoder-checkpoint "$A1" \
      --upscale 3 --clip-len "$C" --dit-overlap 0 --dtype bfloat16 \
      --attention-backend sdpa --png \
      --chunk-trace "$ROOT_VIDEO/current_c${C}/trace"
  done
  python tools/compare_streaming_chunk_traces.py \
    --a "$ROOT_VIDEO/current_c24/trace" --b "$ROOT_VIDEO/current_c48/trace" \
    --crop "$ROI" --fps 30 --output-dir "$ROOT_VIDEO/current_c24_vs_c48"

  # Original conditional SwiftVR, NOT Stage-A. Separate paired comparison.
  for C in 24 48; do
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/inference.py \
      --input "$FRAMES" --output "$ROOT_VIDEO/original_c${C}/png" \
      --checkpoint "$ORIGINAL" --upscale 3 --clip-len "$C" --dit-overlap 0 \
      --dtype bfloat16 --attention_backend sdpa --png \
      --chunk-trace "$ROOT_VIDEO/original_c${C}/trace"
  done
  python tools/compare_streaming_chunk_traces.py \
    --a "$ROOT_VIDEO/original_c24/trace" --b "$ROOT_VIDEO/original_c48/trace" \
    --crop "$ROI" --fps 30 --output-dir "$ROOT_VIDEO/original_c24_vs_c48"
done
