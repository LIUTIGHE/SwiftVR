#!/usr/bin/env bash
# B1: same MOVING source window + same model, different canonical chunk lengths.
# No training, no alternate forward, no TinyCNN/FPS temporal remapping.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${GPU:-7}"
VIDEO="${VIDEO:-Wetland1}"      # Wetland1, Mangrove1, or both
: "${START_FRAME:?Set START_FRAME to the ORIGINAL decoded frame index where camera motion starts. Use tools/preview_video_motion_windows.py.}"
FRAMES="${FRAMES:-161}"         # Must be 4k+1. Increase to 193/241 if useful.
ROI="${ROI:-960,540,960,960}"  # x,y,w,h in unpadded SR output pixels; never resized
INPUT1="${INPUT1:-../data/yuv_S_10bit/720p_mp4/S5_Wetland1_1920x1080p444_10bit.mp4}"
INPUT2="${INPUT2:-../data/yuv_S_10bit/720p_mp4/S5_Mangrove1_1920x1080p444_10bit.mp4}"
CURRENT_TRANSFORMER="${CURRENT_TRANSFORMER:-outputs/b2b/m8a_fullboost_ta_5k_v1/checkpoints/step_00001500}"
BASE="checkpoints_prompt_free_no_time"
A1="outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder"
ORIGINAL="checkpoints"

if ! [[ "$START_FRAME" =~ ^[0-9]+$ && "$FRAMES" =~ ^[0-9]+$ ]]; then
  echo "START_FRAME and FRAMES must be nonnegative integer strings." >&2
  exit 2
fi
if (( FRAMES < 97 || (FRAMES - 1) % 4 != 0 )); then
  echo "FRAMES must be >=97 and 4k+1 (e.g. 161, 193, 241)." >&2
  exit 2
fi
END_FRAME=$((START_FRAME + FRAMES - 1))
PROBE_ROOT="${PROBE_ROOT:-outputs/diagnostics/chunk_B1_${VIDEO}_s${START_FRAME}_${FRAMES}f_v2}"

case "$VIDEO" in
  Wetland1)
    INPUTS=("$INPUT1"); NAMES=("Wetland1") ;;
  Mangrove1)
    INPUTS=("$INPUT2"); NAMES=("Mangrove1") ;;
  both)
    INPUTS=("$INPUT1" "$INPUT2"); NAMES=("Wetland1" "Mangrove1") ;;
  *)
    echo "VIDEO must be Wetland1, Mangrove1, or both." >&2
    exit 2 ;;
esac

for file in "${INPUTS[@]}" "$BASE/reae.safetensors" \
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

echo "[B1] VIDEO=$VIDEO START_FRAME=$START_FRAME FRAMES=$FRAMES END_FRAME=$END_FRAME ROI=$ROI"
echo "[B1] The original source frame index is deliberately NOT converted to time/FPS."
echo "[B1] Both chunk lengths restart Encoder/DiT/Decoder from the same selected physical frame."
echo "[B1] This preserves pairing within this run, but resets the model's 4-phase origin."
python -m unittest discover -s tests -p 'test_streaming_chunk_diagnostics.py'

for i in "${!INPUTS[@]}"; do
  ROOT_VIDEO="$PROBE_ROOT/${NAMES[$i]}"
  PNG_DIR="$ROOT_VIDEO/input_png"
  mkdir -p "$PNG_DIR"

  # Exact zero-based decoded frame window: [START_FRAME, END_FRAME], inclusive.
  # The select filter uses frame number n, not approximate -ss/keyframe seeking.
  # -vsync 0 prevents FFmpeg from duplicating/dropping frames to fit an FPS.
  ffmpeg -nostdin -v error -n -i "${INPUTS[$i]}" -map 0:v:0 \
    -vf "select='between(n,${START_FRAME},${END_FRAME})'" \
    -vsync 0 -frames:v "$FRAMES" -start_number 0 "$PNG_DIR/%08d.png"

  python - "$PNG_DIR" "$FRAMES" "${INPUTS[$i]}" "$START_FRAME" "$END_FRAME" "$ROOT_VIDEO/window.json" <<'PY'
import json
from pathlib import Path
import sys

folder, count, source, start, end, metadata = sys.argv[1:]
count, start, end = map(int, (count, start, end))
paths = sorted(Path(folder).glob("*.png"))
expected = [f"{i:08d}.png" for i in range(count)]
if [p.name for p in paths] != expected:
    raise SystemExit(
        f"Expected {count} frames from physical [{start},{end}]; got {len(paths)}. "
        "Move START_FRAME earlier or decrease FRAMES."
    )
meta = {
    "source": str(Path(source).resolve()),
    "physical_start_frame": start,
    "physical_end_frame_inclusive": end,
    "frames": count,
    "local_index_to_physical_index": f"source_index = {start} + local_index",
    "note": "Streaming state and internal 4-phase origin reset at this input-window start.",
}
Path(metadata).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(json.dumps(meta, indent=2), flush=True)
PY

  # Same components. ONLY --clip-len changes within each paired comparison.
  for C in 24 48; do
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/inference_custom_components.py \
      --input "$PNG_DIR" --output "$ROOT_VIDEO/current_c${C}/png" \
      --base-checkpoint "$BASE" --transformer-checkpoint "$CURRENT_TRANSFORMER" \
      --transformer-type moe --decoder-type m9a1 --decoder-checkpoint "$A1" \
      --upscale 3 --clip-len "$C" --dit-overlap 0 --dtype bfloat16 \
      --attention-backend sdpa --png \
      --chunk-trace "$ROOT_VIDEO/current_c${C}/trace"
  done
  python tools/compare_streaming_chunk_traces.py \
    --a "$ROOT_VIDEO/current_c24/trace" --b "$ROOT_VIDEO/current_c48/trace" \
    --crop "$ROI" --fps 30 --output-dir "$ROOT_VIDEO/current_c24_vs_c48"

  # Original conditional SwiftVR, NOT Stage-A. Independent C24/C48 pairing.
  for C in 24 48; do
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/inference.py \
      --input "$PNG_DIR" --output "$ROOT_VIDEO/original_c${C}/png" \
      --checkpoint "$ORIGINAL" --upscale 3 --clip-len "$C" --dit-overlap 0 \
      --dtype bfloat16 --attention_backend sdpa --png \
      --chunk-trace "$ROOT_VIDEO/original_c${C}/trace"
  done
  python tools/compare_streaming_chunk_traces.py \
    --a "$ROOT_VIDEO/original_c24/trace" --b "$ROOT_VIDEO/original_c48/trace" \
    --crop "$ROI" --fps 30 --output-dir "$ROOT_VIDEO/original_c24_vs_c48"
done
