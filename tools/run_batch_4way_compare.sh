#!/usr/bin/env bash
set -euo pipefail

# Batch 720p->3x comparison inference with resume-safe outputs.
# Each GPU processes a disjoint subset of videos sequentially.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT_GLOB="${INPUT_GLOB:-../data/yuv_S_10bit/720p_mp4/*.mp4}"
BASIC_DIR="${BASIC_DIR:-../data/yuv_S_10bit/AISR_pred}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/batch_4way_compare}"

ORIGINAL_CKPT="${ORIGINAL_CKPT:-checkpoints}"
BASE_CKPT="${BASE_CKPT:-checkpoints_prompt_free_no_time}"
OURS_TRANSFORMER="${OURS_TRANSFORMER:-outputs/b2b/m8a_d1024_l20_gate200k/checkpoints/step_00030000}"
OURS_TRANSFORMER_TYPE="${OURS_TRANSFORMER_TYPE:-moe}"
OURS_DECODER_TYPE="${OURS_DECODER_TYPE:-m9a1}"
OURS_DECODER="${OURS_DECODER:-outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
UPSCALE="${UPSCALE:-3}"
CLIP_LEN="${CLIP_LEN:-24}"
DIT_OVERLAP="${DIT_OVERLAP:-0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN="${ATTN:-sdpa}"
QUALITY="${QUALITY:-95}"
QUEUE_SIZE="${QUEUE_SIZE:-2}"
BASIC_INDEX_SCALE="${BASIC_INDEX_SCALE:-2}"
BASIC_INDEX_OFFSET="${BASIC_INDEX_OFFSET:-0}"
COMPARE_PIX_FMT="${COMPARE_PIX_FMT:-yuv444p}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs in GPU_IDS=$GPU_IDS" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

valid_video() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height \
    -of csv=p=0 "$path" >/dev/null 2>&1
}

shopt -s nullglob
inputs=( $INPUT_GLOB )
shopt -u nullglob
if [[ "${#inputs[@]}" -eq 0 ]]; then
  echo "No inputs matched: $INPUT_GLOB" >&2
  exit 2
fi

resolve_basic() {
  local input="$1"
  local stem
  stem="$(basename "${input%.*}")"
  local exact="$BASIC_DIR/${stem}_SR_YUV444_RC1.0.mp4"
  if [[ -f "$exact" ]]; then
    printf '%s\n' "$exact"
    return 0
  fi

  shopt -s nullglob
  local matches=( "$BASIC_DIR/${stem}"*.mp4 )
  shopt -u nullglob
  if [[ "${#matches[@]}" -eq 1 ]]; then
    printf '%s\n' "${matches[0]}"
    return 0
  fi

  echo "Cannot uniquely resolve BasicCNN for $input under $BASIC_DIR" >&2
  printf 'matches:' >&2
  printf ' %q' "${matches[@]}" >&2
  printf '\n' >&2
  return 1
}

run_one() {
  local gpu="$1"
  local input="$2"
  local stem out basic

  stem="$(basename "${input%.*}")"
  out="$OUTPUT_ROOT/$stem"
  basic="$(resolve_basic "$input")"
  mkdir -p "$out"

  if valid_video "$out/compare/quadrant.mp4" && \
     valid_video "$out/compare/vertical_quarters.mp4"; then
    echo "[$(date '+%F %T')] skip complete :: $stem" >&2
    return 0
  fi

  if valid_video "$out/original_swiftvr.mp4"; then
    echo "[$(date '+%F %T')] reuse :: $stem :: Original SwiftVR" >&2
  else
    rm -f "$out/original_swiftvr.mp4"
    echo "[$(date '+%F %T')] GPU $gpu :: $stem :: Original SwiftVR" >&2
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/inference.py \
      --input "$input" \
      --output "$out/original_swiftvr.mp4" \
      --checkpoint "$ORIGINAL_CKPT" \
      --upscale "$UPSCALE" \
      --clip-len "$CLIP_LEN" \
      --dit-overlap "$DIT_OVERLAP" \
      --dtype "$DTYPE" \
      --attention_backend "$ATTN" \
      --quality "$QUALITY" \
      --save-format yuv444p \
      --queue-size "$QUEUE_SIZE" \
      --quiet
  fi

  if valid_video "$out/ours.mp4"; then
    echo "[$(date '+%F %T')] reuse :: $stem :: Ours" >&2
  else
    rm -f "$out/ours.mp4"
    echo "[$(date '+%F %T')] GPU $gpu :: $stem :: Ours" >&2
    ours_cmd=(
      python scripts/inference_custom_components.py
      --input "$input"
      --output "$out/ours.mp4"
      --base-checkpoint "$BASE_CKPT"
      --transformer-checkpoint "$OURS_TRANSFORMER"
      --transformer-type "$OURS_TRANSFORMER_TYPE"
      --decoder-type "$OURS_DECODER_TYPE"
      --upscale "$UPSCALE"
      --clip-len "$CLIP_LEN"
      --dit-overlap "$DIT_OVERLAP"
      --dtype "$DTYPE"
      --attention-backend "$ATTN"
      --quality "$QUALITY"
      --save-format yuv444p
      --queue-size "$QUEUE_SIZE"
      --quiet
    )
    if [[ "$OURS_DECODER_TYPE" != "original" ]]; then
      ours_cmd+=(--decoder-checkpoint "$OURS_DECODER")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "${ours_cmd[@]}"
  fi

  rm -rf "$out/compare"
  echo "[$(date '+%F %T')] CPU :: $stem :: compose" >&2
  python tools/compose_4way_video.py \
    --lq "$input" \
    --basiccnn "$basic" \
    --original "$out/original_swiftvr.mp4" \
    --ours "$out/ours.mp4" \
    --output-dir "$out/compare" \
    --basiccnn-index-scale "$BASIC_INDEX_SCALE" \
    --basiccnn-index-offset "$BASIC_INDEX_OFFSET" \
    --pix-fmt "$COMPARE_PIX_FMT"

  printf '%s\n' "$basic" > "$out/basiccnn_source.txt"
  echo "[$(date '+%F %T')] done :: $stem" >&2
}

pids=()
for gi in "${!GPUS[@]}"; do
  gpu="${GPUS[$gi]}"
  (
    for i in "${!inputs[@]}"; do
      if (( i % ${#GPUS[@]} == gi )); then
        run_one "$gpu" "${inputs[$i]}"
      fi
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
