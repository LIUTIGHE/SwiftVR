#!/usr/bin/env bash
# Explicit control/main launcher. Nothing is launched in the background.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${LPIPS_WEIGHTS:?Set the complete locally exported Alex-LPIPS state path}"
: "${HR_ROOT:?Set the HR root containing shard_0..3/paired_views.jsonl}"
: "${CUSTOM_REF_ROOT:?Prepare the two custom references before formal training}"
: "${RUN_DIR:?Choose a new directory for this specific recipe}"
RECIPE="${RECIPE:-perceptual}"
case "$RECIPE" in
  reconstruction) PWEIGHT=0 ;;
  perceptual) PWEIGHT="${LPIPS_WEIGHT:-0.1}" ;;
  *) echo 'RECIPE must be reconstruction or perceptual' >&2; exit 2 ;;
esac
# Default global batch = 4 GPUs * local2 * accum8 = 64 views/update.
# If memory requires local1, use ACCUM=16 rather than silently halving exposure.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
mkdir -p "$(dirname "$RUN_DIR")"
torchrun --standalone --nproc_per_node="${GPUS:-4}" \
  tools/train_m8_task_posttrain.py \
  --base-checkpoint checkpoints_prompt_free_no_time \
  --student-init "${STUDENT_INIT:-outputs/b2b/m8a_fullboost_ta_5k_v1/checkpoints/step_00001500}" \
  --decoder-checkpoint outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder \
  --legacy-view-cache outputs/b2a/cache_stage_a200k_train_full \
  --val-view-cache outputs/b2a/cache_stage_a200k_val13 \
  --ultra-pairs "$HR_ROOT/shard_0/paired_views.jsonl" \
  --ultra-pairs "$HR_ROOT/shard_1/paired_views.jsonl" \
  --ultra-pairs "$HR_ROOT/shard_2/paired_views.jsonl" \
  --ultra-pairs "$HR_ROOT/shard_3/paired_views.jsonl" \
  --path-root . --output-dir "$RUN_DIR" \
  --lpips-weights "$LPIPS_WEIGHTS" --lpips-weight "$PWEIGHT" \
  --pixel-weight 1 --temporal-weight 1 --router-weight 0.01 \
  --batch-size "${LOCAL_BATCH:-2}" --gradient-accumulation-steps "${ACCUM:-8}" \
  --num-workers "${WORKERS:-4}" --prefetch-factor 2 \
  --learning-rate 5e-6 --warmup-steps 100 \
  --max-steps "${MAX_STEPS:-5000}" --schedule-steps 10000 \
  --validate-every 500 --save-every 500 --log-every 10 \
  --custom-ref-root "$CUSTOM_REF_ROOT" \
  "$@" 2>&1 | tee "$RUN_DIR.console.log"
