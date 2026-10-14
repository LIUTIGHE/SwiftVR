#!/usr/bin/env bash
# P2 adversarial finetuning. P1-P step5000 remains immutable.
# Explicit foreground execution; no background GPU jobs or auto-resume.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${CUDA_VISIBLE_DEVICES:=4,5,6,7}"
export CUDA_VISIBLE_DEVICES
: "${HR_ROOT:=outputs/data_audit/ultravideo_hr_train_v1}"
: "${CUSTOM_REF_ROOT:=outputs/custom/p1_reference_v1}"
: "${LPIPS_WEIGHTS:=outputs/local_weights/lpips_alex_complete.pt}"
: "${STUDENT_INIT:=outputs/b2b/p1_gt_perceptual_v1/checkpoints/step_00005000}"
: "${RUN_DIR:=outputs/b2b/p2_video_gan_p1p5000_v1}"

# P1 full-run local2/accum8 can OOM after adding the 3D discriminator.
# Safer default retains global64: 4 GPUs x local1 x accumulation16.
: "${GPUS:=4}"
: "${LOCAL_BATCH:=1}"
: "${ACCUM:=16}"
: "${WORKERS:=4}"
: "${MAX_STEPS:=1000}"

test -f "$STUDENT_INIT/transformer/config.json" || { echo "Missing P1-P5000 checkpoint: $STUDENT_INIT" >&2; exit 1; }
test -f "$LPIPS_WEIGHTS" || { echo "Missing complete local LPIPS state" >&2; exit 1; }
test -f "$CUSTOM_REF_ROOT/reference.json" || { echo "Missing full custom reference bank" >&2; exit 1; }
for S in 0 1 2 3; do
  test -f "$HR_ROOT/shard_$S/paired_views.jsonl" || { echo "Missing complete HR shard $S" >&2; exit 1; }
done

if [[ -z "${RESUME:-}" && -e "$RUN_DIR" ]]; then
  echo "P2 RUN_DIR already exists; refusing overwrite. Set RESUME to an explicit P2 checkpoint." >&2
  exit 1
fi
mkdir -p "$(dirname "$RUN_DIR")"
echo "[P2] init=$STUDENT_INIT"
echo "[P2] run=$RUN_DIR; GPUs=$CUDA_VISIBLE_DEVICES; global_batch=$((GPUS * LOCAL_BATCH * ACCUM))"
echo "[P2] GT L1=1, temporal=1, LPIPS=0.1; 3D hinge PatchGAN lambda=${GAN_WEIGHT:-0.01}, ramp100"
echo "[P2] P1 checkpoints and frozen ReAE/A1 are not modified."

ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  test -f "$RESUME/training_state.pt" || { echo "P2 resume checkpoint is incomplete" >&2; exit 1; }
  test -f "$RESUME/discriminator.safetensors" || { echo "Missing P2 discriminator state" >&2; exit 1; }
  ARGS+=(--resume "$RESUME")
fi

# P2 checkpoint step0 = P1-P5000 warm start. P2 step250/500/750/1000 are NEW updates.
torchrun --standalone --nproc_per_node="$GPUS" tools/train_m8_p2_video_gan.py \
  --base-checkpoint checkpoints_prompt_free_no_time \
  --student-init "$STUDENT_INIT" \
  --decoder-checkpoint outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder \
  --legacy-view-cache outputs/b2a/cache_stage_a200k_train_full \
  --val-view-cache outputs/b2a/cache_stage_a200k_val13 \
  --ultra-pairs "$HR_ROOT/shard_0/paired_views.jsonl" \
  --ultra-pairs "$HR_ROOT/shard_1/paired_views.jsonl" \
  --ultra-pairs "$HR_ROOT/shard_2/paired_views.jsonl" \
  --ultra-pairs "$HR_ROOT/shard_3/paired_views.jsonl" \
  --path-root . --output-dir "$RUN_DIR" \
  --lpips-weights "$LPIPS_WEIGHTS" --lpips-weight 0.1 \
  --pixel-weight 1 --temporal-weight 1 --router-weight 0.01 \
  --gan-weight "${GAN_WEIGHT:-0.01}" --gan-ramp-steps "${GAN_RAMP_STEPS:-100}" \
  --disc-learning-rate "${DISC_LR:-0.0001}" --disc-crop "${DISC_CROP:-128}" \
  --disc-width "${DISC_WIDTH:-32}" \
  --batch-size "$LOCAL_BATCH" --gradient-accumulation-steps "$ACCUM" \
  --num-workers "$WORKERS" --prefetch-factor 2 \
  --learning-rate "${G_LR:-0.000001}" --warmup-steps 50 \
  --max-steps "$MAX_STEPS" --schedule-steps 1000 \
  --validate-every 250 --save-every 250 --log-every 10 \
  --custom-ref-root "$CUSTOM_REF_ROOT" \
  "${ARGS[@]}" "${@}" 2>&1 | tee "$RUN_DIR.console.log"
