#!/usr/bin/env bash
# Continue completed P2-1000 to TOTAL P2 step 50000.
# Preserves BOTH optimizers, generator/discriminator weights, sampler cursor
# and per-rank RNG. Writes to NEW run dir; original P2 is never modified.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
: "${GPUS:=4}"
: "${LOCAL_BATCH:=32}"  # Must match the actual P2-1000 run_config!
: "${ACCUM:=1}"        # Must match P2-1000.
: "${WORKERS:=16}"     # May change; more workers is not always faster.
: "${HR_ROOT:=outputs/data_audit/ultravideo_hr_train_v1}"
: "${CUSTOM_REF_ROOT:=outputs/custom/p1_reference_v1}"
: "${LPIPS_WEIGHTS:=outputs/local_weights/lpips_alex_complete.pt}"
: "${STUDENT_INIT:=outputs/b2b/p1_gt_perceptual_v1/checkpoints/step_00005000}"
: "${P2_PARENT:=outputs/b2b/p2_video_gan_p1p5000_v1/checkpoints/step_00001000}"
: "${RUN_DIR:=outputs/b2b/p2_video_gan_p1p5000_to50k_v1}"
: "${MAX_STEPS:=50000}"
: "${G_LR:=0.000001}"
: "${DISC_LR:=0.0001}"

for file in "$P2_PARENT/training_state.pt" "$P2_PARENT/discriminator.safetensors" \
            "$P2_PARENT/transformer/diffusion_pytorch_model.safetensors" \
            "$P2_PARENT/metadata.json" \
            "$P2_PARENT/../../run_config.json" \
            "$STUDENT_INIT/transformer/config.json" \
            "$LPIPS_WEIGHTS" "$CUSTOM_REF_ROOT/reference.json"; do
  test -f "$file" || { echo "Missing required source: $file" >&2; exit 1; }
done
for S in 0 1 2 3; do
  test -f "$HR_ROOT/shard_$S/paired_views.jsonl" || { echo "Missing HR shard $S" >&2; exit 1; }
done
if [[ -z "${RESUME:-}" ]]; then
  test ! -e "$RUN_DIR" || { echo "New continuation RUN_DIR already exists; set RESUME explicitly." >&2; exit 1; }
else
  test -f "$RESUME/training_state.pt" || { echo "Missing continuation checkpoint: $RESUME" >&2; exit 1; }
fi

# Source metadata and batch/optimizer-relevant settings must agree exactly.
# Do this *before* torchrun so a mismatch does not allocate 4 big models.
P2_PARENT="$P2_PARENT" STUDENT_INIT="$STUDENT_INIT" \
LPIPS_WEIGHTS="$LPIPS_WEIGHTS" CUSTOM_REF_ROOT="$CUSTOM_REF_ROOT" \
HR_ROOT="$HR_ROOT" GPUS="$GPUS" LOCAL_BATCH="$LOCAL_BATCH" ACCUM="$ACCUM" \
G_LR="$G_LR" DISC_LR="$DISC_LR" python - <<'PY'
import json, os
from pathlib import Path
source=Path(os.environ["P2_PARENT"]).resolve()
cfg=json.loads((source.parent.parent/"run_config.json").read_text())
meta=json.loads((source/"metadata.json").read_text())
expect={
    "student_init":str(Path(os.environ["STUDENT_INIT"]).resolve()),
    "lpips_weights":str(Path(os.environ["LPIPS_WEIGHTS"]).resolve()),
    "custom_ref_root":str(Path(os.environ["CUSTOM_REF_ROOT"]).resolve()),
    "batch_size":int(os.environ["LOCAL_BATCH"]),
    "gradient_accumulation_steps":int(os.environ["ACCUM"]),
    "world_size":int(os.environ["GPUS"]),
    "learning_rate":float(os.environ["G_LR"]),
    "disc_learning_rate":float(os.environ["DISC_LR"]),
    "ultra_pairs":[str(Path(os.environ["HR_ROOT"],f"shard_{s}","paired_views.jsonl").resolve()) for s in range(4)],
}
for key, val in expect.items():
    if cfg.get(key)!=val: raise SystemExit(f"Old P2 config mismatch {key}: {cfg.get(key)!r} != {val!r}")
if cfg.get("kind")!="m8_p2_video_patchgan_v1" or int(meta["global_step"])!=1000:
    raise SystemExit("P2_PARENT is not the expected completed P2-1000 checkpoint")
print("[safe-continue] Original P2 config, source and batch validated")
PY

mkdir -p "$(dirname "$RUN_DIR")"
echo "[P2-50K] old P2 source: $P2_PARENT"
echo "[P2-50K] new run: $RUN_DIR; total steps: $MAX_STEPS"
echo "[P2-50K] global batch: $((GPUS*LOCAL_BATCH*ACCUM))"
echo "[P2-50K] M8 LR: smooth 0.1x->1x ramp for first 200 NEW updates, then cosine to 50K."
echo "[P2-50K] D LR and both optimizer moments continue from P2-1000."
extra=()
if [[ -n "${RESUME:-}" ]]; then extra+=(--resume "$RESUME"); fi

torchrun --standalone --nproc_per_node="$GPUS" \
 tools/train_m8_p2_gan_continue.py \
 --continue-from "$P2_PARENT" \
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
 --gan-weight 0.01 --gan-ramp-steps 100 \
 --disc-learning-rate "$DISC_LR" --disc-crop 128 --disc-width 32 \
 --batch-size "$LOCAL_BATCH" --gradient-accumulation-steps "$ACCUM" \
 --num-workers "$WORKERS" --prefetch-factor 2 \
 --learning-rate "$G_LR" --warmup-steps 200 \
 --max-steps "$MAX_STEPS" --schedule-steps 50000 \
 --validate-every 2500 --save-every 1000 --log-every 10 \
 --custom-ref-root "$CUSTOM_REF_ROOT" \
 "${extra[@]}" "${@}" 2>&1 | tee "$RUN_DIR.console.log"
