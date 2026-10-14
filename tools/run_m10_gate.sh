#!/usr/bin/env bash
# Run from the manually synchronized fork; no git/network operations.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${1:-spatial}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  spatial)  TEMPORAL_WEIGHT=0 ;;
  temporal) TEMPORAL_WEIGHT=1 ;;
  *) echo "Usage: bash tools/run_m10_gate.sh spatial|temporal [trainer arguments]" >&2; exit 2 ;;
esac

BASE="${BASE:-checkpoints_prompt_free_no_time}"
M8A_RUN="${M8A_RUN:-outputs/b2b/m8a_d1024_l20_gate200k}"
M8A="${M8A:-$M8A_RUN/checkpoints/step_00030000}"
A1_RUN="${A1_RUN:-outputs/b2b/m9a1_factorized_m8a30k}"
A1_DECODER="${A1_DECODER:-$A1_RUN/checkpoints/epoch_099_step_00024552/tiny_decoder}"
M3_RUN="${M3_RUN:-outputs/b2a/formal100k_full}"

read_config_path() {
  python - "$1" "$2" <<'PY'
import json
from pathlib import Path
import sys
path, key = Path(sys.argv[1]), sys.argv[2]
if not path.is_file():
    raise SystemExit(f"Missing {path}. Set STAGEA_TRAIN_CACHE / STAGEA_VAL_CACHE to the actual cache roots.")
value = json.loads(path.read_text(encoding="utf-8")).get(key)
if not isinstance(value, str) or not value:
    raise SystemExit(f"Missing path field {key!r} in {path}; supply the Stage-A cache path explicitly.")
print(value)
PY
}

# D1536's run consumed Stage-A training targets; M8-A's validation also used
# Stage-A. Do NOT take M8-A's training teacher_cache: that is the D1536 TA.
STAGEA_TRAIN_CACHE="${STAGEA_TRAIN_CACHE:-$(read_config_path "$M3_RUN/run_config.json" teacher_cache)}"
STAGEA_VAL_CACHE="${STAGEA_VAL_CACHE:-$(read_config_path "$M8A_RUN/run_config.json" val_teacher_cache)}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
MAX_STEPS="${MAX_STEPS:-500}"
OUT="${OUT:-outputs/b2b/m10_${MODE}_gate${MAX_STEPS}}"
printf 'M10 %s\nM8-A: %s\nFrozen A1: %s\nStage-A train cache: %s\nStage-A val cache: %s\nOutput: %s\n' \
  "$MODE" "$M8A" "$A1_DECODER" "$STAGEA_TRAIN_CACHE" "$STAGEA_VAL_CACHE" "$OUT"

exec env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  torchrun --standalone --nproc_per_node="${#GPUS[@]}" \
  tools/train_m10_backbone_recovery_ddp.py \
  --base-checkpoint "$BASE" \
  --student-init "$M8A" \
  --decoder-checkpoint "$A1_DECODER" \
  --teacher-cache "$STAGEA_TRAIN_CACHE" \
  --val-teacher-cache "$STAGEA_VAL_CACHE" \
  --output-dir "$OUT" \
  --batch-size "${MICRO_BS:-1}" \
  --gradient-accumulation-steps "${ACCUM:-4}" \
  --max-steps "$MAX_STEPS" \
  --learning-rate "${LR:-5e-6}" \
  --lr-warmup-steps "${WARMUP:-50}" \
  --hf-temporal-weight "$TEMPORAL_WEIGHT" \
  --validate-every 250 --save-every 250 --log-every 10 \
  --visual-samples 13 --visual-fps 30 \
  --dtype bfloat16 --attention-backend sdpa --seed 0 \
  "$@"
