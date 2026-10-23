#!/usr/bin/env bash
# M11B-S1-T1B: use exactly this entry for matched A/B experiment.
# bash tools/run_m11b_s1_t1b.sh smoke A|B
# bash tools/run_m11b_s1_t1b.sh pilot A|B
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:?Usage: bash tools/run_m11b_s1_t1b.sh smoke|pilot A|B}"
VARIANT="${2:?Usage: bash tools/run_m11b_s1_t1b.sh smoke|pilot A|B}"
case "$MODE" in smoke|pilot) ;; *) echo "Unknown mode: $MODE" >&2; exit 2;; esac
case "$VARIANT" in A|B) ;; *) echo "Variant must be A or B" >&2; exit 2;; esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
BASE="${BASE:-checkpoints_prompt_free_no_time}"
CKPT="${CKPT:-outputs/b2b/m11b_stagea_ultra_bootstrap30k_v1/checkpoints/step_00030000}"
A1_INIT="${A1_INIT:-outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder}"
D1="${D1:-outputs/m11b/M11B-S1-D1}"
LPIPS_WEIGHTS="${LPIPS_WEIGHTS:-outputs/local_weights/lpips_alex_complete.pt}"
ROOT="${OUT_ROOT:-outputs/m11b/M11B-S1-T1B}"
COMMON=(
  --variant "$VARIANT"
  --base-checkpoint "$BASE"
  --m11b-checkpoint "$CKPT"
  --a1-init "$A1_INIT"
  --legacy-train "$D1/legacy_train_full"
  --ultra-train "$D1/ultra_train_full"
  --legacy-val "$D1/legacy_val13"
  --path-root .
  --batch-size 1
  --dtype bfloat16
  --teacher-weight 1.0
  --lowfreq-gt-weight 1.0
  --temporal-weight 0.5
  --highpass-kernel 7
  --gt-lpips-weight 0
  --lpips-weights "$LPIPS_WEIGHTS"
)
if [[ "$MODE" == smoke ]]; then
  OUT="$ROOT/smoke2_${VARIANT,,}_v1"
  OPT=(--smoke-steps 2 --gradient-accumulation-steps 1 --num-workers 0
       --save-every 2 --log-every 1)
else
  OUT="$ROOT/pilot250_${VARIANT,,}_v1"
  OPT=(--max-steps 250 --schedule-steps 6000 --warmup-steps 100
       --learning-rate 1e-5 --gradient-accumulation-steps 8
       --num-workers 2 --save-every 125 --validate-every 125
       --log-every 10)
  SMOKE="$ROOT/smoke2_${VARIANT,,}_v1/summary.json"
  python - "$SMOKE" "$VARIANT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);v=sys.argv[2]
if not p.is_file():
    raise FileNotFoundError(f"T1B-{v} smoke result missing: {p}")
meta=json.loads(p.read_text())
if (meta.get("status"),meta.get("phase"),meta.get("variant"),meta.get("step")) != ("smoke_pass","M11B-S1-T1B",v,2):
    raise ValueError(f"T1B-{v} smoke not yet passed: {meta}")
print(f"T1B-{v} smoke gate PASS")
PY
fi

for p in "$CKPT/transformer/diffusion_pytorch_model.safetensors" \
         "$BASE/reae.safetensors" "$A1_INIT/model.safetensors" \
         "$LPIPS_WEIGHTS" \
         "$D1/legacy_train_full/metadata.json" \
         "$D1/ultra_train_full/metadata.json" \
         "$D1/legacy_val13/metadata.json"; do
  test -f "$p" || { echo "Missing immutable source file: $p" >&2; exit 1; }
done
test ! -e "$OUT" || {
  echo "Refusing overwrite of $OUT; this launcher uses fresh run directories." >&2
  exit 1
}

echo "[M11B-S1-T1B] mode=$MODE variant=$VARIANT output=$OUT"
python tools/train_m11b_s1_t1b_decoder.py "${COMMON[@]}" \
  --output-dir "$OUT" "${OPT[@]}"
