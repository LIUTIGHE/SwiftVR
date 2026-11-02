#!/usr/bin/env bash
# Compare already-trained T1B-A/B on full-origin Wetland1 and Mangrove1.
# One foreground command, cached Original SwiftVR/E99/TinyCNN; render only A/B.
# BANK=/actual/reference/root CUSTOM_VIDEO=1 bash tools/run_m11b_t1b_custom_ab.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
BANK="${BANK:-outputs/custom/p1_reference_v1}"
ROOT="${T1B_ROOT:-outputs/m11b/M11B-S1-T1B}"
STEP="${T1B_STEP:-250}"
EVAL="${EVAL_ROOT:-$ROOT/custom_pilot250}"
VIDEO="${CUSTOM_VIDEO:-0}"

test -f "$BANK/reference.json" || {
  echo "Missing fixed Original SwiftVR + TinyCNN reference.json: $BANK" >&2
  echo "Available banks (choose BANK=...):" >&2
  find outputs/custom -name reference.json -type f -print 2>/dev/null || true
  exit 1
}

printf -v STEP_PAD '%08d' "$STEP"
EXTRA=()
if [[ "$VIDEO" == 1 ]]; then EXTRA+=(--video); fi

for variant in a b; do
  candidate="$ROOT/pilot250_${variant}_v1/checkpoints/step_$STEP_PAD/tiny_decoder"
  test -f "$candidate/model.safetensors" || {
    echo "Missing T1B-$variant A1 checkpoint: $candidate" >&2; exit 1;
  }
  python tools/m11b_custom_val.py \
    --decoder-checkpoint "$candidate" \
    --reference-bank "$BANK" \
    --output-root "$EVAL/$variant" \
    "${EXTRA[@]}"
done

if [[ -e "$EVAL/ab_comparison" ]]; then
  test -f "$EVAL/ab_comparison/summary.json" || {
    echo "Partial A/B comparison output exists, inspect before retry: $EVAL/ab_comparison" >&2
    exit 1
  }
  echo "A/B report already exists, preserved: $EVAL/ab_comparison/summary.json"
else
  python tools/compare_m11b_t1b_ab.py \
    --root-a "$EVAL/a" --root-b "$EVAL/b" \
    --output "$EVAL/ab_comparison"
fi
echo "T1B A/B custom comparison: $EVAL/ab_comparison/summary.json"
