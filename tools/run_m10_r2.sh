#!/usr/bin/env bash
# Same immutable M8-A30k/A1 sources and Stage-A caches as M10; no server git.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MAX_STEPS="${MAX_STEPS:-1000}"
export LR="${LR:-1e-5}"
export OUT="${OUT:-outputs/b2b/m10_r2_perceptual_gate${MAX_STEPS}}"
EXTRA=()
if [[ -n "${ALEXNET_WEIGHTS:-}" ]]; then
  EXTRA+=(--alexnet-weights "$ALEXNET_WEIGHTS")
fi
exec bash tools/run_m10_gate.sh spatial \
  --velocity-nmse-weight 0.01 --velocity-cosine-weight 0 \
  --rgb-weight 1 --hf-weight 0 --hf-temporal-weight 0 \
  --lpips-weight 0.2 --lpips-microbatch-frames 4 \
  --probe-loss-gradients \
  --visual-frame-indices 0,1,2,3,4,5,6,7,8,9,10,11,12 \
  "${EXTRA[@]}" "$@"
