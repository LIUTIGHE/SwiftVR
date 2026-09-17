#!/usr/bin/env bash
# Online fork -> manual Web/VSCode sync; no git, Hub downloads, or checkpoint edits.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-screen}"
if [[ $# -gt 0 ]]; then shift; fi
WORK="${WORK:-outputs/b2b/m10_layer_selection_v1}"
if [[ "$STAGE" == recover || "$STAGE" == validate ]]; then
  exec python tools/select_m10_layers.py "$STAGE" --work-dir "$WORK" "$@"
fi
if [[ "$STAGE" != screen ]]; then
  echo 'Usage: bash tools/run_m10_layer_selection.sh screen|recover|validate [arguments]' >&2
  exit 2
fi
BASE="${BASE:-checkpoints_prompt_free_no_time}"
A1_DECODER="${A1_DECODER:-outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder}"
DEPTH_INIT="${DEPTH_INIT:-outputs/b2b/m8a_d1024_l20_init_from_m5}"
M8A_RUN="${M8A_RUN:-outputs/b2b/m8a_d1024_l20_gate200k}"
M8A="${M8A:-$M8A_RUN/checkpoints/step_00030000}"
M3_RUN="${M3_RUN:-outputs/b2a/formal100k_full}"
config_path() {
  python - "$1" "$2" <<'PY'
import json
from pathlib import Path
import sys
path, key = Path(sys.argv[1]), sys.argv[2]
if not path.is_file():
    raise SystemExit(f'Missing {path}; supply STAGEA_TRAIN_CACHE/STAGEA_VAL_CACHE explicitly')
value = json.loads(path.read_text(encoding='utf-8')).get(key)
if not isinstance(value, str) or not value:
    raise SystemExit(f'Missing {key} in {path}; use explicit cache variables')
print(value)
PY
}
STAGEA_TRAIN_CACHE="${STAGEA_TRAIN_CACHE:-$(config_path "$M3_RUN/run_config.json" teacher_cache)}"
STAGEA_VAL_CACHE="${STAGEA_VAL_CACHE:-$(config_path "$M8A_RUN/run_config.json" val_teacher_cache)}"
exec python tools/select_m10_layers.py screen \
  --work-dir "$WORK" \
  --base-checkpoint "$BASE" --decoder-checkpoint "$A1_DECODER" \
  --depth-init "$DEPTH_INIT" --baseline-checkpoint "$M8A" \
  --teacher-cache "$STAGEA_TRAIN_CACHE" --val-teacher-cache "$STAGEA_VAL_CACHE" \
  "$@"
