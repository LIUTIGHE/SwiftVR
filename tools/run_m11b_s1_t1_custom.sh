#!/usr/bin/env bash
# M11B-S1-T1: exact full-origin C24 E99 / 250 / 500 custom comparison.
# Usage: bash tools/run_m11b_s1_t1_custom.sh render|analyze
# Optionally: SCENES='Wetland1', VARIANTS='step250 step500'
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:?Usage: bash tools/run_m11b_s1_t1_custom.sh render|analyze}"
case "$MODE" in render|analyze) ;; *) echo "Invalid mode: $MODE" >&2; exit 2;; esac

BASE="${BASE:-checkpoints_prompt_free_no_time}"
CKPT="${CKPT:-outputs/b2b/m11b_stagea_ultra_bootstrap30k_v1/checkpoints/step_00030000}"
E99="${E99:-outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder}"
PILOT="${PILOT:-outputs/m11b/M11B-S1-T1/pilot500_v1}"
OUT_ROOT="${OUT_ROOT:-outputs/m11b/M11B-S1-T1/custom_pilot500_v1}"
INPUT_WETLAND="${INPUT_WETLAND:-../data/yuv_S_10bit/720p_mp4/S5_Wetland1_1920x1080p444_10bit.mp4}"
INPUT_MANGROVE="${INPUT_MANGROVE:-../data/yuv_S_10bit/720p_mp4/S5_Mangrove1_1920x1080p444_10bit.mp4}"
SCENES="${SCENES:-Wetland1 Mangrove1}"
VARIANTS="${VARIANTS:-e99 step250 step500}"
ROI="${ROI:-960,540,960,960}"

if [[ "$MODE" == render ]]; then
  test -f "$BASE/reae.safetensors" || { echo "Missing base ReAE" >&2; exit 1; }
  test -f "$CKPT/transformer/diffusion_pytorch_model.safetensors" || { echo "Missing M11-B Transformer" >&2; exit 1; }
fi
for scene in $SCENES; do
  case "$scene" in
    Wetland1) INPUT="$INPUT_WETLAND";;
    Mangrove1) INPUT="$INPUT_MANGROVE";;
    *) echo "Unknown scene: $scene" >&2; exit 2;;
  esac
  if [[ "$MODE" == render ]]; then
    test -f "$INPUT" || { echo "Missing full-origin input video: $INPUT" >&2; exit 1; }
    for variant in $VARIANTS; do
      case "$variant" in
        e99) DECODER="$E99";;
        step250) DECODER="$PILOT/checkpoints/step_00000250/tiny_decoder";;
        step500) DECODER="$PILOT/checkpoints/step_00000500/tiny_decoder";;
        *) echo "Unknown variant: $variant" >&2; exit 2;;
      esac
      test -f "$DECODER/model.safetensors" || { echo "Missing A1 checkpoint: $DECODER" >&2; exit 1; }
      DEST="$OUT_ROOT/$scene/$variant"
      test ! -e "$DEST" || { echo "Refusing existing output $DEST. Set VARIANTS to render only missing runs." >&2; exit 1; }
      mkdir -p "$DEST"
      echo "[M11B-S1-T1] Render scene=$scene decoder=$variant -> $DEST/png"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}" python scripts/inference_custom_components.py \
        --input "$INPUT" --output "$DEST/png" \
        --base-checkpoint "$BASE" --transformer-checkpoint "$CKPT" \
        --transformer-type moe --decoder-type m9a1 \
        --decoder-checkpoint "$DECODER" --upscale 3 \
        --clip-len 24 --dit-overlap 0 --dtype bfloat16 \
        --attention-backend sdpa --png 2>&1 | tee "$DEST/inference.log"
      python - "$INPUT" "$CKPT" "$DECODER" "$DEST" "$scene" "$variant" "$BASE" <<'PY'
import hashlib,json,sys
from pathlib import Path
src,dit,decoder,out,scene,variant,base=sys.argv[1:]
out=Path(out)
frames=sorted((out/"png").glob("*.png"))
if len(frames)<8:
    raise RuntimeError(f"Too few produced PNGs: {len(frames)} in {out}")
ids=[int(p.stem) for p in frames]
if ids!=list(range(len(ids))):
    raise RuntimeError("Rendered frame IDs are not contiguous from global origin zero")
def fingerprint(path):
    p=Path(path).resolve()
    digest=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            digest.update(chunk)
    return {"path":str(p),"sha256":digest.hexdigest()}
record={
    "kind":"M11B-S1-T1-custom-C24-v1","scene":scene,"variant":variant,
    "source_video":fingerprint(src),
    "transformer":fingerprint(Path(dit)/"transformer/diffusion_pytorch_model.safetensors"),
    "transformer_config":fingerprint(Path(dit)/"transformer/config.json"),
    "decoder":fingerprint(Path(decoder)/"model.safetensors"),
    "decoder_config":fingerprint(Path(decoder)/"config.json"),
    "reae":fingerprint(Path(base)/"reae.safetensors"),
    "inference_script_sha256":fingerprint("scripts/inference_custom_components.py")["sha256"],
    "inference":{"upscale":3,"clip_len":24,"dit_overlap":0,
                 "dtype":"bfloat16","attention_backend":"sdpa","png":True},
    "frame_count":len(frames),
    "note":"Complete full-origin PNGs. No implicit temporal retiming or montage crops."
}
(out/"provenance.json").write_text(json.dumps(record,indent=2)+"\n")
print(f"[M11B-S1-T1] provenance saved, frames={len(frames)}")
PY
    done
  else
    for variant in e99 step250 step500; do
      test -f "$OUT_ROOT/$scene/$variant/provenance.json" || {
        echo "Missing complete render metadata for $scene $variant" >&2; exit 1;
      }
    done
    python - "$OUT_ROOT" "$scene" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])/sys.argv[2]
runs={name:json.loads((root/name/"provenance.json").read_text())
      for name in ("e99","step250","step500")}
baseline=runs["e99"]
for name,record in runs.items():
    for key in ("source_video","transformer","transformer_config","reae",
                "inference_script_sha256","inference","frame_count"):
        if record[key]!=baseline[key]:
            raise ValueError(f"Source/M11B/C24 provenance mismatch: {name} differs on {key}")
print(f"[M11B-S1-T1] matched same-source, same-M11B, same-C24 frames={baseline['frame_count']} scene={sys.argv[2]}")
PY
    args=(--e99 "$OUT_ROOT/$scene/e99/png"
          --step250 "$OUT_ROOT/$scene/step250/png"
          --step500 "$OUT_ROOT/$scene/step500/png"
          --roi "$ROI" --snapshots "245,248"
          --output-dir "$OUT_ROOT/$scene/analysis")
    if [[ "$scene" == Wetland1 ]]; then args+=(--start 200 --end 248); fi
    python tools/m11b_s1_t1_custom_compare.py "${args[@]}"
  fi
done
