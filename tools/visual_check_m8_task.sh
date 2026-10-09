#!/usr/bin/env bash
# Explicit inference + composition, no hidden model-selection wrapper.
# prepare: run Original once. validate CHECKPOINT OUT: run only current student.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:?Use prepare or validate CHECKPOINT OUTPUT}"
GPU="${GPU:-7}"
export CUSTOM_REF_ROOT="${CUSTOM_REF_ROOT:?Set a new custom reference bank directory}"
export INPUT1="${INPUT1:-../data/yuv_S_10bit/720p_mp4/S5_Wetland1_1920x1080p444_10bit.mp4}"
export BASIC1="${BASIC1:-../data/yuv_S_10bit/AISR_pred/S5_Wetland1_1920x1080p444_10bit_SR_YUV444_RC1.0.mp4}"
export INPUT2="${INPUT2:-../data/yuv_S_10bit/720p_mp4/S5_Mangrove1_1920x1080p444_10bit.mp4}"
export BASIC2="${BASIC2:-../data/yuv_S_10bit/AISR_pred/S5_Mangrove1_1920x1080p444_10bit_SR_YUV444_RC1.0.mp4}"
export ORIGINAL="${ORIGINAL:-checkpoints}"
export BASE="${BASE:-checkpoints_prompt_free_no_time}"
export A1="${A1:-outputs/b2b/m9a1_factorized_m8a30k/checkpoints/epoch_099_step_00024552/tiny_decoder}"
# Native SR coordinates, SAME region for all four sources. No spatial downsample.
export ROI1="${ROI1:-960,540,960,960}"
export ROI2="${ROI2:-960,540,960,960}"
export KEEP_CUSTOM_PNG="${KEEP_CUSTOM_PNG:-0}"

if [[ "$MODE" == prepare ]]; then
  test ! -e "$CUSTOM_REF_ROOT" || { echo "Reference bank exists: use a new root" >&2; exit 1; }
  python - <<'PY'
import os
from pathlib import Path
for key in ('INPUT1','BASIC1','INPUT2','BASIC2'):
    p=Path(os.environ[key])
    if not p.is_file(): raise FileNotFoundError(p)
PY
  mkdir -p "$CUSTOM_REF_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/inference.py \
    --input "$INPUT1" --output "$CUSTOM_REF_ROOT/Wetland1/png" \
    --checkpoint "$ORIGINAL" --upscale 3 --clip-len 24 --dit-overlap 0 \
    --dtype bfloat16 --attention_backend sdpa --png
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/inference.py \
    --input "$INPUT2" --output "$CUSTOM_REF_ROOT/Mangrove1/png" \
    --checkpoint "$ORIGINAL" --upscale 3 --clip-len 24 --dit-overlap 0 \
    --dtype bfloat16 --attention_backend sdpa --png
  python - <<'PY'
import hashlib,json,os
from pathlib import Path
from tools.compare_720p3x_outputs import FrameSource
root=Path(os.environ['CUSTOM_REF_ROOT']).resolve()
def signature(path):
    p=Path(path).resolve(); h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    s=p.stat()
    return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h.hexdigest())
clips=[]
for i,name in ((1,'Wetland1'),(2,'Mangrove1')):
    lq=FrameSource(Path(os.environ[f'INPUT{i}']))
    original=FrameSource(root/name/'png')
    tiny=FrameSource(Path(os.environ[f'BASIC{i}']))
    n=4*((len(lq)-1)//4)+1
    if len(original)!=n or len(tiny)<n: raise ValueError('Incomplete Original/TinyCNN sequence')
    if (original.width,original.height)!=(lq.width*3,lq.height*3): raise ValueError('Original is not exact 3x')
    if (tiny.width,tiny.height)!=(original.width,original.height): raise ValueError('TinyCNN geometry mismatch')
    clips.append(dict(name=name,input=signature(os.environ[f'INPUT{i}']),basic=signature(os.environ[f'BASIC{i}']),
                      original=str(root/name/'png'),frames=n,roi=[int(v) for v in os.environ[f'ROI{i}'].split(',')]))
checkpoint=Path(os.environ['ORIGINAL'])
files=[checkpoint/'reae.safetensors',checkpoint/'prompt_embedding.safetensors',checkpoint/'transformer/config.json',*sorted((checkpoint/'transformer').glob('*.safetensors'))]
meta=dict(kind='m8_task_custom_reference_v1',clips=clips,original_files=[signature(p) for p in files],
          inference=dict(upscale=3,clip_len=24,dit_overlap=0,dtype='bfloat16',attention_backend='sdpa'),
          alignment='TinyCNN normalized ordinal-length mapping to Current; not FPS/PTS alignment')
(root/'reference.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Prepared',root/'reference.json')
PY
  exit 0
fi
[[ "$MODE" == validate ]] || { echo "Unknown mode: $MODE" >&2; exit 2; }
export CURRENT="${2:?Current checkpoint required}"
export OUT="${3:?New validation output root required}"
test ! -e "$OUT" || { echo "Use new validation output, do not reuse stale PNGs: $OUT" >&2; exit 1; }
# Read the immutable bank instead of trusting changing INPUT/BASIC environment variables.
python - <<'PY'
import json,os
from pathlib import Path
m=json.loads((Path(os.environ['CUSTOM_REF_ROOT'])/'reference.json').read_text())
if m['kind']!='m8_task_custom_reference_v1': raise ValueError('Wrong reference bank')
for s in m['original_files']+[c[k] for c in m['clips'] for k in ('input','basic')]:
    p=Path(s['path']); st=p.stat()
    if (st.st_size,st.st_mtime_ns)!=(s['size'],s['mtime_ns']): raise ValueError(f'Reference changed: {p}')
PY
mkdir -p "$OUT"
exec > >(tee "$OUT/visual_check.log") 2>&1
for NAME in Wetland1 Mangrove1; do
  INPUT=$(python - "$NAME" <<'PY'
import json,os,sys
from pathlib import Path
m=json.loads((Path(os.environ['CUSTOM_REF_ROOT'])/'reference.json').read_text())
print(next(c['input']['path'] for c in m['clips'] if c['name']==sys.argv[1]))
PY
)
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/inference_custom_components.py \
    --input "$INPUT" --output "$OUT/$NAME/current_png" \
    --base-checkpoint "$BASE" --transformer-checkpoint "$CURRENT" \
    --transformer-type moe --decoder-type m9a1 --decoder-checkpoint "$A1" \
    --upscale 3 --clip-len 24 --dit-overlap 0 --dtype bfloat16 \
    --attention-backend sdpa --png
  # Reuse canonical FrameSource and length alignment. Only compose native ROI here.
  python - "$NAME" <<'PY'
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from tools.compare_720p3x_outputs import FrameSource,_resize_rgb
from tools.compose_4way_video import _auto_basic_index
name=sys.argv[1]; bank=Path(os.environ['CUSTOM_REF_ROOT'])
meta=json.loads((bank/'reference.json').read_text())
c=next(c for c in meta['clips'] if c['name']==name)
out=Path(os.environ['OUT'])/name
lq=FrameSource(Path(c['input']['path'])); orig=FrameSource(Path(c['original']))
tiny=FrameSource(Path(c['basic']['path'])); current=FrameSource(out/'current_png')
n=len(current); w,h=current.width,current.height
if n!=c['frames'] or len(orig)!=n or not 0<=len(lq)-n<=3 or len(tiny)<n: raise ValueError('Frame-count contract failed')
if (w,h)!=(orig.width,orig.height) or (w,h)!=(tiny.width,tiny.height): raise ValueError('Do not resize unequal SR outputs')
x,y,cw,ch=c['roi']
if min(x,y)<0 or min(cw,ch)<=0 or x+cw>w or y+ch>h: raise ValueError('ROI outside SR output')
indices=[_auto_basic_index(k,basic_count=len(tiny),reference_count=n) for k in range(n)]
alignment=dict(reference='current inference ordinal',current_frames=n,lq_frames=len(lq),tiny_frames=len(tiny),tiny_indices=indices,
               note='Length-normalized sampling, not strict PTS; LQ/Original preserve frame prefix. Video is RGB8/FFV1, native ROI.')
(out/'alignment.json').write_text(json.dumps(alignment,indent=2)+'\n')
width,height=2*cw,2*(ch+28)
cmd=['ffmpeg','-nostdin','-n','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(lq.fps),'-i','-',
     '-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0',str(out/'native_fourway.mkv')]
labels=['LQ 3x','Original SwiftVR','Current','TinyCNN']
with (out/'ffmpeg.log').open('w') as log:
    writer=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
    try:
        for k in range(n):
            frames=[_resize_rgb(lq.frame(k),w,h),orig.frame(k),current.frame(k),tiny.frame(indices[k])]
            canvas=Image.new('RGB',(width,height),'black'); draw=ImageDraw.Draw(canvas)
            for j,arr in enumerate(frames):
                xx=(j%2)*cw; yy=(j//2)*(ch+28)
                canvas.paste(Image.fromarray(arr[y:y+ch,x:x+cw]),(xx,yy+28))
                draw.text((xx+5,yy+7),f'{labels[j]} frame={k} phase={(k+3)%4}',fill='white')
            writer.stdin.write(np.asarray(canvas,np.uint8).tobytes())
            if k<13 or n//2<=k<n//2+13 or k>=n-4:
                canvas.save(out/f'native_frame_{k:05d}.png')
    finally:
        writer.stdin.close()
        rc=writer.wait()
    if rc: raise RuntimeError(f'FFmpeg failed: {out}/ffmpeg.log')
(out/'complete.json').write_text(json.dumps(dict(frames=n,current=os.environ['CURRENT'],roi=c['roi'],lossless=True))+'\n')
print(name,'native four-way complete, frames=',n)
# Only our just-produced temporary full PNGs, AFTER successful lossless composition.
if os.environ['KEEP_CUSTOM_PNG']=='0':
    for p in (out/'current_png').glob('*.png'): p.unlink()
    (out/'current_png').rmdir()
PY
done
