# M11: 525–530 GFLOPs Transformer-only capacity gate

Status: **both runtime MAC gates PASS on user's CUDA server; M11 initializers and TA trainer routing integrated but untested on real CUDA checkpoints**. No M11 training launched by these changes.

## Motivation / locked facts

D0 custom inspection: Stage-A+Original vs Stage-A+A1 loses only modest detail,
whereas M8-A+Original already loses substantial detail. The deployed decoder
remains **M9-A1 E99**; Original ReAE encoder, latent geometry and canonical
streaming protocol remain unchanged.

T1-A/B at 5k did not materially recover texture. Do not launch another LPIPS
weight sweep. The long custom sequences are the acceptance test, not val13:
- **Mangrove1 720p x3:** preserve the current best strengths (no visible regression).
- **Wetland1 720p x3:** recover missing detail, reduce phase0 weakness and
  motion-texture instability.
- Wetland1 local **221–253** is a fixed worst-phase review window. Labels are
  zero-based local output frames, phase=(k+3)%4, only for the same input reset
  origin. Do not treat absolute physical source frame numbers as phase IDs.
- Reuse the exact same custom starting frame, C24 protocol, ROI and reference
  bank for M8 baseline, new candidates, Original SwiftVR, and TinyCNN.
- Report native-resolution crops or lossless comparisons, not downscaled video
  or high-frequency energy alone. Existing val13 can remain historical but is
  **not** an optimization/selection/validation gate in M11.

## The two candidates

| | Baseline M8/A1 | M11-A Depth | M11-B FFN |
|---|---:|---:|---:|
| DiT hidden / heads | 1024 / 8 | 1024 / 8 | 1024 / 8 |
| DiT blocks | 20 | **22** | 20 |
| Shared expert width | 1024 | 1024 | 1024 |
| Routed experts / top-k | 12 / 2 | 12 / 2 | 12 / 2 |
| Routed expert width | 256 | 256 | **576** |
| Activated FFN width | 1536 | 1536 | **2176** |
| Total FFN expert width | 4096 | 4096 | **7936** |
| DiT GMAC/frame, calibrated | 132.1590 | **145.3749** | **145.5283** |
| Encoder GMAC/frame, frozen | 42.2780 | 42.2780 | 42.2780 |
| Decoder GMAC/frame, frozen | 75.9504 | 75.9504 | 75.9504 |
| **Whole-system GFLOPs/frame: estimate** | **500.775** | **527.207** | **527.513** |
| **Whole-system GFLOPs/frame: runtime DiT + fixed E/A1** | previously profiled | **527.314448128** | **527.647993600** |
| Runtime-counted DiT GMAC/frame | previously profiled | 145.428824064 | 145.595596800 |
| Total parameters | M8 checkpoint | **283,981,760** | **415,564,480** |
| Active parameters | M8 checkpoint | **168,356,800** | **179,314,880** |
| 530G runtime budget gate | baseline | **PASS** | **PASS** |

Canonical accounting: 1920x1088 internal output compute geometry, 24 RGB
frames per steady-state MIDDLE chunk, six latent frames, 2x2 DiT spatial
patching; 12,240 tokens per MIDDLE chunk = **510 tokens/output frame**.
1 MAC = 2 FLOPs. This refers to **1080p-class output**. For real
**720p→2160p (3x) custom validation**, cost is higher; do not label full 4K
inference as 527G/frame.

The base M8 132.159 GMAC is the previously validated count. The existing
`tools/audit_b2b_moe_architecture.py` computes active QKV/out linear, active
FFN linear and router MACs explicitly. Its remaining per-layer hidden work
comes from the calibrated 30-layer dense profile and is extrapolated in depth.
The M11 report anchors to M8 and adds those shape-dependent deltas; therefore
it is a **calibrated estimate, not an independent new runtime count**.
For M11-B the extra FFN linear cost is exactly 13.369344 GMAC/frame under
the stated canonical routing and token-count convention. M11-A adds two
architecturally identical blocks, and has some extrapolated non-FFN work.
The 530G MAC gate is now confirmed by the user's random-weight actual DiT
forward-hook counts with frozen E/A1 costs carried from their earlier profile.
Peak memory reported by this audit is **not training memory** (A 0.945GB,
B 1.217GB). Hook-instrumented forward times are **not production inference
latency** (A 0.817s, B 1.727s). GPU latency and training memory remain
independent future gates.

## Server: configuration and MAC audit

Synchronize the modified file
`tools/audit_b2b_moe_architecture.py` and the new
`tests/test_m11_architecture_budget.py` manually, without modifying running
training jobs or current checkpoints.

```bash
cd /data1/a/SwiftVR
python -m unittest discover -s tests -p 'test_m11_architecture_budget.py' -v

M8_CONFIG=outputs/b2b/m8a_d1024_l20_gate200k/checkpoints/step_00030000/transformer/config.json
# Select one IDLE GPU; no existing training job is touched.
export CUDA_VISIBLE_DEVICES=7

python tools/audit_b2b_moe_architecture.py \
  --architecture m11a-d1024-l22 \
  --source-config "$M8_CONFIG" --meta-parameters \
  --runtime-count --runtime-device cuda:0 --runtime-dtype bfloat16 \
  --export-config outputs/b2b/m11_budget_audit/m11a_config.json \
  --max-total-gflops 530 \
  --output-json outputs/b2b/m11_budget_audit/m11a_report.json

python tools/audit_b2b_moe_architecture.py \
  --architecture m11b-d1024-l20-e576 \
  --source-config "$M8_CONFIG" --meta-parameters \
  --runtime-count --runtime-device cuda:0 --runtime-dtype bfloat16 \
  --export-config outputs/b2b/m11_budget_audit/m11b_config.json \
  --max-total-gflops 530 \
  --output-json outputs/b2b/m11_budget_audit/m11b_report.json
```

The generated JSON files are **architecture configs only**, not reloadable
checkpoints. Do not point `from_pretrained` at them until weights are
initialized and saved in a proper `transformer/` checkpoint.

The `--meta-parameters` option instantiates candidate modules on the
PyTorch meta device and records exact parameter counts without downloading
weights. `--runtime-count` independently builds an **untrained** model from
that JSON (no model checkpoints or data files needed), executes its MoE forward
on latent shape **[1, in_channels, 6, 68, 120]** with SDPA and records
actual activated Linear/Conv/attention GEMM MAC via the existing
`RuntimeMacCounter`. It normalizes by **24 emitted RGB frames** and adds the
locked Encoder/A1 measured costs. Each report stores both the prior calibrated
estimate and the actual runtime-counted DiT MACs, with an independent hard
**530G** gate. Run on an idle GPU; it allocates the full random-weight model.

The MAC value is weight-independent for the existing top-2 no-dropping sparse
routing (every token executes one shared + two routed experts), but the
random-weight model **cannot be used for quality tests or inference**.
The instrumented elapsed time is also NOT production latency. It does
not include non-MAC index/gather/scatter, activation or elementwise costs; this
is the same defined MAC convention as previous stages.

For a full streaming end-to-end consistency check after initializing the
candidate checkpoint, reuse the canonical C24 MIDDLE pipeline, but note that
the existing `profile_stage_a_streaming_macs.py` loader is dense-only
and must not be pointed at a MoE checkpoint without adaptation.

## Candidate initialization / next stage (code integrated, CUDA smoke pending)

- **M11-A**: reuse trained D1024/L30 M5 blocks; prune only eight complete
  blocks to L22 with existing activation-based, edge-protected ordered mapping;
  do NOT insert two randomly initialized blocks into M8. Keep existing
  12 experts ×256, router and shared expert exactly.
- **M11-B**: 20 layers, same-width QKV/adapters as M8. Extra expert width
  cannot be loaded directly from an M8-E256 state. Use a deterministic
  D1536-TA dense-to-MoE transfer, with the established FFN-neuron ranking and
  round-robin expert partition. TA FFN width8960 is >= target 7936, so the
  target admits distinct teacher-neuron initialization; check the actual
  shape, transfer report and router stats.
- **Matched optimization**: same fixed-view data distribution, cached TA /
  Stage-A velocity and training budget for both candidates. First verify
  checkpoint initialization and a forward/backward smoke. Do not select models
  based solely on teacher velocity relL2 or val13.
- **Visual gates**: first at 5k, then at ~15k, final ~30k or justified plateau;
  compare Mangrove1 strength retention and Wetland1 improvement directly.
  Do not repeatedly rerun whole 4K video for every micro-checkpoint.
- Updated compatible entrypoints (existing files; no new model class):
  `tools/build_b2b_moe_depth_init.py` supports `--architecture m11a-d1024-l22`,
  `tools/build_b2b_moe_init.py` supports `--architecture m11b-d1024-l20-e576`,
  and `tools/train_b2b_moe_ta_distill_ddp_cache_safe.py` supports both
  `--architecture` values. Existing M5/M7/M8 modes and defaults remain.
- **Do not pass `--val-manifest` or `--val-teacher-cache`** for M11. The underlying
  trainer still writes checkpoints at `--save-every` and the final step.
  This deliberately disables val13 model selection and the legacy best metric.
  Checkpoint choices are made from Mangrove1 / Wetland1 custom inference,
  following the existing `tools/visual_check_m8_task.sh` workflow.
- Do not begin formal runs until CUDA smoke establishes loaded model/velocity
  tensors, shapes, full-cache matching, backwards and (for B) expanded-expert
  gradient coverage. There has been no M11 quality validation or pretrained
  full checkpoint yet.

### Server: inspect historical checkpoints before initialization

```bash
cd /data1/a/SwiftVR
python -m unittest discover -s tests -p 'test_m11_architecture_budget.py' -v

python - <<'PY'
from pathlib import Path
import json
base = Path("outputs/b2b")
reference = base/"m8a_d1024_l20_gate200k"/"run_config.json"
depth = base/"m8a_d1024_l20_init_from_m5"/"moe_depth_init_report.json"
for path in (reference, depth):
    if not path.is_file():
        print("MISSING:", path)
        continue
    obj = json.loads(path.read_text())
    print("\\nFILE:", path)
    for key in ("base_checkpoint", "manifests", "teacher_cache",
                "ultravideo_materialized_manifest", "ultravideo_teacher_cache",
                "source_checkpoint", "learning_rate", "runtime_dtype",
                "global_effective_batch_size"):
        if key in obj:
            print(f"  {key}: {obj[key]}")
PY
```

The `source_checkpoint` in the original M8 depth-initialization report is
the **trained M5 D1024/L30** source required for M11-A. For M11-B, recover the
**D1536 TA checkpoint** by inspecting the corresponding original MoE initialization
report, not by mistaking the 200k Stage-A D3072 for the TA.

**Initialization CLI:** both existing scripts use `--base-checkpoint`,
`--manifest` (repeatable), `--path-root`, `--output-dir` and their respective
`--source-checkpoint` (A) or `--teacher-checkpoint` (B) plus the new
`--architecture`. Use the exact historical manifest/view calibration
parameters and fresh output directories. Source weights are not overwritten.
The initialization scripts must produce `transformer/config.json`,
`transformer/diffusion_pytorch_model.safetensors` and their old audit reports.

**Training CLI:** invoke `tools/train_b2b_moe_ta_distill_ddp_cache_safe.py`
with new `--architecture`, the initialized `--student-init`, legacy/Ultra
TA teacher caches and the original M8-A training settings. Omit both val13
flags, and use the existing `--save-every` cadence to export deployable
Transformer snapshots. These are optimizer-independent research snapshots,
not proof of convergence. Compare fixed 720p×3 Mangrove1 / Wetland1 only.
