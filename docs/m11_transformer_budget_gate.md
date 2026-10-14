# M11: 525–530 GFLOPs Transformer-only capacity gate

Status: **architecture definition and calibrated MAC audit implemented; not trained; no new full-model runtime count yet**.

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
| **Whole-system GFLOPs/frame** | **500.775** | **527.207** | **527.513** |
| 530G gate (estimate) | PASS | **PASS** | **PASS** |

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
The 530G gate still needs confirmation with the real runtime counter before
a training commitment. GPU latency and peak-memory are independent gates.

## Server: configuration and MAC audit

Synchronize the modified file
`tools/audit_b2b_moe_architecture.py` and the new
`tests/test_m11_architecture_budget.py` manually, without modifying running
training jobs or current checkpoints.

```bash
cd /data1/a/SwiftVR
python -m unittest discover -s tests -p 'test_m11_architecture_budget.py' -v

M8_CONFIG=outputs/b2b/m8a_d1024_l20_gate200k/checkpoints/step_00030000/transformer/config.json

python tools/audit_b2b_moe_architecture.py \
  --architecture m11a-d1024-l22 \
  --source-config "$M8_CONFIG" --meta-parameters \
  --export-config outputs/b2b/m11_budget_audit/m11a_config.json \
  --max-total-gflops 530 \
  --output-json outputs/b2b/m11_budget_audit/m11a_report.json

python tools/audit_b2b_moe_architecture.py \
  --architecture m11b-d1024-l20-e576 \
  --source-config "$M8_CONFIG" --meta-parameters \
  --export-config outputs/b2b/m11_budget_audit/m11b_config.json \
  --max-total-gflops 530 \
  --output-json outputs/b2b/m11_budget_audit/m11b_report.json
```

The generated JSON files are **architecture configs only**, not reloadable
checkpoints. Do not point `from_pretrained` at them until weights are
initialized and saved in a proper `transformer/` checkpoint.

The `--meta-parameters` option instantiates candidate modules on the
PyTorch meta device and records exact parameter counts without downloading
weights. It does **not** measure GPU speed or real inference MACs.
For an auditable full-model runtime count, materialize an initialized candidate
checkpoint and use `tools/runtime_macs.py::RuntimeMacCounter` with the canonical
streaming MIDDLE path and an A1 decoder; do not use the dense-only
`profile_stage_a_streaming_macs.py` loader for MoE without adapting it.

## Candidate initialization / next stage (not implemented here)

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
- No legacy training script or inference pipeline is modified in this audit
  commit; runtime counting and compatible initializers must be prepared before
  starting M11-A/B training.
