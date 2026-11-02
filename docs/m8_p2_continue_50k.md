# P2-1000 → TOTAL P2-50000: exact optimizer continuation

**Why not** `MAX_STEPS=50000 RESUME=<P2-1000> bash tools/run_m8_p2_gan.sh`?
Old trainer was coded for `schedule_steps=1000`; it rejects `max_steps > schedule_steps`. Mutating the old trainer changes its hashed config, and changing schedule_steps fails old checkpoint strict checks. A direct 50K cosine extension would either silently change the LR on a completed run or remain at its 0.1x LR floor. Do not edit old run_config/training_state to trick validation.

**Implemented solution:** separate-run `tools/train_m8_p2_gan_continue.py` (derived from original P2 trainer), launched by `tools/run_m8_p2_gan_50k.sh`.

- Source: immutable original GAN `outputs/b2b/p2_video_gan_p1p5000_v1/checkpoints/step_00001000`.
- New run: `outputs/b2b/p2_video_gan_p1p5000_to50k_v1`. Original P2 output remains untouched.
- Load generator, 3D discriminator, **both full optimizer states**, per-rank RNG, epoch/cursor from P2-1000 checkpoint, not P1.
- Source run_config and source checkpoint state must exactly agree; verify immutable dataset/model/loss/architecture/seed/D/training batch/world config; allow only explicit new schedule, output dir, logging cadence, worker count and new trainer file. Shared source-file and weight/data hashes must still match.
- New run `step_00001000` is an audit/snapshot of original P2-1000. `step_00050000` represents **total** 50,000 P2 updates, i.e. **49,000 new updates**.
- 1K P2 cosine had reached 0.1× base generator LR. Continuation generator LR starts at ~0.1×base, linearly increases to 1.0×base over 200 new updates, then cosines down to 0.1×base at total step 50000. Discriminator retains its previous Adam moments and LR. This is a **declared new LR phase**, not a mathematically identical continuation of the old 1K cosine.
- Checkpoint each 1000 total steps; val13 and full Wetland1/Mangrove1 every 2500 steps. Initial step1000 val checks origin against prior run. Beware disk costs (50 model/optimizer snapshots and ~20 custom video validations).

### Start on the user's server

Sync the following **new** files; old trainer/launcher/P1/inference are untouched:

```text
tools/train_m8_p2_gan_continue.py
tools/run_m8_p2_gan_50k.sh
tests/test_m8_p2_gan_continue.py
docs/m8_p2_continue_50k.md
```

Verify Python and shell:
```bash
cd /data1/a/SwiftVR
python -m unittest discover -s tests -p 'test_m8_p2_gan_continue.py'
bash -n tools/run_m8_p2_gan_50k.sh
```

Run with actual prior batch/optimizer settings. If old P2 was `4 GPUs × local32 × accum1`, the defaults below match it. Launcher preflights old run_config *before* starting four GPUs:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
GPUS=4 LOCAL_BATCH=32 ACCUM=1 WORKERS=16 \
P2_PARENT=outputs/b2b/p2_video_gan_p1p5000_v1/checkpoints/step_00001000 \
RUN_DIR=outputs/b2b/p2_video_gan_p1p5000_to50k_v1 \
MAX_STEPS=50000 \
bash tools/run_m8_p2_gan_50k.sh
```

To **resume a later interruption of the NEW 50K continuation**, pass RESUME pointing to an existing checkpoint IN THE NEW RUN (not original 1K run):

```bash
RESUME=outputs/b2b/p2_video_gan_p1p5000_to50k_v1/checkpoints/step_00005000 \
bash tools/run_m8_p2_gan_50k.sh
```

Keep P2_PARENT (original 1K checkpoint), model/data/loss config, GPU count, global batch and source files unchanged for a true optimizer continuation. `--resume` is intentionally distinct from `--continue-from`.

If the old P2 actually used `LOCAL_BATCH=1 ACCUM=16` etc., set those values instead. The launcher refuses a mismatch to avoid silently making the run incomparable.

### Quality interpretation

User reported **no visible benefit after P2-1000**. That does not prove GAN never works, but cannot justify calling a 50K run effective. Review early checkpoints around step2500, 5000, 10000 before committing all available compute; detect D saturation/oscillation, false detail, temporal texture crawling, brightness flicker, GT val13 regression. No guarantee that 50K fixes blur; earlier P1 improved stability but not fine textures.

### Verification status

New continuation mode's source invariants, schedule boundaries, CLI/help and shell syntax have unit tests. These are not a substitute for the user's real server CUDA, DDP, and resumed-optimizer smoke; no long-run completion is claimed.
