from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import select_m8_joint_safe_checkpoint as selector
from tools import train_m8_joint_recovery_ddp as recovery


def _default(parser, dest):
    for action in parser._actions:
        if action.dest == dest:
            return action.default
    raise AssertionError(dest)


def test_formal_m8c_profile_defaults_are_locked():
    parser = recovery.build_formal_parser()
    assert _default(parser, "batch_size") == 4
    assert _default(parser, "gradient_accumulation_steps") == 4
    assert _default(parser, "expected_global_batch_size") == 64
    assert _default(parser, "transformer_light_learning_rate") == 1e-6
    assert _default(parser, "transformer_tail_learning_rate") == 2e-6
    assert _default(parser, "decoder_learning_rate") == 2e-5
    assert _default(parser, "max_steps") == 5000
    assert _default(parser, "validate_every") == 250
    assert _default(parser, "save_every") == 500
    assert _default(parser, "validate_at_start") is True
    assert _default(parser, "visual_validation_samples") == 0
    assert _default(parser, "dtype") == "float16"


def test_visual_validation_compatibility_argument_parses():
    parser = recovery.build_formal_parser()
    required = {
        "base_checkpoint": "base",
        "transformer_init": "transformer",
        "decoder_init": "decoder",
        "teacher_cache": "train-cache",
        "manifest": "train.jsonl",
        "val_teacher_cache": "val-cache",
        "val_manifest": "val.jsonl",
        "output_dir": "out",
    }
    argv = []
    for dest, value in required.items():
        argv += [f"--{dest.replace('_', '-')}", value]
    argv += ["--visual-validation-samples", "13"]
    args = parser.parse_args(argv)
    assert args.visual_validation_samples == 13


def test_safe_selector_rejects_better_loss_when_velocity_drift_is_too_large(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints" / "step_00000500").mkdir(parents=True)
    (run / "checkpoints" / "step_00001000").mkdir(parents=True)
    records = [
        {
            "global_step": 0,
            "loss": 1.0,
            "velocity_relative_l2": 0.560,
            "velocity_cosine": 0.83,
            "student_teacher_psnr": 30.0,
            "student_teacher_ssim": 0.90,
            "student_gt_psnr": 24.0,
            "student_gt_ssim": 0.78,
        },
        {
            "global_step": 500,
            "loss": 0.5,
            "velocity_relative_l2": 0.563,
            "velocity_cosine": 0.84,
            "student_teacher_psnr": 31.0,
            "student_teacher_ssim": 0.92,
            "student_gt_psnr": 24.2,
            "student_gt_ssim": 0.79,
        },
        {
            "global_step": 1000,
            "loss": 0.4,
            "velocity_relative_l2": 0.570,
            "velocity_cosine": 0.85,
            "student_teacher_psnr": 32.0,
            "student_teacher_ssim": 0.93,
            "student_gt_psnr": 24.3,
            "student_gt_ssim": 0.80,
        },
    ]
    with (run / "val_log.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    result = selector.select_safe_checkpoint(
        run,
        max_drift=0.005,
        checkpoint_every=500,
    )
    assert result["status"] == "PASS"
    assert result["selected"]["global_step"] == 500
    assert len(result["safe_candidates"]) == 1
