from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from tools import train_b2b_dit_full_decoder_distill_ddp as wrapper


class B2BDiTFullDecoderDistillTest(unittest.TestCase):
    def test_d768_shape_defaults_are_locked(self) -> None:
        parser = wrapper.build_parser()
        args = parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--student-init", "student",
                "--teacher-cache", "cache",
                "--manifest", "train.jsonl",
                "--max-steps", "10",
                "--output-dir", "out",
            ]
        )
        self.assertEqual(args.student_hidden_dim, 768)
        self.assertEqual(args.student_num_heads, 6)
        self.assertEqual(args.student_head_dim, 128)
        self.assertEqual(args.student_ffn_dim, 4080)
        self.assertEqual(args.student_num_layers, 30)
        self.assertEqual(args.student_adapter_dim, 128)

    def test_base_training_remains_decoder_free_teacher_velocity_kd(self) -> None:
        source = inspect.getsource(wrapper.base.main)
        self.assertIn('"decoder_in_training_loss": False', source)
        self.assertIn("velocity_distillation_objective", source)
        self.assertIn("validation_decoder", source)

    def test_run_config_declares_teacher_only_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_config.json"
            wrapper._write_json(path, {"trainer": "test"})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["deployment_priority"], "teacher_behavior")
            self.assertEqual(payload["training_decoder"], "none")
            self.assertEqual(payload["validation_decoder"], "original_frozen_reae")
            self.assertEqual(payload["gt_role"], "diagnostic_only")
            self.assertEqual(payload["checkpoint_selection_metric"], "velocity_relative_l2")


if __name__ == "__main__":
    unittest.main()
