from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import build_b2b_ta_teacher_cache as cache_builder
from tools import train_b2b_dit_ta_distill_ddp as trainer


class B2BTeachingAssistantDistillationTest(unittest.TestCase):
    def test_ta_cache_builder_locks_d1536_teacher_shape(self) -> None:
        self.assertEqual(
            cache_builder.EXPECTED_TA_SHAPE,
            {
                "hidden_dim": 1536,
                "num_heads": 12,
                "head_dim": 128,
                "ffn_dim": 8960,
                "num_layers": 30,
                "adapter_dim": 128,
            },
        )
        self.assertEqual(cache_builder.TA_CACHE_KIND, trainer.TA_CACHE_KIND)

    def test_d768_student_shape_defaults_are_locked(self) -> None:
        parser = trainer.build_parser()
        args = parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--student-init", "student",
                "--teacher-cache", "ta-cache",
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

    def test_run_config_records_train_ta_and_stage_a_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_config.json"
            trainer._write_json(path, {"trainer": "test"})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["training_teacher"], "b2a_d1536_teaching_assistant")
            self.assertEqual(payload["training_teacher_cache_kind"], trainer.TA_CACHE_KIND)
            self.assertEqual(payload["validation_teacher"], "stage_a_d3072_reference")
            self.assertEqual(payload["validation_teacher_cache_kind"], trainer.STAGE_A_CACHE_KIND)
            self.assertEqual(payload["checkpoint_selection_metric"], "stage_a_velocity_relative_l2")
            self.assertEqual(payload["training_decoder"], "none")
            self.assertEqual(payload["gt_role"], "diagnostic_only")

    def test_role_aware_cache_accepts_only_ta_train_and_stage_a_val(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_root = root / "train"
            val_root = root / "val"
            train_root.mkdir()
            val_root.mkdir()
            for path, kind in (
                (train_root, trainer.TA_CACHE_KIND),
                (val_root, trainer.STAGE_A_CACHE_KIND),
            ):
                (path / "metadata.json").write_text(
                    json.dumps(
                        {
                            "format_version": 1,
                            "kind": kind,
                            "sample_count": 0,
                            "samples": [],
                        }
                    ),
                    encoding="utf-8",
                )

            old_train = trainer._train_cache_root
            old_val = trainer._val_cache_root
            try:
                trainer._train_cache_root = train_root.resolve()
                trainer._val_cache_root = val_root.resolve()
                train_cache = trainer._RoleAwareTeacherVelocityCache(train_root)
                val_cache = trainer._RoleAwareTeacherVelocityCache(val_root)
                self.assertEqual(train_cache.metadata["actual_kind"], trainer.TA_CACHE_KIND)
                self.assertEqual(train_cache.metadata["kind"], trainer.STAGE_A_CACHE_KIND)
                self.assertEqual(val_cache.metadata["kind"], trainer.STAGE_A_CACHE_KIND)
            finally:
                trainer._train_cache_root = old_train
                trainer._val_cache_root = old_val


if __name__ == "__main__":
    unittest.main()
