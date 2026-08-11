from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "audit_stage_a_distillation.py"


class StageAAuditTest(unittest.TestCase):
    def test_script_parses_as_python(self):
        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        for expected in (
            "parse_model_specs",
            "evaluate_model",
            "profile_model",
            "export_visuals",
            "write_markdown",
            "main",
        ):
            self.assertIn(expected, function_names)

    def test_cli_help_smoke(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--base-checkpoint", completed.stdout)
        self.assertIn("--teacher-cache", completed.stdout)
        self.assertIn("--model", completed.stdout)
        self.assertIn("--profile-flops", completed.stdout)
        self.assertIn("--lpips", completed.stdout)

    def test_audit_contract_strings_are_present(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for expected in (
            "stage_a_audit.json",
            "stage_a_audit.md",
            "velocity_relative_l2",
            "student_teacher_psnr",
            "student_gt_psnr",
            "encoder_latency",
            "transformer_latency",
            "decoder_latency",
            "end_to_end_latency",
            "comparison.mp4",
            "differences.mp4",
        ):
            self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
