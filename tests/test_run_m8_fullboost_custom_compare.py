"""CPU tests for tools/run_m8_fullboost_custom_compare.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "run_m8_fullboost_custom_compare.py"
_SPEC = importlib.util.spec_from_file_location("run_m8_fullboost_custom_compare", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class RunM8FullBoostCustomCompareTest(unittest.TestCase):
    def test_parser_only_requires_input(self):
        parser = _TOOL.build_parser()
        args = parser.parse_args(["--input", "demo.mp4"])
        self.assertEqual(str(args.input), "demo.mp4")
        self.assertEqual(args.cuda_visible_devices, "7")
        self.assertEqual(args.frame_indices, "0,8,16,24,32")
        self.assertEqual(args.panel_width, 960)

    def test_default_candidates_match_current_fullboost_race(self):
        self.assertTrue(str(_TOOL.DEFAULTS["m8a"]).endswith("step_00030000"))
        self.assertTrue(str(_TOOL.DEFAULTS["ta1500"]).endswith("step_00001500"))
        self.assertTrue(str(_TOOL.DEFAULTS["ta4000"]).endswith("step_00004000"))
        self.assertTrue(str(_TOOL.DEFAULTS["stagea3500"]).endswith("step_00003500"))
        self.assertIn("epoch_099_step_00024552", str(_TOOL.DEFAULTS["decoder"]))

    def test_run_dry_run_does_not_execute(self):
        _TOOL._run(
            [sys.executable, "-c", "raise SystemExit(99)"],
            dry_run=True,
        )


if __name__ == "__main__":
    unittest.main()
