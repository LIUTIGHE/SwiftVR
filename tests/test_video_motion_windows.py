"""CPU checks for the frame-index-based B1 moving-window selection."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREVIEW_PATH = ROOT / "tools" / "preview_video_motion_windows.py"
SPEC = importlib.util.spec_from_file_location("preview_video_motion_windows_test_target", PREVIEW_PATH)
assert SPEC is not None and SPEC.loader is not None
PREVIEW = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREVIEW
SPEC.loader.exec_module(PREVIEW)


class MotionWindowIndexTest(unittest.TestCase):
    def test_whole_video_spans_first_and_last_frame(self):
        indices = PREVIEW.select_indices(1000, 0, 1000, 24)
        self.assertEqual(len(indices), 24)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 999)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(indices, sorted(indices))

    def test_late_window_uses_absolute_source_indices(self):
        indices = PREVIEW.select_indices(1000, 200, 481, 10)
        self.assertEqual(indices[0], 200)
        self.assertEqual(indices[-1], 480)

    def test_invalid_windows_rejected(self):
        for args in ((0, 0, 0, 4), (10, 10, 10, 4), (10, 0, 11, 4), (10, 2, 8, 0)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    PREVIEW.select_indices(*args)

    def test_shell_script_syntax(self):
        subprocess.run(
            ["bash", "-n", str(ROOT / "tools" / "visual_check_chunk_B1.sh")],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
