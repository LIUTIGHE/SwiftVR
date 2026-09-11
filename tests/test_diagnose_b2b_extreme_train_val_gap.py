from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import diagnose_b2b_extreme_train_val_gap as diag


class B2BExtremeTrainValGapDiagnosticTest(unittest.TestCase):
    def test_uniform_positions_cover_cache_without_duplicates(self) -> None:
        positions = diag._uniform_positions(15896, 13)
        self.assertEqual(len(positions), 13)
        self.assertEqual(len(set(positions)), 13)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 15895)
        self.assertTrue(all(a < b for a, b in zip(positions, positions[1:])))

    def test_resolve_run_directory_best_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            checkpoint = run / "checkpoints" / "epoch_005_step_00001240"
            decoder = checkpoint / "tiny_decoder"
            decoder.mkdir(parents=True)
            (decoder / "config.json").write_text("{}", encoding="utf-8")
            (decoder / "model.safetensors").write_bytes(b"test")
            run.mkdir(parents=True, exist_ok=True)
            (run / "best.json").write_text(
                json.dumps({"checkpoint": "checkpoints/epoch_005_step_00001240"}),
                encoding="utf-8",
            )
            resolved, report = diag._resolve_student_root(run)
            self.assertEqual(resolved, decoder.resolve())
            self.assertEqual(report["resolved_from"], "best.json")


if __name__ == "__main__":
    unittest.main()
