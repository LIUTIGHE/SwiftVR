"""CPU tests: exact frame identities, canonical chunk accounting and phase metrics."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import m10_phase_metrics as metrics
from tools import diagnose_m10_phase_offsets as offsets


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


chunk = load_file("_m10_phase_chunk_contract", ROOT / "swiftvr/streaming/chunk.py")
losses = load_file("_m10_phase_existing_loss", ROOT / "swiftvr/training/m10_recovery.py")


class PhaseTests(unittest.TestCase):
    def test_phase_and_first_trim(self):
        self.assertEqual([metrics.decoder_phase(t) for t in range(9)], [3, 0, 1, 2, 3, 0, 1, 2, 3])
        with self.assertRaises(ValueError):
            metrics.decoder_phase(-1)

    def test_canonical_emission_counts(self):
        rows = offsets.output_frame_map(chunk.build_chunk_specs(81, 24), 0)
        self.assertEqual(len(rows), 81)
        self.assertEqual([sum(r["chunk"] == i for r in rows) for i in range(4)], [25, 24, 24, 8])
        self.assertFalse(rows[0]["interior"])
        self.assertFalse(rows[25]["interior"])
        self.assertTrue(rows[27]["interior"])
        self.assertFalse(rows[73]["interior"])
        self.assertEqual(len(offsets.output_frame_map(chunk.build_chunk_specs(13, 24), 0)), 13)

    def test_offset_aligns_physical_frame_not_local_index(self):
        a = offsets.output_frame_map(chunk.build_chunk_specs(81, 24), 100)
        b = offsets.output_frame_map(chunk.build_chunk_specs(81, 24), 101)
        pairs = offsets.align_runs(a, b)
        self.assertEqual(len(pairs), 80)
        x, y = pairs[0]
        self.assertEqual(x["source_id"], y["source_id"])
        self.assertEqual((x["local_frame"], y["local_frame"]), (1, 0))
        self.assertEqual((x["phase"], y["phase"]), (0, 3))
        self.assertEqual(sum(x["interior"] and y["interior"] for x, y in pairs), 38)
        self.assertTrue(all(x["phase"] == (y["phase"] + 1) % 4 for x, y in pairs))
        with self.assertRaises(ValueError):
            offsets.align_runs(a + a[:1], b)

    def test_high_pass_matches_existing_m10_filter(self):
        rng = np.random.default_rng(13)
        x = rng.random((17, 21, 3), dtype=np.float32)
        video = torch.from_numpy(x.transpose(2, 0, 1).copy())[None, None]
        expected = losses.GaussianHighPass()(video)[0, 0].permute(1, 2, 0).numpy()
        np.testing.assert_allclose(metrics.high_pass(x), expected, atol=4e-7, rtol=1e-5)
        np.testing.assert_allclose(metrics.high_pass(np.ones_like(x)), 0, atol=2e-7)

    def test_missing_phase_no_nan_or_infinite_json(self):
        rows = [{"phase": 3, "rgb_mae": 0., "rgb_mse": 0., "hf_mae": 0., "hf_temporal_mae": None}]
        report = metrics.aggregate(rows)
        self.assertEqual(report["by_phase"]["0"]["frames"], 0)
        self.assertIsNone(report["all"]["psnr_db"])
        self.assertTrue(report["all"]["rgb_exact"])
        json.dumps(report, allow_nan=False)

    def test_common_teacher_flicker_can_hide_in_teacher_matching(self):
        pattern = torch.from_numpy((np.indices((8, 8)).sum(axis=0) % 2).astype(np.float32))
        gt = pattern.expand(1, 13, 3, 8, 8).clone()
        bad = gt.clone()
        bad[:, 1::4] = 0.5
        methods = [("GT", gt), ("StageA+Orig", bad), ("M8A+Orig", bad), ("M8A+M9A1", bad)]
        rows = metrics.val_phase_rows(methods, sample_index=0, identity={"frame_indices": list(range(100, 113))})
        teacher_match = [r for r in rows if r["pair"] == "M8A+M9A1 vs StageA+Orig"]
        self.assertEqual(max(r["hf_mae"] for r in teacher_match), 0)
        gt_match = [r for r in rows if r["pair"] == "StageA+Orig vs GT"]
        grouped = metrics.aggregate(gt_match)
        self.assertGreater(grouped["by_phase"]["0"]["hf_mae"], .1)
        self.assertEqual(grouped["by_phase"]["3"]["hf_mae"], 0)
        self.assertEqual(gt_match[1]["source_frame_index"], 101)
        self.assertEqual(gt_match[1]["phase"], 0)  # local, NOT (101+3)%4
        with tempfile.TemporaryDirectory() as d:
            metrics.write_val_phase_report(Path(d), rows)
            report = json.loads((Path(d) / "phase_report.json").read_text())
            balanced = report["pairs"]["StageA+Orig vs GT"]["exclude_first_frame"]
            self.assertEqual([v["frames"] for v in balanced["by_phase"].values()], [3, 3, 3, 3])
            self.assertEqual(balanced["all"]["temporal_pairs"], 11)

    def test_report_does_not_cross_sample_temporal_boundaries(self):
        gt = torch.zeros(1, 13, 3, 8, 8)
        methods = [("GT", gt), ("StageA+Orig", gt), ("M8A+Orig", gt), ("M8A+M9A1", gt)]
        for sample in (0, 1):
            rows = metrics.val_phase_rows(methods, sample_index=sample, identity={"frame_indices": list(range(13))})
            self.assertTrue(all(r["hf_temporal_mae"] is None for r in rows if r["local_frame"] == 0))

    def test_png_compare_end_to_end_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "source_png").mkdir()
            runs = []
            for offset in (0, 1):
                name = f"output_offset{offset}"
                (root / name).mkdir()
                rows = offsets.output_frame_map(chunk.build_chunk_specs(81, 24), offset)
                for row in rows:
                    row["filename"] = f"{row['source_id']:08d}.png"
                    image = Image.new("RGB", (16, 16), (row["source_id"], 40, 80))
                    image.save(root / name / row["filename"])
                    image.save(root / "source_png" / row["filename"])
                runs.append({"output_dir": name, "frames": rows})
            (root / "run.json").write_text(json.dumps({"source": "synthetic", "source_fps": 30, "runs": runs}))
            with patch.object(offsets.shutil, "which", return_value=None):
                offsets.compare(root, crop=None, name="comparison", panel_width=64, fps=30)
            result = json.loads((root / "comparison/report.json").read_text())
            self.assertEqual(result["middle_interiors"]["all"]["frames"], 38)
            self.assertEqual(result["middle_interiors"]["all"]["rgb_mae"], 0)
            self.assertEqual(len(list((root / "comparison/frames").glob("*.png"))), 80)
            self.assertTrue((root / "comparison/contact_sheet.png").is_file())
            with self.assertRaises(FileExistsError):
                offsets.compare(root, crop=None, name="comparison", panel_width=64, fps=30)

    def test_bad_crop_rejected_not_silently_resized(self):
        with self.assertRaises(ValueError):
            offsets._crop_box("0,0,200,200", 16, 16)

    def test_existing_val13_api_kept_and_flags_opt_in(self):
        # Execute ONLY parser AST: the real model loader needs full GPU dependencies.
        source = ast.parse((ROOT / "tools/visualize_m9_val13_components.py").read_text())
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "build_parser")
        import argparse
        env = {"argparse": argparse, "Path": Path, "DTYPES": {"bfloat16": None}, "__doc__": "test"}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<parser>", "exec"), env)
        args = env["build_parser"]().parse_args(["--a1-run", "a", "--a1-checkpoint", "b", "--stagea-checkpoint", "c", "--output-dir", "d"])
        self.assertFalse(args.include_stagea_a1)
        self.assertFalse(args.phase_report)
        self.assertEqual(args.frame_indices, "0,6,12")


if __name__ == "__main__":
    unittest.main()
