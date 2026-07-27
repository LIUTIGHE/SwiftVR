"""Frame-sequence tests for tools/audit_triplet_alignment.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "audit_triplet_alignment.py"
_SPEC = importlib.util.spec_from_file_location(
    "audit_triplet_alignment_frames", _TOOL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class AuditTripletAlignmentFramesTest(unittest.TestCase):
    @staticmethod
    def _write_image(path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)

    def _make_explicit_triplet(self, root: Path, frame_count: int = 9):
        paths = {name: [] for name in ("hr", "hq", "lr")}
        for index in range(frame_count):
            hq = np.zeros((8, 12, 3), dtype=np.uint8)
            hq[:, index % 12, :] = 255
            hr = np.repeat(np.repeat(hq, 3, axis=0), 3, axis=1)
            for name, frame in (("hr", hr), ("hq", hq), ("lr", hq)):
                path = root / name / f"vid0_comp{index % 10}_{index:06d}.png"
                self._write_image(path, frame)
                paths[name].append(str(path))
        return paths

    def test_resolve_explicit_component_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._make_explicit_triplet(Path(tmp), frame_count=3)
            record = {
                "sample_id": "vid0",
                "media_type": "frames",
                "frame_path_mode": "explicit",
                "frame_indices": [0, 1, 2],
                "frame_count": 3,
                "hr_frames": paths["hr"],
                "hq_frames": paths["hq"],
                "lr_frames": paths["lr"],
            }
            sources = _TOOL.resolve_media_sources(record)
            self.assertEqual(sources["hq"].frame_indices, (0, 1, 2))
            self.assertEqual(sources["hq"].frame_path_mode, "explicit")
            self.assertTrue(
                sources["hq"].frame_paths[1].endswith("vid0_comp1_000001.png")
            )

    def test_audit_explicit_component_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._make_explicit_triplet(Path(tmp), frame_count=9)
            record = {
                "sample_id": "vid0",
                "split": "train",
                "media_type": "frames",
                "frame_path_mode": "explicit",
                "frame_indices": list(range(9)),
                "frame_count": 9,
                "hr_frames": paths["hr"],
                "hq_frames": paths["hq"],
                "lr_frames": paths["lr"],
            }
            result = _TOOL.audit_record(
                record,
                expected_scale=3.0,
                scale_tolerance=0.01,
                fps_tolerance=1e-3,
                duration_tolerance=0.01,
                offset_radius=1,
                sample_frames=3,
                timestamp_scan_limit=0,
                metric_thumbnail_max_side=64,
            )
            self.assertEqual(result["status"], "pass")
            temporal = result["temporal_alignment"]
            self.assertEqual(temporal["best_hr_offset_relative_to_hq"], 0)
            self.assertEqual(temporal["best_hq_offset_relative_to_lr"], 0)
            self.assertEqual(
                temporal["sampled_frame_indices"], temporal["sampled_positions"]
            )
            self.assertEqual(result["probes"]["hr"]["media_type"], "frames")

    def test_pattern_manifest_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(3):
                frame = np.full((8, 12, 3), index, dtype=np.uint8)
                for name in ("hr", "hq", "lr"):
                    self._write_image(root / name / f"clip_{index:06d}.png", frame)
            record = {
                "sample_id": "clip",
                "media_type": "frames",
                "frame_path_mode": "pattern",
                "frame_start": 0,
                "frame_end": 2,
                "frame_count": 3,
                "hr": str(root / "hr" / "clip_{frame:06d}.png"),
                "hq": str(root / "hq" / "clip_{frame:06d}.png"),
                "lr": str(root / "lr" / "clip_{frame:06d}.png"),
            }
            sources = _TOOL.resolve_media_sources(record)
            self.assertEqual(sources["hr"].frame_path_mode, "pattern")
            self.assertEqual(sources["hr"].frame_indices, (0, 1, 2))

    def test_explicit_lengths_must_match_indices(self):
        record = {
            "sample_id": "bad",
            "media_type": "frames",
            "frame_indices": [0, 1],
            "frame_count": 2,
            "hr_frames": ["a", "b"],
            "hq_frames": ["c"],
            "lr_frames": ["d", "e"],
        }
        with self.assertRaisesRegex(ValueError, "hq_frames length"):
            _TOOL.resolve_media_sources(record)

    def test_read_manifest_accepts_explicit_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._make_explicit_triplet(root, frame_count=2)
            record = {
                "sample_id": "vid0",
                "split": "val",
                "media_type": "frames",
                "frame_indices": [0, 1],
                "frame_count": 2,
                "hr_frames": paths["hr"],
                "hq_frames": paths["hq"],
                "lr_frames": paths["lr"],
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            rows = _TOOL.read_manifest(manifest, split="val")
            self.assertEqual([row["sample_id"] for row in rows], ["vid0"])


if __name__ == "__main__":
    unittest.main()
