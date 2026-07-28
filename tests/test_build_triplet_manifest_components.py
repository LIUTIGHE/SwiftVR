"""Regression tests for interleaved ``_compN`` frame grouping."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "build_triplet_manifest.py"
_SPEC = importlib.util.spec_from_file_location(
    "build_triplet_manifest_components", _TOOL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class ComponentFrameManifestTest(unittest.TestCase):
    @staticmethod
    def _touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frame")

    def _make_roots(self, root: Path) -> dict[str, Path]:
        roots = {name: root / name for name in ("hr", "hq", "lr")}
        for path in roots.values():
            path.mkdir()
        return roots

    def test_group_components_merges_interleaved_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for frame in range(20):
                component = frame % 10
                for dataset_root in roots.values():
                    self._touch(
                        dataset_root
                        / f"vid0_comp{component}_{frame:06d}.png"
                    )

            records, summary = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                media_mode="frames",
                group_components=True,
                split_all="train",
                strict=True,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.sample_id, "vid0")
            self.assertEqual(record.frame_count, 20)
            self.assertTrue(record.frame_contiguous)
            self.assertEqual(record.frame_path_mode, "explicit")
            self.assertIsNone(record.hr)
            self.assertEqual(record.frame_indices, tuple(range(20)))
            self.assertTrue(record.hr_frames[0].endswith("vid0_comp0_000000.png"))
            self.assertTrue(record.hr_frames[11].endswith("vid0_comp1_000011.png"))
            self.assertEqual(summary["indexed_counts"], {"hr": 1, "hq": 1, "lr": 1})
            self.assertEqual(summary["non_contiguous_sequence_count"], 0)
            self.assertEqual(summary["frame_path_mode_counts"]["explicit"], 1)

    def test_group_components_preserves_text_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for frame in range(10):
                component = frame % 10
                for dataset_root in roots.values():
                    self._touch(
                        dataset_root
                        / f"vid2_comp{component}_{frame:06d}_text.png"
                    )

            text_regex = r"^(?P<clip>.+)_(?P<frame>\d{6})_text$"
            records, _ = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                media_mode="frames",
                frame_regex=text_regex,
                group_components=True,
                split_all="val",
                strict=True,
            )

            record = records[0]
            self.assertEqual(record.sample_id, "vid2")
            self.assertEqual(record.split, "val")
            self.assertTrue(record.lr_frames[-1].endswith("vid2_comp9_000009_text.png"))

    def test_explicit_frame_lists_are_written_to_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = self._make_roots(root)
            for frame in range(3):
                for dataset_root in roots.values():
                    self._touch(
                        dataset_root / f"vid3_comp{frame}_{frame:06d}.png"
                    )

            records, summary = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                media_mode="frames",
                group_components=True,
                split_all="train",
                strict=True,
            )
            output = root / "manifest.jsonl"
            _TOOL.write_manifest(records, summary, output)
            payload = json.loads(output.read_text().strip())

            self.assertNotIn("hr", payload)
            self.assertEqual(payload["frame_path_mode"], "explicit")
            self.assertEqual(payload["frame_indices"], [0, 1, 2])
            self.assertEqual(len(payload["hr_frames"]), 3)
            self.assertEqual(len(payload["hq_frames"]), 3)
            self.assertEqual(len(payload["lr_frames"]), 3)


if __name__ == "__main__":
    unittest.main()
