"""Regression tests for synchronized non-contiguous frame sequences."""

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
    "build_triplet_manifest_noncontiguous", _TOOL_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class NonContiguousFrameManifestTest(unittest.TestCase):
    @staticmethod
    def _touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frame")

    def _make_roots(self, root: Path) -> dict[str, Path]:
        roots = {name: root / name for name in ("hr", "hq", "lr")}
        for path in roots.values():
            path.mkdir()
        return roots

    def test_strict_preserves_synchronized_non_contiguous_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for frame in (0, 2, 5):
                for dataset_root in roots.values():
                    self._touch(dataset_root / f"vid0_comp0_{frame:06d}.png")

            records, summary = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                media_mode="frames",
                split_all="train",
                strict=True,
            )

            record = records[0]
            self.assertFalse(record.frame_contiguous)
            self.assertEqual(record.frame_indices, (0, 2, 5))
            self.assertEqual(record.frame_count, 3)
            self.assertEqual(summary["non_contiguous_sequence_count"], 1)

            output = Path(tmp) / "manifest.jsonl"
            _TOOL.write_manifest(records, summary, output)
            payload = json.loads(output.read_text().strip())
            self.assertEqual(payload["frame_indices"], [0, 2, 5])
            self.assertFalse(payload["frame_contiguous"])

    def test_strict_still_rejects_mismatched_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for frame in (0, 2, 5):
                self._touch(roots["hr"] / f"vid0_comp0_{frame:06d}.png")
                self._touch(roots["hq"] / f"vid0_comp0_{frame:06d}.png")
            for frame in (0, 2):
                self._touch(roots["lr"] / f"vid0_comp0_{frame:06d}.png")

            with self.assertRaisesRegex(ValueError, "same_indices=False"):
                _TOOL.build_manifest(
                    hr_root=roots["hr"],
                    hq_root=roots["hq"],
                    lr_root=roots["lr"],
                    media_mode="frames",
                    split_all="train",
                    strict=True,
                )


if __name__ == "__main__":
    unittest.main()
