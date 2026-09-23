"""CPU tests for tools/index_ultravideo_sources.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "index_ultravideo_sources.py"
_SPEC = importlib.util.spec_from_file_location("index_ultravideo_sources", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class UltraVideoIndexTest(unittest.TestCase):
    def test_same_source_url_has_same_group_and_split(self):
        row_a = {"url": "https://example.test/source"}
        row_b = {"url": "https://example.test/source"}
        method_a, uid_a = _TOOL._source_group(row_a, "clip-a")
        method_b, uid_b = _TOOL._source_group(row_b, "clip-b")
        self.assertEqual(method_a, "source_url")
        self.assertEqual(method_b, "source_url")
        self.assertEqual(uid_a, uid_b)
        self.assertEqual(
            _TOOL._split_for_group(uid_a, seed=7, val_fraction=0.2),
            _TOOL._split_for_group(uid_b, seed=7, val_fraction=0.2),
        )

    def test_missing_url_falls_back_to_clip_id(self):
        method_a, uid_a = _TOOL._source_group({}, "clip-a")
        method_b, uid_b = _TOOL._source_group({}, "clip-b")
        self.assertEqual(method_a, "clip_id_fallback")
        self.assertNotEqual(uid_a, uid_b)

    def test_discover_sharded_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for shard in (1, 2):
                folder = root / f"clips_short_{shard}" / "clips_short"
                folder.mkdir(parents=True)
                (folder / f"clip-{shard}.mp4").write_bytes(b"")
            videos, shards = _TOOL._discover_videos(root)
            self.assertEqual(shards, [1, 2])
            self.assertEqual([path.stem for _, path in videos], ["clip-1", "clip-2"])

    def test_metadata_reader_joins_by_clip_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.csv"
            path.write_text(
                "clip_id,url,frame_width,frame_height,fps\n"
                "clip-a,https://example.test/a,3840,2160,24\n",
                encoding="utf-8",
            )
            rows, fields = _TOOL._read_metadata(path)
            self.assertIn("clip_id", fields)
            self.assertEqual(rows["clip-a"]["frame_width"], "3840")
            self.assertEqual(rows["clip-a"]["fps"], "24")


if __name__ == "__main__":
    unittest.main()
