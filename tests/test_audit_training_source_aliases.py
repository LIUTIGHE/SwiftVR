"""CPU tests for tools/audit_training_source_aliases.py."""

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
_TOOL_PATH = _REPO_ROOT / "tools" / "audit_training_source_aliases.py"
_SPEC = importlib.util.spec_from_file_location("audit_training_source_aliases", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


def _write_variant(root: Path, variant: str, *, change: int = 0) -> Path:
    hr, hq, lr = [], [], []
    for index in range(5):
        base = np.full((8, 10, 3), 40 + index, dtype=np.uint8)
        high = np.repeat(np.repeat(base, 3, axis=0), 3, axis=1)
        if change and index == 2:
            high = high.copy()
            high[0, 0, 0] = np.uint8(min(255, int(high[0, 0, 0]) + change))
        for name, array, store in (("hr", high, hr), ("hq", base, hq), ("lr", base, lr)):
            folder = root / variant / name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"vid0_{index:06d}.png"
            Image.fromarray(array).save(path)
            store.append(str(path))
    row = {
        "sample_id": "vid0",
        "split": "train",
        "media_type": "frames",
        "frame_start": 0,
        "frame_end": 4,
        "frame_count": 5,
        "frame_indices": list(range(5)),
        "hr_frames": hr,
        "hq_frames": hq,
        "lr_frames": lr,
    }
    manifest = root / f"vsr_triplets_{variant}_train.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


class SourceAliasAuditTest(unittest.TestCase):
    def test_compare_frame_detects_exact_match_and_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = _write_variant(root, "plain")
            right = _write_variant(root, "text")
            left_row = json.loads(left.read_text(encoding="utf-8"))
            right_row = json.loads(right.read_text(encoding="utf-8"))
            exact = _TOOL._compare_frame(
                left_row["hr_frames"][2],
                right_row["hr_frames"][2],
                32,
            )
            self.assertTrue(exact["exact_rgb_equal"])
            self.assertEqual(exact["rgb_mae"], 0.0)

            changed = _write_variant(root, "text_changed", change=5)
            changed_row = json.loads(changed.read_text(encoding="utf-8"))
            different = _TOOL._compare_frame(
                left_row["hr_frames"][2],
                changed_row["hr_frames"][2],
                32,
            )
            self.assertFalse(different["exact_rgb_equal"])
            self.assertGreater(different["rgb_mae"], 0.0)

    def test_stable_order_is_reproducible(self):
        values = ["vid3", "vid1", "vid2"]
        self.assertEqual(_TOOL._stable_order(values, 7), _TOOL._stable_order(values, 7))


if __name__ == "__main__":
    unittest.main()
