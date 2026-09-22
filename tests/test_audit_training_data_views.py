"""CPU tests for tools/audit_training_data_views.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "audit_training_data_views.py"
_SPEC = importlib.util.spec_from_file_location("audit_training_data_views", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)

from swiftvr.data import TripletVideoDataset
from swiftvr.training.distillation import DeterministicTripletViewDataset


def _write_sequence(root: Path, count: int = 9) -> dict[str, object]:
    paths = {name: [] for name in ("hr", "hq", "lr")}
    for index in range(count):
        yy, xx = np.meshgrid(
            np.arange(8, dtype=np.uint8),
            np.arange(10, dtype=np.uint8),
            indexing="ij",
        )
        base = np.stack(
            [
                (xx * 17 + index * 5) % 255,
                (yy * 23 + index * 7) % 255,
                np.full_like(xx, index * 11 % 255),
            ],
            axis=-1,
        )
        hr = np.repeat(np.repeat(base, 3, axis=0), 3, axis=1)
        for name, array in (("hr", hr), ("hq", base), ("lr", base)):
            folder = root / name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"clip_{index:06d}.png"
            Image.fromarray(array).save(path)
            paths[name].append(str(path))
    return {
        "sample_id": "clip",
        "split": "train",
        "media_type": "frames",
        "frame_path_mode": "explicit",
        "frame_start": 0,
        "frame_end": count - 1,
        "frame_count": count,
        "frame_indices": list(range(count)),
        "hr_frames": paths["hr"],
        "hq_frames": paths["hq"],
        "lr_frames": paths["lr"],
    }


class TrainingDataAuditTest(unittest.TestCase):
    def test_metadata_only_view_spec_matches_dataset_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "vsr_triplets_plain_train.jsonl"
            manifest.write_text(json.dumps(_write_sequence(root)) + "\n", encoding="utf-8")
            base = TripletVideoDataset(
                manifest,
                split="train",
                training=True,
                clip_length=5,
                crop_size=(4, 6),
                scale=3,
                load_hq=False,
                horizontal_flip_probability=0.5,
                vertical_flip_probability=0.5,
            )
            full = DeterministicTripletViewDataset(base, views_per_record=3, view_seed=123)
            sizes = [(8, 10)]
            for index in range(len(full)):
                spec = _TOOL._view_spec(base, full, index, sizes)
                sample = full[index]
                self.assertEqual(spec["temporal_start"], sample["temporal_start"])
                self.assertEqual(spec["crop_box_lr"][0], sample["crop_top"])
                self.assertEqual(spec["crop_box_lr"][1], sample["crop_left"])
                self.assertEqual(spec["frame_indices"], sample["frame_indices"].tolist())
                self.assertEqual(spec["horizontal_flip"], sample["horizontal_flip"])
                self.assertEqual(spec["vertical_flip"], sample["vertical_flip"])

    def test_temporal_strip_sheet_writes_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "vsr_triplets_plain_train.jsonl"
            manifest.write_text(json.dumps(_write_sequence(root)) + "\n", encoding="utf-8")
            base = TripletVideoDataset(
                manifest,
                split="train",
                training=True,
                clip_length=5,
                crop_size=(4, 6),
                scale=3,
                load_hq=False,
                horizontal_flip_probability=0.0,
                vertical_flip_probability=0.0,
            )
            full = DeterministicTripletViewDataset(base, views_per_record=1, view_seed=123)
            metrics = {
                0: {
                    "record_uid": "plain:clip",
                    "view_index": 0,
                    "lr_temporal_l1": 0.1,
                }
            }
            output = root / "motion.jpg"
            _TOOL._temporal_strip_sheet(
                output,
                "motion",
                [0],
                full,
                metrics,
                frame_positions=(0, 2, 4),
                cell_size=32,
            )
            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)

    def test_profile_metrics_are_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "vsr_triplets_plain_train.jsonl"
            manifest.write_text(json.dumps(_write_sequence(root)) + "\n", encoding="utf-8")
            base = TripletVideoDataset(
                manifest,
                split="train",
                training=False,
                clip_length=5,
                crop_size=(4, 6),
                scale=3,
                load_hq=False,
            )
            metrics = _TOOL._profile_view(base[0])
            self.assertEqual(
                set(metrics),
                {
                    "hr_highpass_l1",
                    "lr_bicubic_highpass_l1",
                    "recoverable_highpass_gap",
                    "hr_vs_bicubic_lr_mae",
                    "lr_temporal_l1",
                },
            )
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            self.assertGreaterEqual(metrics["lr_temporal_l1"], 0.0)


if __name__ == "__main__":
    unittest.main()
