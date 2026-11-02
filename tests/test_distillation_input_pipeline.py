"""CPU tests for the teacher-distillation input pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from swiftvr.data import TripletVideoDataset
from swiftvr.training.distillation import DeterministicTripletViewDataset
from swiftvr.training.input_pipeline import dataloader_worker_kwargs


def _write_sequence(root: Path, *, frames: int = 13) -> Path:
    paths = {"hr": [], "hq": [], "lr": []}
    for index in range(frames):
        base = np.full((8, 10, 3), index, dtype=np.uint8)
        hr = np.repeat(np.repeat(base, 3, axis=0), 3, axis=1)
        for name, array in (("hr", hr), ("hq", base), ("lr", base)):
            folder = root / name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{index:06d}.png"
            Image.fromarray(array).save(path)
            paths[name].append(str(path))
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "split": "train",
                "media_type": "frames",
                "frame_indices": list(range(frames)),
                "frame_count": frames,
                "hr_frames": paths["hr"],
                "hq_frames": paths["hq"],
                "lr_frames": paths["lr"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


class DistillationInputPipelineTest(unittest.TestCase):
    def test_load_hq_false_does_not_open_hq_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_sequence(root)
            for path in (root / "hq").glob("*.png"):
                path.unlink()

            dataset = TripletVideoDataset(
                manifest,
                split="train",
                training=False,
                clip_length=13,
                crop_size=(4, 6),
                scale=3,
                load_hq=False,
                horizontal_flip_probability=0.0,
                vertical_flip_probability=0.0,
                verify_paths=False,
            )
            item = dataset[0]
            self.assertNotIn("hq", item)
            self.assertEqual(item["lr"].shape, (13, 3, 4, 6))
            self.assertEqual(item["hr"].shape, (13, 3, 12, 18))

            with self.assertRaises(FileNotFoundError):
                TripletVideoDataset(
                    manifest,
                    split="train",
                    training=False,
                    clip_length=13,
                    crop_size=(4, 6),
                    scale=3,
                    load_hq=True,
                    horizontal_flip_probability=0.0,
                    vertical_flip_probability=0.0,
                )[0]

    def test_hq_skip_preserves_deterministic_view_identity_and_lr_hr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_sequence(root)
            common = dict(
                split="train",
                training=True,
                clip_length=13,
                crop_size=(4, 6),
                scale=3,
                horizontal_flip_probability=0.5,
                vertical_flip_probability=0.0,
            )
            with_hq = DeterministicTripletViewDataset(
                TripletVideoDataset(manifest, load_hq=True, **common),
                views_per_record=2,
                view_seed=20260805,
            )[1]
            without_hq = DeterministicTripletViewDataset(
                TripletVideoDataset(manifest, load_hq=False, **common),
                views_per_record=2,
                view_seed=20260805,
            )[1]

            for key in (
                "frame_indices",
                "crop_top",
                "crop_left",
                "horizontal_flip",
                "vertical_flip",
                "distillation_index",
                "distillation_view_index",
                "distillation_view_seed",
            ):
                left = with_hq[key]
                right = without_hq[key]
                if isinstance(left, torch.Tensor):
                    self.assertTrue(torch.equal(left, right), key)
                else:
                    self.assertEqual(left, right, key)
            self.assertTrue(torch.equal(with_hq["lr"], without_hq["lr"]))
            self.assertTrue(torch.equal(with_hq["hr"], without_hq["hr"]))
            self.assertNotIn("hq", without_hq)

    def test_worker_kwargs_only_emit_worker_only_options_when_enabled(self):
        self.assertEqual(
            dataloader_worker_kwargs(
                num_workers=0,
                prefetch_factor=2,
                persistent_workers=False,
            ),
            {"num_workers": 0},
        )
        self.assertEqual(
            dataloader_worker_kwargs(
                num_workers=2,
                prefetch_factor=3,
                persistent_workers=True,
            ),
            {
                "num_workers": 2,
                "prefetch_factor": 3,
                "persistent_workers": True,
            },
        )
        with self.assertRaisesRegex(ValueError, "requires num_workers"):
            dataloader_worker_kwargs(
                num_workers=0,
                prefetch_factor=2,
                persistent_workers=True,
            )
        with self.assertRaisesRegex(ValueError, "prefetch_factor"):
            dataloader_worker_kwargs(
                num_workers=1,
                prefetch_factor=0,
                persistent_workers=False,
            )


if __name__ == "__main__":
    unittest.main()
