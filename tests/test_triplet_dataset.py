"""CPU tests for swiftvr/data/triplet_dataset.py."""

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
_TOOL_PATH = _REPO_ROOT / "swiftvr" / "data" / "triplet_dataset.py"
_SPEC = importlib.util.spec_from_file_location("triplet_dataset", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)

TripletVideoDataset = _TOOL.TripletVideoDataset
build_triplet_dataloader = _TOOL.build_triplet_dataloader
read_triplet_manifests = _TOOL.read_triplet_manifests


def _frame_array(index: int, height: int = 8, width: int = 10) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.uint8),
        np.arange(width, dtype=np.uint8),
        indexing="ij",
    )
    return np.stack(
        [
            (xx + index * 7) % 255,
            (yy + index * 11) % 255,
            np.full_like(xx, index * 13 % 255),
        ],
        axis=-1,
    )


def _write_explicit_sequence(
    root: Path,
    *,
    sample_id: str,
    count: int,
    suffix: str = "",
    split: str = "train",
) -> dict[str, object]:
    paths = {"hr": [], "hq": [], "lr": []}
    for index in range(count):
        base = _frame_array(index)
        hr = np.repeat(np.repeat(base, 3, axis=0), 3, axis=1)
        for name, array in (("hr", hr), ("hq", base), ("lr", base)):
            folder = root / name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{sample_id}_{index:06d}{suffix}.png"
            Image.fromarray(array).save(path)
            paths[name].append(str(path))
    return {
        "sample_id": sample_id,
        "split": split,
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


class TripletVideoDatasetTest(unittest.TestCase):
    def test_explicit_manifest_center_clip_and_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _write_explicit_sequence(root, sample_id="vid0", count=21)
            manifest = root / "vsr_triplets_plain_train.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            dataset = TripletVideoDataset(
                manifest,
                split="train",
                training=False,
                clip_length=17,
                crop_size=(4, 6),
                scale=3,
                horizontal_flip_probability=0.0,
            )
            item = dataset[0]
            self.assertEqual(item["lr"].shape, (17, 3, 4, 6))
            self.assertEqual(item["hq"].shape, (17, 3, 4, 6))
            self.assertEqual(item["hr"].shape, (17, 3, 12, 18))
            self.assertEqual(item["temporal_start"], 2)
            self.assertEqual(item["crop_top"], 2)
            self.assertEqual(item["crop_left"], 2)
            self.assertEqual(item["frame_indices"].tolist(), list(range(2, 19)))
            self.assertTrue(torch.equal(item["lr"], item["hq"]))
            self.assertTrue(torch.equal(item["hr"][:, :, ::3, ::3], item["hq"]))
            self.assertEqual(item["variant"], "plain")

    def test_explicit_text_and_plain_manifests_can_be_combined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = _write_explicit_sequence(root / "plain", sample_id="vid0", count=17)
            text = _write_explicit_sequence(
                root / "text", sample_id="vid0", count=17, suffix="_text"
            )
            plain_manifest = root / "vsr_triplets_plain_train.jsonl"
            text_manifest = root / "vsr_triplets_text_train.jsonl"
            plain_manifest.write_text(json.dumps(plain) + "\n", encoding="utf-8")
            text_manifest.write_text(json.dumps(text) + "\n", encoding="utf-8")

            dataset = TripletVideoDataset(
                [plain_manifest, text_manifest],
                split="train",
                training=False,
                clip_length=17,
                crop_size=None,
                scale=3,
            )
            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                {dataset[index]["record_uid"] for index in range(2)},
                {"plain:vid0", "text:vid0"},
            )

    def test_pattern_manifest_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("hr", "hq", "lr"):
                (root / name).mkdir()
            for index in range(17):
                base = _frame_array(index)
                Image.fromarray(base).save(root / "hq" / f"clip_{index:06d}.png")
                Image.fromarray(base).save(root / "lr" / f"clip_{index:06d}.png")
                Image.fromarray(
                    np.repeat(np.repeat(base, 3, axis=0), 3, axis=1)
                ).save(root / "hr" / f"clip_{index:06d}.png")
            row = {
                "sample_id": "clip",
                "split": "train",
                "media_type": "frames",
                "frame_start": 0,
                "frame_end": 16,
                "frame_count": 17,
                "hr": str(root / "hr" / "clip_{frame:06d}.png"),
                "hq": str(root / "hq" / "clip_{frame:06d}.png"),
                "lr": str(root / "lr" / "clip_{frame:06d}.png"),
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            records = read_triplet_manifests(manifest, split="train", verify_paths=True)
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0].hq_paths[3].endswith("clip_000003.png"))
            item = TripletVideoDataset(
                manifest,
                split="train",
                training=False,
                clip_length=17,
                scale=3,
            )[0]
            self.assertEqual(item["hq"].shape, (17, 3, 8, 10))

    def test_short_sequence_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            short = _write_explicit_sequence(root / "short", sample_id="short", count=9)
            full = _write_explicit_sequence(root / "full", sample_id="full", count=17)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(short) + "\n" + json.dumps(full) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "shorter than clip_length"):
                TripletVideoDataset(manifest, clip_length=17, scale=3)
            dataset = TripletVideoDataset(
                manifest,
                clip_length=17,
                scale=3,
                drop_short_sequences=True,
                training=False,
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.dropped_short_count, 1)
            self.assertEqual(dataset.records[0].sample_id, "full")

    def test_swiftvr_temporal_constraint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _write_explicit_sequence(root, sample_id="vid0", count=20)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "T=4k\\+1"):
                TripletVideoDataset(manifest, clip_length=16, scale=3)
            dataset = TripletVideoDataset(
                manifest,
                clip_length=16,
                require_4k_plus_1=False,
                scale=3,
                training=False,
            )
            self.assertEqual(dataset[0]["hq"].shape[0], 16)

    def test_dataloader_batches_btchw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                _write_explicit_sequence(root / f"s{i}", sample_id=f"vid{i}", count=17)
                for i in range(2)
            ]
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            dataset, loader = build_triplet_dataloader(
                manifest,
                batch_size=2,
                num_workers=0,
                shuffle=False,
                drop_last=False,
                pin_memory=False,
                training=False,
                clip_length=17,
                crop_size=(4, 6),
                scale=3,
            )
            batch = next(iter(loader))
            self.assertEqual(len(dataset), 2)
            self.assertEqual(batch["lr"].shape, (2, 17, 3, 4, 6))
            self.assertEqual(batch["hr"].shape, (2, 17, 3, 12, 18))
            self.assertEqual(batch["frame_indices"].shape, (2, 17))

    def test_relative_paths_use_path_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _write_explicit_sequence(root / "data", sample_id="vid0", count=17)
            for field in ("hr_frames", "hq_frames", "lr_frames"):
                row[field] = [
                    str(Path(path).relative_to(root)) for path in row[field]
                ]
            manifest = root / "manifests" / "manifest.jsonl"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = TripletVideoDataset(
                manifest,
                path_root=root,
                training=False,
                clip_length=17,
                scale=3,
                verify_paths=True,
            )
            self.assertEqual(dataset[0]["sample_id"], "vid0")


if __name__ == "__main__":
    unittest.main()
