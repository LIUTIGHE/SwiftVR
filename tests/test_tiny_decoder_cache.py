from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from swiftvr.training.tiny_decoder_cache import TinyDecoderLatentCache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TinyDecoderLatentCacheTests(unittest.TestCase):
    def test_load_and_identity_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text('{"split":"train"}\n', encoding="utf-8")
            samples = root / "samples"
            samples.mkdir()
            latent = torch.randn(4, 48, 2, 3, dtype=torch.float16)
            filename = "samples/00000000_deadbeef.safetensors"
            save_file({"z_sr": latent}, str(root / filename))
            identity = {
                "distillation_index": 0,
                "key": "deadbeef",
                "record_uid": "record0",
                "frame_indices": [0, 1, 2, 3, 4],
                "crop_top": 1,
                "crop_left": 2,
                "horizontal_flip": False,
                "vertical_flip": False,
                "view_index": 0,
                "view_seed": 123,
            }
            metadata = {
                "format_version": 1,
                "kind": "swiftvr_stage_b1_sr_latent",
                "sample_count": 1,
                "selected_indices": [0],
                "split": "train",
                "clip_length": 5,
                "crop_size": 32,
                "scale": 3,
                "views_per_record": 1,
                "view_seed": 123,
                "horizontal_flip_probability": 0.0,
                "vertical_flip_probability": 0.0,
                "full_dataset_length": 1,
                "manifest_sha256": {str(manifest.resolve()): _sha256(manifest)},
                "samples": [{**identity, "file": filename}],
            }
            (root / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            cache = TinyDecoderLatentCache(root)
            actual = cache.load(identity)
            self.assertTrue(torch.equal(actual, latent))
            self.assertEqual(cache.selected_indices(), (0,))
            cache.validate_dataset(
                manifests=[manifest],
                split="train",
                clip_length=5,
                crop_size=32,
                scale=3,
                views_per_record=1,
                view_seed=123,
                horizontal_flip_probability=0.0,
                vertical_flip_probability=0.0,
                dataset_length=1,
            )

            wrong = dict(identity)
            wrong["crop_left"] = 99
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                cache.load(wrong)

    def test_rejects_dataset_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            samples = root / "samples"
            samples.mkdir()
            save_file(
                {"z_sr": torch.zeros(1, 48, 1, 1)},
                str(samples / "sample.safetensors"),
            )
            metadata = {
                "format_version": 1,
                "kind": "swiftvr_stage_b1_sr_latent",
                "sample_count": 1,
                "selected_indices": [0],
                "split": "train",
                "clip_length": 5,
                "crop_size": 32,
                "scale": 3,
                "views_per_record": 1,
                "view_seed": 1,
                "horizontal_flip_probability": 0.0,
                "vertical_flip_probability": 0.0,
                "full_dataset_length": 1,
                "manifest_sha256": {str(manifest.resolve()): _sha256(manifest)},
                "samples": [
                    {
                        "distillation_index": 0,
                        "key": "k",
                        "record_uid": "r",
                        "frame_indices": [0, 1, 2, 3, 4],
                        "crop_top": 0,
                        "crop_left": 0,
                        "horizontal_flip": False,
                        "vertical_flip": False,
                        "view_index": 0,
                        "view_seed": 1,
                        "file": "samples/sample.safetensors",
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            cache = TinyDecoderLatentCache(root)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                cache.validate_dataset(
                    manifests=[manifest],
                    split="val",
                    clip_length=5,
                    crop_size=32,
                    scale=3,
                    views_per_record=1,
                    view_seed=1,
                    horizontal_flip_probability=0.0,
                    vertical_flip_probability=0.0,
                    dataset_length=1,
                )


if __name__ == "__main__":
    unittest.main()
