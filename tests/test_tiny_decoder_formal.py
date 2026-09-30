from __future__ import annotations

import argparse
import unittest

from tools import train_tiny_decoder_formal_ddp as formal


class _Cache:
    def __init__(self, **metadata):
        self.metadata = metadata


class TinyDecoderFormalTests(unittest.TestCase):
    def test_formal_defaults_match_frozen_recipe(self):
        parser = formal.build_parser()
        args = parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--init-decoder", "tiny",
                "--train-cache", "train",
                "--val-cache", "val",
                "--manifest", "train.jsonl",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
            ]
        )
        self.assertEqual(args.views_per_record, 8)
        self.assertEqual(args.view_seed, 20260805)
        self.assertEqual(args.val_views_per_record, 1)
        self.assertEqual(args.val_view_seed, 9000001)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.epochs, 8)
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertEqual(args.gt_l2_weight, 1.0)
        self.assertEqual(args.teacher_l2_weight, 1.0)
        self.assertEqual(args.lpips_weight, 2.0)
        self.assertEqual(args.dtype, "bfloat16")
        formal._validate_args(args)

    def test_cache_pair_requires_same_stage_a_source(self):
        common = {
            "base_checkpoint": "/base",
            "source_checkpoint": "/long/step",
            "source_weights_sha256": "weights",
            "source_metadata_sha256": "metadata",
            "reae_sha256": "reae",
            "transformer_config_sha256": "transformer",
        }
        formal._validate_cache_pair(_Cache(**common), _Cache(**common))
        changed = dict(common)
        changed["source_weights_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "source_weights_sha256"):
            formal._validate_cache_pair(_Cache(**common), _Cache(**changed))

    def test_resume_fingerprint_detects_training_semantic_change(self):
        saved = {
            "world_size": 4,
            "local_batch_size": 4,
            "lpips_weight": 2.0,
            "train_cache_metadata_sha256": "abc",
        }
        formal._assert_fingerprint(saved, dict(saved))
        changed = dict(saved)
        changed["local_batch_size"] = 8
        with self.assertRaisesRegex(ValueError, "local_batch_size"):
            formal._assert_fingerprint(saved, changed)

    def test_invalid_formal_view_configuration_is_rejected(self):
        parser = formal.build_parser()
        args = parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--init-decoder", "tiny",
                "--train-cache", "train",
                "--val-cache", "val",
                "--manifest", "train.jsonl",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
                "--clip-length", "12",
            ]
        )
        with self.assertRaisesRegex(ValueError, "4k\\+1"):
            formal._validate_args(args)


if __name__ == "__main__":
    unittest.main()
