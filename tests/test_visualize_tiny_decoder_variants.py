from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tools import visualize_tiny_decoder_variants as visual


class TinyDecoderVariantVisualizerTests(unittest.TestCase):
    def _args(self, *extra: str):
        parser = visual.build_parser()
        return parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--decoder", "cd45=canonical",
                "--val-cache", "cache",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
                *extra,
            ]
        )

    def test_decoder_spec_requires_label_and_path(self):
        label, path = visual._parse_decoder_spec("resize_e10=/tmp/model")
        self.assertEqual(label, "resize_e10")
        self.assertEqual(path, Path("/tmp/model"))
        with self.assertRaises(argparse.ArgumentTypeError):
            visual._parse_decoder_spec("/tmp/model")
        with self.assertRaises(argparse.ArgumentTypeError):
            visual._parse_decoder_spec("label=")

    def test_duplicate_labels_are_rejected(self):
        args = self._args("--decoder", "cd45=other")
        with self.assertRaisesRegex(ValueError, "labels must be unique"):
            visual._validate_args(args)

    def test_checkpoint_class_routing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            resize = root / "resize"
            canonical.mkdir()
            resize.mkdir()
            (canonical / "config.json").write_text(
                json.dumps({"class_name": "TinyConditionalDecoder"}), encoding="utf-8"
            )
            (resize / "config.json").write_text(
                json.dumps({"class_name": "ResizeConvTinyConditionalDecoder"}), encoding="utf-8"
            )
            self.assertEqual(
                visual._checkpoint_class_name(canonical), "TinyConditionalDecoder"
            )
            self.assertEqual(
                visual._checkpoint_class_name(resize), "ResizeConvTinyConditionalDecoder"
            )

    def test_unknown_checkpoint_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps({"class_name": "UnknownDecoder"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Unsupported Tiny decoder class_name"):
                visual._checkpoint_class_name(root)


if __name__ == "__main__":
    unittest.main()
