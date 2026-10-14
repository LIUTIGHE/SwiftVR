from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from tools import diagnose_tiny_decoder_variant_phase_bias as audit


class TinyDecoderVariantPhaseBiasTests(unittest.TestCase):
    def _args(self, *extra: str):
        parser = audit.build_parser()
        return parser.parse_args(
            [
                "--base-checkpoint", "base",
                "--decoder", "CD45=canonical",
                "--val-cache", "cache",
                "--val-manifest", "val.jsonl",
                "--output-dir", "out",
                *extra,
            ]
        )

    def test_parser_defaults_to_full_multiscale_audit(self):
        args = self._args()
        self.assertEqual(args.periods, (2, 4, 8, 16))
        self.assertIsNone(args.sample_indices)
        self.assertEqual(args.dtype, "bfloat16")
        audit._validate_args(args)

    def test_duplicate_labels_are_rejected(self):
        args = self._args("--decoder", "CD45=other")
        with self.assertRaisesRegex(ValueError, "labels must be unique"):
            audit._validate_args(args)

    def test_decoder_spec_requires_label_and_path(self):
        parser = audit.build_parser()
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--base-checkpoint", "base",
                    "--decoder", "missing_equals",
                    "--val-cache", "cache",
                    "--val-manifest", "val.jsonl",
                    "--output-dir", "out",
                ]
            )
        self.assertIn("--decoder expects LABEL=PATH", stderr.getvalue())

    def test_checkpoint_class_routing_accepts_both_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = root / "canonical"
            resize = root / "resize"
            canonical.mkdir()
            resize.mkdir()
            (canonical / "config.json").write_text(
                json.dumps({"class_name": "TinyConditionalDecoder"}),
                encoding="utf-8",
            )
            (resize / "config.json").write_text(
                json.dumps({"class_name": "ResizeConvTinyConditionalDecoder"}),
                encoding="utf-8",
            )
            self.assertEqual(
                audit._checkpoint_class_name(canonical),
                "TinyConditionalDecoder",
            )
            self.assertEqual(
                audit._checkpoint_class_name(resize),
                "ResizeConvTinyConditionalDecoder",
            )

    def test_unknown_checkpoint_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(
                json.dumps({"class_name": "UnknownDecoder"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported Tiny decoder"):
                audit._checkpoint_class_name(root)


if __name__ == "__main__":
    unittest.main()
