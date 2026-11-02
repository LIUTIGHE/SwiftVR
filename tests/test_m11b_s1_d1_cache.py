"""CPU contract tests for the M11B-S1-D1 paired restored-latent cache builder."""
from __future__ import annotations
import argparse
import ast
import inspect
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools import m11b_s1_d1_cache as cache_module
from tools.m11b_s1_d1_cache import (
    _verify_cached, cache_plan, sample_file_name, validate_args,
)
from swiftvr.training.forward import prepare_training_batch


class M11BS1D1CacheTests(unittest.TestCase):
    def test_encoding_block_is_executable_and_ordered(self):
        # A past edit accidentally wrote literal backslash-n characters inside a
        # comment, swallowing prepare_training_batch() and encode_reae_clip().
        tree = ast.parse(inspect.getsource(cache_module.main))
        call_lines = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                call_lines.setdefault(node.func.id, []).append(node.lineno)
        for function in ("prepare_training_batch", "encode_reae_clip",
                         "extract_transformer_sample"):
            self.assertIn(function, call_lines, msg=f"{function} is not executable")
        self.assertLess(call_lines["prepare_training_batch"][0],
                        call_lines["encode_reae_clip"][0])
        self.assertLess(call_lines["encode_reae_clip"][0],
                        call_lines["extract_transformer_sample"][0])

    def test_sample_identity_path(self):
        self.assertEqual(sample_file_name(12, "legacy:sample"), sample_file_name(12, "legacy:sample"))
        self.assertNotEqual(sample_file_name(12, "legacy:sample"), sample_file_name(12, "ultra:sample"))
        self.assertTrue(sample_file_name(12, "legacy:sample").startswith("samples/00000012_"))

    def test_plan_binds_source_file_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "source.txt"
            path.write_text("old")
            a = argparse.Namespace(
                domain="ultra", base_checkpoint=Path(d), source_checkpoint=Path(d),
                legacy_split="train", dtype="bfloat16", attention_backend="sdpa"
            )
            plan_a = cache_plan(a, [path], total=10, selected=2)
            path.write_text("new")
            plan_b = cache_plan(a, [path], total=10, selected=2)
            self.assertNotEqual(plan_a["source_files"], plan_b["source_files"])
            self.assertEqual(plan_a["source_checkpoint"], str(Path(d).resolve()))

    def test_reject_inconsistent_domains(self):
        args = argparse.Namespace(
            domain="legacy", legacy_view_cache=None, ultra_pairs=[],
            allow_partial_ultra=False, max_samples=4, batch_size=1,
            progress_every=1, num_workers=0
        )
        with self.assertRaises(ValueError):
            validate_args(args)
        args.domain = "ultra"
        args.ultra_pairs = [Path("paired_views.jsonl")]
        validate_args(args)
        args.allow_partial_ultra = True
        args.max_samples = 0
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_cached_latent_dtype_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.safetensors"
            save_file({"z_sr": torch.ones(4, 48, 24, 24).half()}, str(p))
            self.assertEqual(_verify_cached(p, expected_shape=(4, 48, 24, 24)), (4, 48, 24, 24))
            with self.assertRaises(ValueError):
                _verify_cached(p, expected_shape=(4, 48, 8, 8))

    def test_canonical_3x_lr_upscaling_before_encode(self):
        lr = torch.rand(1, 13, 3, 16, 16)
        hr = torch.rand(1, 13, 3, 48, 48)
        prepared = prepare_training_batch({"lr": lr, "hr": hr}, auxiliary_key=None)
        self.assertEqual(tuple(prepared["lq_input"].shape), (1, 13, 3, 48, 48))
        self.assertEqual(tuple(prepared["target"].shape), (1, 13, 3, 48, 48))
        self.assertEqual(tuple(prepared["lq_input"][..., ::3, ::3].shape), (1, 13, 3, 16, 16))


if __name__ == "__main__":
    unittest.main()
