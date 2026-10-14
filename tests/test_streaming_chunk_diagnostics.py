"""CPU regressions for opt-in canonical-runner observations and paired reports."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRACE = load("chunk_trace_test_target", "swiftvr/streaming/chunk_trace.py")
COMP = load("chunk_compare_test_target", "tools/compare_streaming_chunk_traces.py")


class StreamingChunkDiagnosticsTest(unittest.TestCase):
    def fixture(self, root, name, counts=(28, 24, 24, 21), clip_len=24):
        source = root / "input"
        source.mkdir(exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(source / "00000000.png")
        output = root / name / "png"
        args = SimpleNamespace(png=True, dit_overlap=0, input=str(source),
                               output=str(output), clip_len=clip_len,
                               chunk_trace=str(root / name / "trace"), dtype="bfloat16")
        emitted = []
        class TAE:
            def encode_chunk_fixed(self, x, spec):
                n = (spec.frame_count + (3 if spec.ctype.value == "last" else 0)) // 4
                self.encoded = torch.ones(1, n, 2, 2, 2, dtype=torch.bfloat16)
                return self.encoded
            def decode_chunk_fixed(self, z, spec):
                n = spec.frame_count + (3 if spec.ctype.value == "last" else 0) - (3 if spec.is_first_decode else 0)
                self.decoded = torch.ones(1, n, 3, 8, 8)
                emitted.append(n)
                return self.decoded
        pipe = SimpleNamespace(tae_stream=TAE(), reae=SimpleNamespace(frames_to_trim=3))
        trace = TRACE.ChunkTrace(pipe, args.chunk_trace, args)
        start = 0
        for i, count in enumerate(counts):
            kind = "last" if i == len(counts) - 1 else ("first" if i == 0 else "middle")
            spec = SimpleNamespace(clip_idx=i, ctype=SimpleNamespace(value=kind),
                                   frame_start=start, frame_count=count, is_first_decode=i == 0)
            x = torch.zeros(1, count, 3, 8, 8)
            z = pipe.tae_stream.encode_chunk_fixed(x, spec)
            self.assertIs(z, pipe.tae_stream.encoded)
            rgb = pipe.tae_stream.decode_chunk_fixed(z * 2, spec)
            self.assertIs(rgb, pipe.tae_stream.decoded)
            start += count
        output.mkdir(parents=True)
        pattern = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        for i in range(sum(emitted)):
            Image.fromarray(pattern).save(output / f"{i:08d}.png")
        report = trace.finish({"frames": sum(emitted)})
        return trace.root, report, pattern

    def test_trace_preserves_returned_tensors_and_observes_emissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, info, _ = self.fixture(Path(tmp), "a")
            self.assertEqual(info["frames"], 97)
            self.assertEqual(info["latent_frames"], 25)
            self.assertEqual([r["output_start"] for r in info["chunks"]], [0, 25, 49, 73])
            stored = load_file(str(root / "chunk_00000.safetensors"))
            self.assertEqual(stored["z_lq"].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(stored["z_sr"], stored["z_lq"] * 2))

    def test_pair_detects_encoder_and_restored_identity_on_different_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            a, _, _ = self.fixture(path, "a")
            b, _, _ = self.fixture(path, "b", (52, 45), 48)
            summary = COMP.compare(a, b, path / "comparison", (0, 0, 8, 8), no_video=True)
            self.assertEqual(summary["output_chunk_boundaries_b"], [49])
            self.assertEqual(summary["rgb_roi"]["mse"], 0)
            self.assertEqual(summary["latent"]["z_lq"]["mse"], 0)
            self.assertEqual(summary["latent"]["z_sr"]["mse"], 0)
            self.assertEqual(sum(v["count"] for v in summary["rgb_roi_by_phase"].values()), 97)
            self.assertEqual(len(list((path / "comparison/native_roi_frames").glob("*.png"))), 97)

    def test_reject_input_or_model_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _, a, _ = self.fixture(path, "a", (13,), 24)
            _, b, _ = self.fixture(path, "b", (13,), 48)
            COMP.validate_pair(a, b)
            b["arguments"]["dtype"] = "float16"
            with self.assertRaisesRegex(ValueError, "Only clip-len"):
                COMP.validate_pair(a, b)
            b["arguments"]["dtype"] = "bfloat16"
            b["input_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "input_sha256"):
                COMP.validate_pair(a, b)

    def test_reject_stale_output_and_unsupported_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, "a", (13,))
            with self.assertRaises(FileExistsError):
                self.fixture(root, "a", (13,))
            with self.assertRaisesRegex(ValueError, "dit-overlap"):
                TRACE.ChunkTrace(None, root / "trace", SimpleNamespace(png=True, dit_overlap=1))

    def test_native_panel_does_not_resize_or_alter_pixels(self):
        a = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
        b = np.flip(a, axis=1).copy()
        panel = np.asarray(COMP.native_panel(a, b, "a", "b"))
        np.testing.assert_array_equal(panel[32:, :11], a)
        np.testing.assert_array_equal(panel[32:, 11:], b)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_lossless_video_matches_native_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            a, _, pattern = self.fixture(path, "a", (5,), 4)
            b, _, _ = self.fixture(path, "b", (5,), 8)
            COMP.compare(a, b, path / "cmp", (0, 0, 8, 8))
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path / "cmp/native_roi.mkv"),
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
            decoded = np.frombuffer(raw, np.uint8).reshape(5, 40, 16, 3)
            for frame in decoded:
                np.testing.assert_array_equal(frame[32:, :8], pattern)
                np.testing.assert_array_equal(frame[32:, 8:], pattern)


if __name__ == "__main__":
    unittest.main()
