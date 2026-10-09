"""CPU tests for tools/smoke_training_forward.py helper logic."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "smoke_training_forward.py"
_SPEC = importlib.util.spec_from_file_location("smoke_training_forward", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


class _DummyReAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 4)


class _DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_free_adapter = nn.Sequential(
            nn.Linear(4, 2),
            nn.SiLU(),
            nn.Linear(2, 4),
        )
        self.ffn = nn.Linear(4, 4)


class _DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock(), _DummyBlock()])
        self.proj_out = nn.Linear(4, 4)


class SmokeTrainingForwardHelperTest(unittest.TestCase):
    def _write_checkpoint(self, root: Path, **config_updates) -> dict[str, object]:
        transformer = root / "transformer"
        transformer.mkdir(parents=True)
        (root / "reae.safetensors").write_bytes(b"dummy")
        (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"dummy")
        config: dict[str, object] = {
            "_class_name": "WanTransformer3DModelPromptFreeNoTime",
            "time_condition_folded": True,
            "folded_runtime_dtype": "float32",
            "folded_timestep": 1000.0,
        }
        config.update(config_updates)
        (transformer / "config.json").write_text(
            json.dumps(config),
            encoding="utf-8",
        )
        return config

    def test_validate_folded_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self._write_checkpoint(root)
            actual = _TOOL.validate_folded_checkpoint(root)
            self.assertEqual(actual, expected)

    def test_validate_rejects_non_folded_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_checkpoint(root, time_condition_folded=False)
            with self.assertRaisesRegex(ValueError, "time_condition_folded"):
                _TOOL.validate_folded_checkpoint(root)

    def test_auto_dtype_uses_folded_dtype(self):
        dtype = _TOOL.resolve_runtime_dtype(
            "auto",
            {"folded_runtime_dtype": "float32"},
            torch.device("cpu"),
        )
        self.assertEqual(dtype, torch.float32)

    def test_explicit_dtype_mismatch_requires_opt_in(self):
        config = {"folded_runtime_dtype": "float16"}
        with self.assertRaisesRegex(ValueError, "does not match"):
            _TOOL.resolve_runtime_dtype(
                "float32",
                config,
                torch.device("cpu"),
            )
        dtype = _TOOL.resolve_runtime_dtype(
            "float32",
            config,
            torch.device("cpu"),
            allow_mismatch=True,
        )
        self.assertEqual(dtype, torch.float32)

    def test_adapter_scope_freezes_reae_and_non_adapter_dit(self):
        reae = _DummyReAE()
        transformer = _DummyTransformer()
        counts = _TOOL.configure_train_scope(reae, transformer, "adapter")

        self.assertEqual(counts["reae_trainable"], 0)
        self.assertGreater(counts["transformer_trainable"], 0)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in reae.parameters())
        )
        for name, parameter in transformer.named_parameters():
            self.assertEqual(
                parameter.requires_grad,
                "prompt_free_adapter" in name,
                msg=name,
            )

    def test_transformer_and_all_scopes(self):
        reae = _DummyReAE()
        transformer = _DummyTransformer()
        _TOOL.configure_train_scope(reae, transformer, "transformer")
        self.assertTrue(all(parameter.requires_grad for parameter in transformer.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in reae.parameters()))

        _TOOL.configure_train_scope(reae, transformer, "all")
        self.assertTrue(all(parameter.requires_grad for parameter in transformer.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in reae.parameters()))

    def test_move_video_batch_preserves_metadata(self):
        batch = {
            "lr": torch.ones(1, 1, 3, 2, 2),
            "hq": torch.ones(1, 1, 3, 2, 2),
            "hr": torch.ones(1, 1, 3, 6, 6),
            "sample_id": ["vid0"],
        }
        moved = _TOOL.move_video_batch(
            batch,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        self.assertEqual(moved["lr"].dtype, torch.float64)
        self.assertEqual(moved["hq"].dtype, torch.float64)
        self.assertEqual(moved["hr"].dtype, torch.float64)
        self.assertEqual(moved["sample_id"], ["vid0"])

    def test_gradient_summary_reports_finite_gradients(self):
        module = nn.Linear(3, 2)
        output = module(torch.ones(1, 3)).sum()
        output.backward()
        summary = _TOOL.gradient_summary(module.named_parameters())
        self.assertEqual(summary["nonfinite_elements"], 0)
        self.assertEqual(summary["missing_gradient_count"], 0)
        self.assertGreater(summary["gradient_tensors"], 0)
        self.assertGreater(summary["global_l2"], 0.0)


if __name__ == "__main__":
    unittest.main()
