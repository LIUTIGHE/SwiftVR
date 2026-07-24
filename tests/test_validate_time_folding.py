"""Regression test for mixed-dtype time-fold validation setup."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn as nn


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "validate_time_folding.py"
_SPEC = importlib.util.spec_from_file_location("validate_time_folding", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_VALIDATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VALIDATOR)


class _MixedDtypeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp32 = nn.Linear(4, 4).float()
        self.fp16 = nn.Linear(4, 4).half()


class ValidateTimeFoldingTest(unittest.TestCase):
    def test_prepare_model_matches_pipeline_runtime_dtype(self):
        model = _MixedDtypeModule()
        before = {parameter.dtype for parameter in model.parameters()}
        self.assertEqual(before, {torch.float32, torch.float16})

        prepared = _VALIDATOR._prepare_model_for_runtime(
            model,
            torch.device("cpu"),
            torch.float16,
        )

        after = {parameter.dtype for parameter in prepared.parameters()}
        self.assertEqual(after, {torch.float16})
        self.assertFalse(prepared.training)

    def test_dtype_summary_counts_parameters(self):
        model = _MixedDtypeModule()
        summary = _VALIDATOR._parameter_dtype_summary(model)

        self.assertEqual(summary["float32"], 20)
        self.assertEqual(summary["float16"], 20)


if __name__ == "__main__":
    unittest.main()
