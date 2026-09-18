import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn as nn


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "diagnose_b2a_backward.py"
SPEC = importlib.util.spec_from_file_location("diagnose_b2a_backward", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.good = nn.Parameter(torch.ones(2))
        self.bad = nn.Parameter(torch.ones(3))


class DiagnoseB2ABackwardTest(unittest.TestCase):
    def test_tensor_stats_counts_nonfinite(self):
        stats = MODULE.tensor_stats(torch.tensor([1.0, float("nan"), float("inf"), -float("inf")]))
        self.assertEqual(stats["nonfinite"], 3)
        self.assertEqual(stats["nan"], 1)
        self.assertEqual(stats["posinf"], 1)
        self.assertEqual(stats["neginf"], 1)
        self.assertEqual(stats["max_abs"], 1.0)

    def test_gradient_report_names_bad_parameter(self):
        model = Toy()
        model.good.grad = torch.tensor([1.0, 2.0])
        model.bad.grad = torch.tensor([0.0, float("nan"), float("inf")])
        report = MODULE.gradient_report(model, limit=4)
        self.assertEqual(report["nonfinite_elements"], 2)
        self.assertEqual(report["nan_elements"], 1)
        self.assertEqual(report["posinf_elements"], 1)
        self.assertEqual(report["missing_gradient_count"], 0)
        self.assertEqual(report["nonfinite_parameter_examples"][0]["name"], "bad")


if __name__ == "__main__":
    unittest.main()
