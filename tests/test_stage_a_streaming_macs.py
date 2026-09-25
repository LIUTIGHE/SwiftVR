from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch.nn as nn

from tools.finalize_stage_a_audit_macs import _format_compute_row
from tools.profile_stage_a_streaming_macs import (
    canonical_parameter_summary,
    parse_resolution,
)


class _FakeReAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 5, bias=False))
        self.decoder = nn.Sequential(nn.Linear(5, 3, bias=False))


class StageAStreamingMacHelpersTest(unittest.TestCase):
    def test_parse_resolution(self):
        self.assertEqual(parse_resolution("1920x1080"), (1920, 1080))
        with self.assertRaises(Exception):
            parse_resolution("1920")
        with self.assertRaises(Exception):
            parse_resolution("0x1080")

    def test_canonical_parameter_summary(self):
        reae = _FakeReAE()
        transformer = nn.Linear(7, 11, bias=False)
        summary = canonical_parameter_summary(reae, transformer)
        self.assertEqual(summary["encoder_params"], 3 * 5)
        self.assertEqual(summary["decoder_params"], 5 * 3)
        self.assertEqual(summary["transformer_params"], 7 * 11)
        self.assertEqual(summary["total_params"], 15 + 15 + 77)

    def test_compute_row_uses_component_sum_and_two_flops_per_mac(self):
        record = {
            "parameters": {"total_params": 1000},
            "macs": {
                "gmacs_per_output_frame": 60.0,
                "by_root_gmacs_per_output_frame": {
                    "encoder": 10.0,
                    "transformer": 40.0,
                    "decoder": 10.0,
                },
            },
        }
        row = _format_compute_row("student", record, teacher_total=100.0)
        self.assertIn("60.000", row)
        self.assertIn("120.000", row)
        self.assertIn("0.6000×", row)
        self.assertIn("40.00%", row)

    def test_compute_row_rejects_inconsistent_component_sum(self):
        record = {
            "parameters": {"total_params": 1000},
            "macs": {
                "gmacs_per_output_frame": 61.0,
                "by_root_gmacs_per_output_frame": {
                    "encoder": 10.0,
                    "transformer": 40.0,
                    "decoder": 10.0,
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "Component MAC sum mismatch"):
            _format_compute_row("student", record, teacher_total=100.0)


if __name__ == "__main__":
    unittest.main()
