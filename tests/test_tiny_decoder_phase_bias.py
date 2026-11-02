from __future__ import annotations

import math
import unittest

import torch

from tools.diagnose_tiny_decoder_phase_bias import PhaseAccumulator, _parse_decoder_spec


class TinyDecoderPhaseBiasTests(unittest.TestCase):
    def test_period2_checkerboard_basis_detects_alternating_pattern(self):
        # One RGB frame whose residual is +a on even/even and odd/odd pixels,
        # and -a on the other two parity classes. Equal RGB channels make luma
        # exactly the same pattern.
        amplitude = 0.02
        residual = torch.empty(1, 1, 3, 8, 8, dtype=torch.float32)
        for y in range(8):
            for x in range(8):
                value = amplitude if (y + x) % 2 == 0 else -amplitude
                residual[0, 0, :, y, x] = value

        accumulator = PhaseAccumulator((2, 4))
        accumulator.update(residual)
        result = accumulator.finalize()

        basis = result["period2_basis"]
        self.assertAlmostEqual(basis["checkerboard"]["mean_abs"], amplitude, places=6)
        self.assertAlmostEqual(basis["checkerboard"]["rms"], amplitude, places=6)
        self.assertAlmostEqual(basis["horizontal"]["mean_abs"], 0.0, places=6)
        self.assertAlmostEqual(basis["vertical"]["mean_abs"], 0.0, places=6)
        self.assertAlmostEqual(basis["dc"]["mean_abs"], 0.0, places=6)

        p2 = result["periods"]["2"]
        grid = p2["phase_mean_luma"]
        self.assertAlmostEqual(grid[0][0], amplitude, places=6)
        self.assertAlmostEqual(grid[0][1], -amplitude, places=6)
        self.assertAlmostEqual(grid[1][0], -amplitude, places=6)
        self.assertAlmostEqual(grid[1][1], amplitude, places=6)
        self.assertGreater(p2["phase_mean_std"], 0.0)

        # The p=4 grid contains only the inherited p=2 pattern, so after
        # subtracting the nested parent there should be essentially no new p=4
        # phase structure.
        p4 = result["periods"]["4"]
        self.assertLess(p4["new_phase_std_beyond_parent"], 1e-7)

    def test_constant_residual_is_dc_not_checkerboard(self):
        residual = torch.full((2, 3, 3, 8, 8), 0.01, dtype=torch.float32)
        accumulator = PhaseAccumulator((2,))
        accumulator.update(residual)
        result = accumulator.finalize()
        basis = result["period2_basis"]
        self.assertAlmostEqual(basis["dc"]["mean"], 0.01, places=6)
        self.assertAlmostEqual(basis["checkerboard"]["mean_abs"], 0.0, places=6)
        self.assertAlmostEqual(basis["horizontal"]["mean_abs"], 0.0, places=6)
        self.assertAlmostEqual(basis["vertical"]["mean_abs"], 0.0, places=6)

    def test_decoder_spec_supports_named_and_unnamed_paths(self):
        name, path = _parse_decoder_spec("baseline=/tmp/checkpoint/tiny_decoder")
        self.assertEqual(name, "baseline")
        self.assertEqual(str(path), "/tmp/checkpoint/tiny_decoder")

        name, path = _parse_decoder_spec("/tmp/checkpoint/cd020")
        self.assertEqual(name, "cd020")
        self.assertEqual(str(path), "/tmp/checkpoint/cd020")


if __name__ == "__main__":
    unittest.main()
