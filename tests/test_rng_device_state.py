"""CPU/mock tests for process-local CUDA RNG checkpoint handling."""

from __future__ import annotations

import random
import unittest
from unittest import mock

import numpy as np
import torch

from swiftvr.training.loop import capture_rng_state, restore_rng_state


def _base_state(cuda_states, **extra):
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
    }
    value.update(extra)
    return value


class CudaRngDeviceStateTest(unittest.TestCase):
    def test_capture_saves_only_current_cuda_device(self):
        expected = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.current_device", return_value=3
        ), mock.patch(
            "torch.cuda.get_rng_state", return_value=expected
        ) as get_rng_state:
            state = capture_rng_state()

        get_rng_state.assert_called_once_with(3)
        self.assertEqual(state["torch_cuda_scope"], "current_device")
        self.assertEqual(state["torch_cuda_device"], 3)
        self.assertEqual(len(state["torch_cuda"]), 1)
        torch.testing.assert_close(state["torch_cuda"][0], expected)

    def test_restore_maps_current_device_state_to_runtime_device(self):
        expected = torch.tensor([4, 5, 6], dtype=torch.uint8)
        state = _base_state(
            [expected],
            torch_cuda_scope="current_device",
            torch_cuda_device=7,
        )
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.current_device", return_value=1
        ), mock.patch("torch.cuda.set_rng_state") as set_rng_state:
            restore_rng_state(state)

        (actual,) = set_rng_state.call_args.args
        torch.testing.assert_close(actual, expected)
        self.assertEqual(set_rng_state.call_args.kwargs["device"], 1)

    def test_restore_legacy_all_device_state_selects_current_index(self):
        states = [
            torch.tensor([index], dtype=torch.uint8)
            for index in range(8)
        ]
        state = _base_state(states)
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.current_device", return_value=3
        ), mock.patch("torch.cuda.set_rng_state") as set_rng_state:
            restore_rng_state(state)

        (actual,) = set_rng_state.call_args.args
        torch.testing.assert_close(actual, states[3])
        self.assertEqual(set_rng_state.call_args.kwargs["device"], 3)

    def test_restore_legacy_eight_device_state_with_one_visible_gpu(self):
        states = [
            torch.tensor([index], dtype=torch.uint8)
            for index in range(8)
        ]
        state = _base_state(states)
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.current_device", return_value=0
        ), mock.patch("torch.cuda.set_rng_state") as set_rng_state:
            restore_rng_state(state)

        (actual,) = set_rng_state.call_args.args
        torch.testing.assert_close(actual, states[0])
        self.assertEqual(set_rng_state.call_args.kwargs["device"], 0)

    def test_current_device_format_rejects_multiple_states(self):
        state = _base_state(
            [
                torch.tensor([0], dtype=torch.uint8),
                torch.tensor([1], dtype=torch.uint8),
            ],
            torch_cuda_scope="current_device",
            torch_cuda_device=0,
        )
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.current_device", return_value=0
        ):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                restore_rng_state(state)


if __name__ == "__main__":
    unittest.main()
