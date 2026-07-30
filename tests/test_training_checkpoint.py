"""CPU tests for lightweight SwiftVR training checkpoints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from swiftvr.training import (
    capture_trainable_parameters,
    load_delta_checkpoint,
    parameter_update_summary,
    save_delta_checkpoint,
    trainable_named_parameters,
)


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.adapter = nn.Sequential(
            nn.Linear(4, 3),
            nn.SiLU(),
            nn.Linear(3, 4),
        )
        for parameter in self.frozen.parameters():
            parameter.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.frozen(value).detach() + self.adapter(value)


def _optimizer(module: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [parameter for _, parameter in trainable_named_parameters(module)],
        lr=1e-2,
        foreach=False,
    )


class TrainingCheckpointTest(unittest.TestCase):
    def test_parameter_update_summary_detects_finite_change(self):
        torch.manual_seed(0)
        module = _TinyModule()
        optimizer = _optimizer(module)
        before = capture_trainable_parameters(module)

        loss = module(torch.ones(2, 4)).square().mean()
        loss.backward()
        optimizer.step()

        summary = parameter_update_summary(before, module)
        self.assertGreater(summary["changed_tensors"], 0)
        self.assertGreater(summary["changed_elements"], 0)
        self.assertGreater(summary["global_l2"], 0.0)
        self.assertEqual(summary["nonfinite_elements"], 0)

    def test_save_and_restore_model_and_optimizer_exactly(self):
        torch.manual_seed(1)
        module = _TinyModule()
        optimizer = _optimizer(module)

        loss = module(torch.randn(3, 4)).abs().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        expected_parameters = capture_trainable_parameters(module)
        expected_state_count = len(optimizer.state)
        self.assertGreater(expected_state_count, 0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = save_delta_checkpoint(
                root,
                module,
                optimizer,
                step=7,
                metadata={"base_checkpoint": "/tmp/base", "scope": "adapter"},
            )
            self.assertEqual(metadata["step"], 7)
            self.assertTrue((root / "trainable.safetensors").is_file())
            self.assertTrue((root / "optimizer.pt").is_file())
            self.assertTrue((root / "metadata.json").is_file())

            with torch.no_grad():
                for _, parameter in trainable_named_parameters(module):
                    parameter.add_(1.0)

            resumed_optimizer = _optimizer(module)
            loaded = load_delta_checkpoint(root, module, resumed_optimizer)
            self.assertEqual(loaded["step"], 7)
            self.assertEqual(len(resumed_optimizer.state), expected_state_count)

            restored_parameters = capture_trainable_parameters(module)
            self.assertEqual(tuple(expected_parameters), tuple(restored_parameters))
            for name in expected_parameters:
                torch.testing.assert_close(
                    restored_parameters[name],
                    expected_parameters[name],
                    rtol=0,
                    atol=0,
                )

            for state in resumed_optimizer.state.values():
                self.assertIn("step", state)
                self.assertIn("exp_avg", state)
                self.assertIn("exp_avg_sq", state)

    def test_strict_restore_rejects_changed_trainable_set(self):
        module = _TinyModule()
        optimizer = _optimizer(module)
        loss = module(torch.ones(1, 4)).sum()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_delta_checkpoint(root, module, optimizer, step=1)
            module.frozen.weight.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "Trainable parameter set"):
                load_delta_checkpoint(root, module)

    def test_no_trainable_parameters_is_rejected(self):
        module = nn.Linear(2, 2)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "no trainable parameters"):
            trainable_named_parameters(module)


if __name__ == "__main__":
    unittest.main()
