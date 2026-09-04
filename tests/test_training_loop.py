"""CPU tests for SwiftVR's minimal resumable training-loop utilities."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from swiftvr.training import (
    TrainingCursor,
    build_fp32_adamw,
    capture_rng_state,
    cast_trainable_parameters,
    load_delta_checkpoint,
    load_trainer_state,
    resolve_resume_checkpoint,
    restore_rng_state,
    save_delta_checkpoint,
    save_trainer_state,
    seed_everything,
    skip_batches,
    write_latest_checkpoint,
)


class _RandomizedDataset(Dataset):
    def __len__(self) -> int:
        return 7

    def __getitem__(self, index: int) -> torch.Tensor:
        noise = random.random() + float(np.random.rand()) + float(torch.rand(()))
        return torch.tensor([float(index), noise], dtype=torch.float32)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[0.25, -0.5]]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


def _loader(epoch: int, seed: int = 17) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(
        _RandomizedDataset(),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )


def _train_to_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cursor: TrainingCursor,
    target_step: int,
    *,
    pending_rng_state=None,
) -> tuple[TrainingCursor, list[float]]:
    losses: list[float] = []
    restore_pending = pending_rng_state is not None
    while cursor.global_step < target_step:
        loader = _loader(cursor.epoch)
        iterator = iter(loader)
        skip_batches(iterator, cursor.batch_in_epoch)
        if restore_pending:
            restore_rng_state(pending_rng_state)
            restore_pending = False
        while cursor.global_step < target_step:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            cursor = cursor.advance(batches_per_epoch=len(loader))
            if cursor.batch_in_epoch == 0:
                break
    return cursor, losses


def _assert_optimizer_equal(
    testcase: unittest.TestCase,
    left: torch.optim.Optimizer,
    right: torch.optim.Optimizer,
) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    testcase.assertEqual(left_state["param_groups"], right_state["param_groups"])
    testcase.assertEqual(set(left_state["state"]), set(right_state["state"]))
    for key in left_state["state"]:
        testcase.assertEqual(
            set(left_state["state"][key]), set(right_state["state"][key])
        )
        for state_key, left_value in left_state["state"][key].items():
            right_value = right_state["state"][key][state_key]
            if isinstance(left_value, torch.Tensor):
                torch.testing.assert_close(left_value, right_value, rtol=0, atol=0)
            else:
                testcase.assertEqual(left_value, right_value)


class TrainingLoopTest(unittest.TestCase):
    def test_cursor_rolls_to_next_epoch(self):
        cursor = TrainingCursor(global_step=4, epoch=2, batch_in_epoch=2)
        next_cursor = cursor.advance(batches_per_epoch=3)
        self.assertEqual(next_cursor, TrainingCursor(5, 3, 0))

    def test_rng_roundtrip_is_exact(self):
        seed_everything(123)
        state = capture_rng_state()
        expected = (
            random.random(),
            float(np.random.rand()),
            torch.rand(4),
        )
        for _ in range(10):
            random.random()
            np.random.rand()
            torch.rand(4)
        restore_rng_state(state)
        actual = (
            random.random(),
            float(np.random.rand()),
            torch.rand(4),
        )
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_trainer_state_and_latest_pointer(self):
        seed_everything(9)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint = run_dir / "checkpoints" / "step_00000003"
            cursor = TrainingCursor(3, 0, 3)
            save_trainer_state(
                checkpoint,
                cursor=cursor,
                config={"crop_size": 32},
            )
            loaded = load_trainer_state(checkpoint)
            self.assertEqual(loaded["cursor"], cursor)
            self.assertEqual(loaded["config"], {"crop_size": 32})
            write_latest_checkpoint(run_dir, checkpoint)
            self.assertEqual(
                resolve_resume_checkpoint("latest", run_dir=run_dir),
                checkpoint.resolve(),
            )

    def test_fp32_adamw_requires_fp32_trainable_parameters(self):
        module = nn.Linear(2, 2).half()
        with self.assertRaisesRegex(RuntimeError, "must be FP32"):
            build_fp32_adamw(module, learning_rate=1e-3)
        summary = cast_trainable_parameters(module, dtype=torch.float32)
        self.assertEqual(summary["target_dtype"], "float32")
        optimizer = build_fp32_adamw(module, learning_rate=1e-3)
        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_mid_epoch_resume_matches_uninterrupted_training(self):
        seed_everything(2026)
        uninterrupted = _TinyModel()
        uninterrupted_optimizer = build_fp32_adamw(
            uninterrupted, learning_rate=1e-3
        )
        uninterrupted_cursor, uninterrupted_losses = _train_to_step(
            uninterrupted,
            uninterrupted_optimizer,
            TrainingCursor(),
            6,
        )
        self.assertEqual(uninterrupted_cursor.global_step, 6)

        seed_everything(2026)
        interrupted = _TinyModel()
        interrupted_optimizer = build_fp32_adamw(
            interrupted, learning_rate=1e-3
        )
        interrupted_cursor, first_losses = _train_to_step(
            interrupted,
            interrupted_optimizer,
            TrainingCursor(),
            3,
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "step_00000003"
            save_delta_checkpoint(
                checkpoint,
                interrupted,
                interrupted_optimizer,
                step=interrupted_cursor.global_step,
            )
            save_trainer_state(
                checkpoint,
                cursor=interrupted_cursor,
                config={"seed": 17},
                rng_state=capture_rng_state(),
            )

            resumed = _TinyModel()
            resumed_optimizer = build_fp32_adamw(resumed, learning_rate=1e-3)
            metadata = load_delta_checkpoint(
                checkpoint,
                resumed,
                resumed_optimizer,
            )
            trainer_state = load_trainer_state(checkpoint)
            self.assertEqual(metadata["step"], 3)
            resumed_cursor, resumed_losses = _train_to_step(
                resumed,
                resumed_optimizer,
                trainer_state["cursor"],
                6,
                pending_rng_state=trainer_state["rng_state"],
            )

        self.assertEqual(resumed_cursor, uninterrupted_cursor)
        self.assertEqual(len(first_losses + resumed_losses), 6)
        torch.testing.assert_close(
            torch.tensor(first_losses + resumed_losses),
            torch.tensor(uninterrupted_losses),
            rtol=0,
            atol=0,
        )
        for expected, actual in zip(
            uninterrupted.parameters(), resumed.parameters()
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        _assert_optimizer_equal(
            self, uninterrupted_optimizer, resumed_optimizer
        )


if __name__ == "__main__":
    unittest.main()
