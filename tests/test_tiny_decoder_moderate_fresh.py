from __future__ import annotations

import unittest
from pathlib import Path

import torch

from swiftvr.models.tiny_conditional_decoder_moderate_resize_conv import (
    MODERATE_CHANNELS,
    MODERATE_CONDITION_CHANNELS,
    MODERATE_INTERNAL_CHANNELS,
)
from tools import train_tiny_decoder_moderate_fresh_ddp as fresh


class ModerateDecoderFreshTest(unittest.TestCase):
    def test_fresh_loader_builds_target_topology_without_source_weights(self) -> None:
        torch.manual_seed(123)
        first, report = fresh._fresh_load_initial_model(
            Path("/definitely/not/a/checkpoint"),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertFalse(report["source_checkpoint_used"])
        self.assertTrue(report["all_learnable_parameters_fresh"])
        self.assertEqual(first.condition_channels, MODERATE_CONDITION_CHANNELS)
        self.assertEqual(tuple(first.channels), MODERATE_CHANNELS)
        self.assertEqual(tuple(first.block_internal_channels), MODERATE_INTERNAL_CHANNELS)

        # A different RNG seed must produce a genuinely different fresh model;
        # this guards against accidentally loading/copying the warm source path.
        torch.manual_seed(456)
        second, _ = fresh._fresh_load_initial_model(
            Path("/another/nonexistent/checkpoint"),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        first_weight = next(first.parameters()).detach()
        second_weight = next(second.parameters()).detach()
        self.assertFalse(torch.equal(first_weight, second_weight))

    def test_fresh_parser_defaults_all_semantic_groups_to_5e5(self) -> None:
        parser = fresh._fresh_build_parser()
        values = {
            action.dest: action.default
            for action in parser._actions
            if action.dest in fresh._LR_DESTINATIONS
        }
        self.assertEqual(set(values), set(fresh._LR_DESTINATIONS))
        for value in values.values():
            self.assertEqual(float(value), fresh.FRESH_DEFAULT_LEARNING_RATE)


if __name__ == "__main__":
    unittest.main()
