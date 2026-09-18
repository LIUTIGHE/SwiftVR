#!/usr/bin/env python3
"""Full-fresh training for the Stage-B1 Moderate TC Decoder v1.

This is an intentionally thin wrapper around
``train_tiny_decoder_moderate_recovery_ddp.py``.  It keeps the Moderate v1
architecture, formal cached-z_SR data path, ReAE teacher, objective, validation,
optimizer grouping, checkpoint format, and DDP loop unchanged, but replaces the
function-preserving R4-Joint widening initialization with a completely fresh
ModerateResizeConvTinyConditionalDecoder initialization.

The source passed through ``--init-decoder`` is retained only because the shared
formal parser/fingerprint contract requires that argument; its weights are NEVER
loaded by this wrapper.  ``run_config.json`` records ``initialization_mode`` and
``source_checkpoint_used=false`` so a fresh run cannot be confused with a warm
recovery run.

Fresh training also changes the six semantic optimizer-group defaults to 5e-5.
Users may still override any group explicitly on the command line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_tiny_decoder_moderate_recovery_ddp as base
from swiftvr.models.tiny_conditional_decoder_moderate_resize_conv import (
    MODERATE_BLOCKS_PER_STAGE,
    MODERATE_CHANNELS,
    MODERATE_CONDITION_CHANNELS,
    MODERATE_INTERNAL_CHANNELS,
    MODERATE_RESIZE_MODE,
    ModerateResizeConvTinyConditionalDecoder,
)


FRESH_TRAINER_ID = "swiftvr_stage_b1_moderate_tc_decoder_full_fresh_ddp_v1"
FRESH_INITIALIZATION_MODE = "full_fresh_random"
FRESH_DEFAULT_LEARNING_RATE = 5e-5
_LR_DESTINATIONS = frozenset(
    {
        "condition_input_learning_rate",
        "early_learning_rate",
        "stage2_learning_rate",
        "transition23_learning_rate",
        "stage3_learning_rate",
        "head_learning_rate",
    }
)


_original_build_parser = base.build_parser
_original_fingerprint = base._fingerprint


def _fresh_build_parser():
    parser = _original_build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest in _LR_DESTINATIONS:
            action.default = FRESH_DEFAULT_LEARNING_RATE
    return parser


def _fresh_load_initial_model(
    init_decoder: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    # Deliberately do not read/load init_decoder.  Construct the target topology
    # directly so every learnable tensor uses its ordinary PyTorch initialization.
    model = ModerateResizeConvTinyConditionalDecoder(
        latent_channels=48,
        condition_channels=MODERATE_CONDITION_CHANNELS,
        channels=MODERATE_CHANNELS,
        blocks_per_stage=MODERATE_BLOCKS_PER_STAGE,
        temporal_factor=4,
        spatial_factor=16,
        patch_size=2,
        frames_to_trim=3,
        block_mode="compact",
        block_internal_channels=MODERATE_INTERNAL_CHANNELS,
        resize_mode=MODERATE_RESIZE_MODE,
    ).to(device=device, dtype=dtype)
    base._assert_target_topology(model)
    report = {
        "mode": FRESH_INITIALIZATION_MODE,
        "source_checkpoint_argument": str(init_decoder.expanduser().resolve()),
        "source_checkpoint_used": False,
        "source_function_exact_at_initialization": False,
        "all_learnable_parameters_fresh": True,
        "condition_channels": MODERATE_CONDITION_CHANNELS,
        "decoder_channels": list(MODERATE_CHANNELS),
        "block_internal_channels": list(MODERATE_INTERNAL_CHANNELS),
        "resize_mode": MODERATE_RESIZE_MODE,
    }
    return model, report


def _fresh_fingerprint(
    args,
    *,
    world_size,
    train_cache,
    val_cache,
    init_decoder,
):
    value = _original_fingerprint(
        args,
        world_size=world_size,
        train_cache=train_cache,
        val_cache=val_cache,
        init_decoder=init_decoder,
    )
    value.update(
        {
            "trainer": FRESH_TRAINER_ID,
            "initialization_mode": FRESH_INITIALIZATION_MODE,
            "source_checkpoint_used": False,
            "fresh_default_learning_rate": FRESH_DEFAULT_LEARNING_RATE,
        }
    )
    return value


def main() -> int:
    # Patch only the initialization/parser/fingerprint hooks consumed by base.main.
    # The already validated warm-recovery source file itself remains untouched.
    base.TRAINER_ID = FRESH_TRAINER_ID
    base.build_parser = _fresh_build_parser
    base._load_initial_model = _fresh_load_initial_model
    base._fingerprint = _fresh_fingerprint
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
