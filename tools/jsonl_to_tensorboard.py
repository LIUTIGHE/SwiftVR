#!/usr/bin/env python3
"""Backfill or live-follow SwiftVR JSONL logs into TensorBoard events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swiftvr.training.telemetry import (
    JSONLTensorBoardFollower,
    backfill_tensorboard,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow append-only train/val logs until interrupted.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.follow:
        if args.overwrite:
            raise ValueError("--overwrite is only valid for one-shot backfill")
        follower = JSONLTensorBoardFollower(args.run_dir, args.log_dir)
        print(
            json.dumps(
                {
                    "status": "FOLLOWING",
                    "run_dir": str(args.run_dir.expanduser().resolve()),
                    "log_dir": str(follower.log_dir),
                },
                indent=2,
            ),
            flush=True,
        )
        follower.follow(args.poll_seconds)
        return 0

    result = backfill_tensorboard(
        args.run_dir,
        log_dir=args.log_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
