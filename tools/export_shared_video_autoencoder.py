#!/usr/bin/env python3
"""Export the current ReAE encoder + ReAE-family decoder as one portable codec.

Typical Stage-B1 Slim100 export:

    python tools/export_shared_video_autoencoder.py \
      --base-checkpoint /data1/a/SwiftVR/checkpoints_prompt_free_no_time \
      --slim-decoder /data1/a/SwiftVR/outputs/stage_b1_reae_slim100_teacher_distill/checkpoints/epoch_050_step_00012400/tiny_decoder \
      --output-dir /data1/a/SwiftVR/outputs/shared_video_codec_slim100

Omit ``--slim-decoder`` to export the original full ReAE encoder+decoder pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.models.shared_video_autoencoder import SharedVideoAutoencoder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True,
                   help="Checkpoint root containing reae.safetensors.")
    p.add_argument("--reae-filename", default="reae.safetensors")
    p.add_argument("--slim-decoder", type=Path, default=None,
                   help="Optional Stage-B1 .../tiny_decoder directory.")
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    reae_path = args.base_checkpoint.expanduser().resolve() / args.reae_filename
    model = SharedVideoAutoencoder.from_component_checkpoints(
        reae_path,
        args.slim_decoder,
        device="cpu",
    )
    output = model.save_pretrained(args.output_dir)
    summary = {
        "output_dir": str(output),
        "reae_checkpoint": str(reae_path),
        "slim_decoder_checkpoint": None if args.slim_decoder is None else str(args.slim_decoder.expanduser().resolve()),
        "config": model.config_dict,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
