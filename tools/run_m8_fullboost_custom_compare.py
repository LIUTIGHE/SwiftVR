#!/usr/bin/env python3
"""Run custom-input visual comparison for the current M8 Full-Boost candidates.

Only --input is required.  By default this evaluates:

  * Original SwiftVR
  * M8-A 30k + M9-A1 decoder
  * TA Full-Boost step 1500 + M9-A1 decoder
  * TA Full-Boost step 4000 + M9-A1 decoder
  * Stage-A Full-Boost step 3500 + M9-A1 decoder

and then invokes tools/compare_720p3x_outputs.py on the generated PNG sequences.

Optional --basiccnn adds an externally generated restoration video/directory.
Repeat --crop LABEL:x,y,w,h for custom detail strips.

The tool runs methods sequentially on one visible GPU so VRAM usage does not scale
with the number of compared checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


DEFAULTS = {
    "original": Path("checkpoints"),
    "base": Path("checkpoints_prompt_free_no_time"),
    "m8a": Path("outputs/b2b/m8a_d1024_l20_gate200k/checkpoints/step_00030000"),
    "ta1500": Path("outputs/b2b/m8a_fullboost_ta_5k_v1/checkpoints/step_00001500"),
    "ta4000": Path("outputs/b2b/m8a_fullboost_ta_5k_v1/checkpoints/step_00004000"),
    "stagea3500": Path("outputs/b2b/m8a_fullboost_stagea_5k_v1/checkpoints/step_00003500"),
    "decoder": Path(
        "outputs/b2b/m9a1_factorized_m8a30k/checkpoints/"
        "epoch_099_step_00024552/tiny_decoder"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root output directory. Default: "
            "outputs/custom/<input_stem>_m8_fullboost_compare"
        ),
    )
    p.add_argument(
        "--cuda-visible-devices",
        default="7",
        help="CUDA_VISIBLE_DEVICES passed to inference subprocesses. Default: 7.",
    )
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--attention-backend", default="sdpa")
    p.add_argument("--upscale", type=int, default=3)
    p.add_argument("--clip-len", type=int, default=24)
    p.add_argument("--dit-overlap", type=int, default=0)
    p.add_argument(
        "--frame-indices",
        default="0,8,16,24,32",
        help="Frames exported by the comparison tool.",
    )
    p.add_argument("--panel-width", type=int, default=960)
    p.add_argument(
        "--crop",
        action="append",
        default=[],
        help="Optional crop LABEL:x,y,w,h; repeat for multiple crops.",
    )
    p.add_argument(
        "--basiccnn",
        type=Path,
        default=None,
        help="Optional external restoration video/image directory.",
    )
    p.add_argument("--basiccnn-label", default="Tiny CNN")
    p.add_argument("--basiccnn-index-scale", type=int, default=1)
    p.add_argument("--basiccnn-index-offset", type=int, default=0)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a method output directory when it already contains PNGs.",
    )
    p.add_argument(
        "--no-original",
        action="store_true",
        help="Skip Original SwiftVR if only comparing compressed candidates.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and resolved paths without running inference/comparison.",
    )

    p.add_argument("--original-checkpoint", type=Path, default=DEFAULTS["original"])
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULTS["base"])
    p.add_argument("--m8a-checkpoint", type=Path, default=DEFAULTS["m8a"])
    p.add_argument("--ta1500-checkpoint", type=Path, default=DEFAULTS["ta1500"])
    p.add_argument("--ta4000-checkpoint", type=Path, default=DEFAULTS["ta4000"])
    p.add_argument(
        "--stagea3500-checkpoint",
        type=Path,
        default=DEFAULTS["stagea3500"],
    )
    p.add_argument("--decoder-checkpoint", type=Path, default=DEFAULTS["decoder"])
    return p


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _require_path(path: Path, label: str) -> Path:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _has_pngs(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.png"))


def _run(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    print("\n$", " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(
        list(command),
        cwd=str(ROOT),
        env=env,
        check=True,
    )


def _run_original(
    *,
    input_path: Path,
    output_path: Path,
    checkpoint: Path,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    command = [
        sys.executable,
        "scripts/inference.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--checkpoint",
        str(checkpoint),
        "--upscale",
        str(args.upscale),
        "--clip-len",
        str(args.clip_len),
        "--dit-overlap",
        str(args.dit_overlap),
        "--dtype",
        args.dtype,
        "--attention_backend",
        args.attention_backend,
        "--png",
    ]
    _run(command, env=env, dry_run=bool(args.dry_run))


def _run_m8(
    *,
    input_path: Path,
    output_path: Path,
    base_checkpoint: Path,
    transformer_checkpoint: Path,
    decoder_checkpoint: Path,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    command = [
        sys.executable,
        "scripts/inference_custom_components.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--base-checkpoint",
        str(base_checkpoint),
        "--transformer-checkpoint",
        str(transformer_checkpoint),
        "--transformer-type",
        "moe",
        "--decoder-type",
        "m9a1",
        "--decoder-checkpoint",
        str(decoder_checkpoint),
        "--upscale",
        str(args.upscale),
        "--clip-len",
        str(args.clip_len),
        "--dit-overlap",
        str(args.dit_overlap),
        "--dtype",
        args.dtype,
        "--attention-backend",
        args.attention_backend,
        "--png",
    ]
    _run(command, env=env)


def main() -> int:
    args = build_parser().parse_args()
    input_path = _require_path(args.input, "input")

    if args.upscale <= 0 or args.clip_len <= 0 or args.panel_width <= 0:
        raise ValueError("upscale/clip-len/panel-width must be positive")
    if args.dit_overlap < 0:
        raise ValueError("dit-overlap must be non-negative")
    if args.basiccnn_index_scale <= 0 or args.basiccnn_index_offset < 0:
        raise ValueError("BasicCNN index mapping is invalid")

    output_root = (
        _resolve(args.output_dir)
        if args.output_dir is not None
        else _resolve(
            Path("outputs/custom")
            / f"{input_path.stem}_m8_fullboost_compare"
        )
    )
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    original = _require_path(args.original_checkpoint, "Original SwiftVR checkpoint")
    base = _require_path(args.base_checkpoint, "prompt-free/no-time base checkpoint")
    decoder = _require_path(args.decoder_checkpoint, "M9-A1 decoder checkpoint")

    candidates = [
        ("M8-A", _require_path(args.m8a_checkpoint, "M8-A checkpoint"), "m8a"),
        (
            "TA-1500",
            _require_path(args.ta1500_checkpoint, "TA1500 checkpoint"),
            "ta1500",
        ),
        (
            "TA-4000",
            _require_path(args.ta4000_checkpoint, "TA4000 checkpoint"),
            "ta4000",
        ),
        (
            "StageA-3500",
            _require_path(args.stagea3500_checkpoint, "StageA3500 checkpoint"),
            "stagea3500",
        ),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    method_outputs: list[tuple[str, Path]] = []

    if not args.no_original:
        original_output = output_root / "original_swiftvr"
        if args.skip_existing and _has_pngs(original_output):
            print(f"[reuse] Original SwiftVR: {original_output}", flush=True)
        else:
            _run_original(
                input_path=input_path,
                output_path=original_output,
                checkpoint=original,
                args=args,
                env=env,
            )
        method_outputs.append(("Original SwiftVR", original_output))

    for label, checkpoint, dirname in candidates:
        output = output_root / dirname
        if args.skip_existing and _has_pngs(output):
            print(f"[reuse] {label}: {output}", flush=True)
        else:
            _run_m8(
                input_path=input_path,
                output_path=output,
                base_checkpoint=base,
                transformer_checkpoint=checkpoint,
                decoder_checkpoint=decoder,
                args=args,
                env=env,
            )
        method_outputs.append((label, output))

    comparison_dir = output_root / "comparison"
    if (
        not args.dry_run
        and comparison_dir.exists()
        and any(comparison_dir.iterdir())
    ):
        if not args.skip_existing:
            raise FileExistsError(
                f"Comparison output is not empty: {comparison_dir}; "
                "use a new --output-dir or --skip-existing"
            )
        print(f"[reuse] comparison directory already exists: {comparison_dir}")
        return 0

    compare_command = [
        sys.executable,
        "tools/compare_720p3x_outputs.py",
        "--lq",
        str(input_path),
    ]
    for label, output in method_outputs:
        compare_command.extend(["--method", f"{label}={output}"])

    if args.basiccnn is not None:
        basiccnn = _require_path(args.basiccnn, "BasicCNN output")
        compare_command.extend(
            [
                "--basiccnn",
                str(basiccnn),
                "--basiccnn-label",
                args.basiccnn_label,
                "--basiccnn-index-scale",
                str(args.basiccnn_index_scale),
                "--basiccnn-index-offset",
                str(args.basiccnn_index_offset),
            ]
        )

    compare_command.extend(
        [
            "--output-dir",
            str(comparison_dir),
            "--frame-indices",
            args.frame_indices,
            "--panel-width",
            str(args.panel_width),
        ]
    )
    for crop in args.crop:
        compare_command.extend(["--crop", crop])

    _run(compare_command, dry_run=bool(args.dry_run))

    if args.dry_run:
        print("\nDry-run complete; no inference outputs were written.", flush=True)
        return 0

    summary = {
        "kind": "m8_fullboost_custom_input_compare_v1",
        "input": str(input_path),
        "output_root": str(output_root),
        "cuda_visible_devices": str(args.cuda_visible_devices),
        "decoder_checkpoint": str(decoder),
        "methods": [
            {"label": label, "output": str(path)}
            for label, path in method_outputs
        ],
        "basiccnn": None if args.basiccnn is None else str(_resolve(args.basiccnn)),
        "comparison_dir": str(comparison_dir),
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"\nCustom comparison complete: {comparison_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
