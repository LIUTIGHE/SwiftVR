#!/usr/bin/env python3
"""Run a custom-input comparison for the current SwiftVR compression node.

The comparison intentionally contains only four visual sources:

  * LQ (added by compare_720p3x_outputs.py)
  * Original SwiftVR
  * Current experiment node (default: TA Full-Boost step 1500 + M9-A1)
  * Tiny CNN

The current compressed node is always assembled through
scripts/inference_custom_components.py so Transformer and decoder lineage remain
explicit. Tiny-CNN video input is normalized to 30 FPS by default, matching the
historical custom-input comparison workflow.

Full-frame montage is only an overview. The wrapper also adds automatic
model-agnostic high-detail crops from the LQ input, plus any manual --crop
regions supplied by the user.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
for search_root in (ROOT, TOOLS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from compare_720p3x_outputs import FrameSource


DEFAULTS = {
    "original": Path("checkpoints"),
    "base": Path("checkpoints_prompt_free_no_time"),
    "current": Path(
        "outputs/b2b/m8a_fullboost_ta_5k_v1/checkpoints/step_00001500"
    ),
    "decoder": Path(
        "outputs/b2b/m9a1_factorized_m8a30k/checkpoints/"
        "epoch_099_step_00024552/tiny_decoder"
    ),
}

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--basiccnn",
        type=Path,
        required=True,
        help="Tiny-CNN restoration video/image directory.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root output directory. Default: "
            "outputs/custom/<input_stem>_current_compare"
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
        "--auto-crops",
        type=int,
        default=3,
        help=(
            "Automatically select this many non-overlapping high-detail crops "
            "from LQ. Default: 3; set 0 to disable."
        ),
    )
    p.add_argument(
        "--auto-crop-size",
        type=int,
        default=960,
        help="Square crop size in target/SR coordinates. Default: 960.",
    )
    p.add_argument(
        "--auto-crop-iou-threshold",
        type=float,
        default=0.10,
    )

    p.add_argument("--basiccnn-label", default="Tiny CNN")
    p.add_argument("--basiccnn-index-scale", type=int, default=1)
    p.add_argument("--basiccnn-index-offset", type=int, default=0)
    p.add_argument(
        "--basiccnn-fps",
        type=float,
        default=30.0,
        help="Normalize Tiny-CNN video to this FPS before comparison. Default: 30.",
    )
    p.add_argument(
        "--no-basiccnn-fps-normalize",
        action="store_true",
        help="Use Tiny-CNN video directly without FPS normalization.",
    )

    p.add_argument(
        "--current-label",
        default="Current FullBoost",
        help="Display label for the current experiment node.",
    )
    p.add_argument(
        "--current-checkpoint",
        type=Path,
        default=DEFAULTS["current"],
        help="Current MoE Transformer checkpoint; default is TA FullBoost step1500.",
    )
    p.add_argument("--original-checkpoint", type=Path, default=DEFAULTS["original"])
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULTS["base"])
    p.add_argument("--decoder-checkpoint", type=Path, default=DEFAULTS["decoder"])

    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse completed Original/Current PNG inference outputs.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands without running them.",
    )
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



def _parse_frame_indices(value: str) -> list[int]:
    values: list[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        index = int(raw)
        if index < 0:
            raise ValueError("frame indices must be non-negative")
        if index not in values:
            values.append(index)
    return values


def _box_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection <= 0:
        return 0.0
    union = aw * ah + bw * bh - intersection
    return float(intersection / max(union, 1))


def _window_candidates(
    energy: np.ndarray,
    *,
    crop_w: int,
    crop_h: int,
) -> list[tuple[float, int, int]]:
    height, width = energy.shape
    crop_w = min(int(crop_w), width)
    crop_h = min(int(crop_h), height)
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("auto crop dimensions must be positive")

    integral = np.pad(
        energy.astype(np.float64, copy=False).cumsum(0).cumsum(1),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    stride_x = max(8, crop_w // 8)
    stride_y = max(8, crop_h // 8)
    xs = list(range(0, max(width - crop_w, 0) + 1, stride_x))
    ys = list(range(0, max(height - crop_h, 0) + 1, stride_y))
    if xs[-1] != width - crop_w:
        xs.append(width - crop_w)
    if ys[-1] != height - crop_h:
        ys.append(height - crop_h)

    candidates: list[tuple[float, int, int]] = []
    area = float(crop_w * crop_h)
    for y in ys:
        y2 = y + crop_h
        for x in xs:
            x2 = x + crop_w
            total = (
                integral[y2, x2]
                - integral[y, x2]
                - integral[y2, x]
                + integral[y, x]
            )
            candidates.append((float(total / area), x, y))
    candidates.sort(reverse=True)
    return candidates


def _auto_detail_crops(
    input_path: Path,
    *,
    upscale: int,
    frame_indices: Sequence[int],
    count: int,
    target_crop_size: int,
    iou_threshold: float,
) -> list[str]:
    if count <= 0:
        return []
    source = FrameSource(input_path)
    valid = [index for index in frame_indices if index < len(source)]
    if not valid:
        valid = [0]
    # Keep the auto-selector cheap and deterministic on long videos.
    valid = valid[: min(len(valid), 5)]

    energy_sum: np.ndarray | None = None
    for index in valid:
        frame = source.frame(index).astype(np.float32) / 255.0
        gray = (
            0.299 * frame[..., 0]
            + 0.587 * frame[..., 1]
            + 0.114 * frame[..., 2]
        )
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, :-1] = np.abs(gray[:, 1:] - gray[:, :-1])
        gy[:-1, :] = np.abs(gray[1:, :] - gray[:-1, :])
        energy = gx + gy
        energy_sum = energy if energy_sum is None else energy_sum + energy
    assert energy_sum is not None
    energy_mean = energy_sum / float(len(valid))

    target_w = int(source.width) * int(upscale)
    target_h = int(source.height) * int(upscale)
    crop_target = min(int(target_crop_size), target_w, target_h)
    crop_lq_w = max(1, int(round(crop_target / float(upscale))))
    crop_lq_h = max(1, int(round(crop_target / float(upscale))))

    candidates = _window_candidates(
        energy_mean,
        crop_w=crop_lq_w,
        crop_h=crop_lq_h,
    )
    selected: list[tuple[int, int, int, int]] = []
    for _score, x_lq, y_lq in candidates:
        x = min(int(x_lq * upscale), target_w - crop_target)
        y = min(int(y_lq * upscale), target_h - crop_target)
        box = (x, y, crop_target, crop_target)
        if any(_box_iou(box, previous) > iou_threshold for previous in selected):
            continue
        selected.append(box)
        if len(selected) >= count:
            break

    return [
        f"auto_detail_{index + 1}:{x},{y},{w},{h}"
        for index, (x, y, w, h) in enumerate(selected)
    ]


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
    _run(command, env=env, dry_run=bool(args.dry_run))



def _prepare_basiccnn(
    *,
    source: Path,
    output_root: Path,
    fps: float,
    normalize: bool,
    skip_existing: bool,
    dry_run: bool,
) -> Path:
    source = _require_path(source, "Tiny CNN output")
    if source.is_dir() or not normalize:
        return source
    if fps <= 0:
        raise ValueError("basiccnn-fps must be positive")

    output = output_root / f"tinycnn_fps{fps:g}.mp4"
    if skip_existing and output.is_file() and output.stat().st_size > 0:
        print(f"[reuse] Tiny CNN normalized video: {output}", flush=True)
        return output

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"setpts=N/({fps:g}*TB)",
        "-r",
        f"{fps:g}",
        "-vsync",
        "cfr",
        str(output),
    ]
    _run(command, dry_run=dry_run)
    return output

def main() -> int:
    args = build_parser().parse_args()
    input_path = _require_path(args.input, "input")

    if args.upscale <= 0 or args.clip_len <= 0 or args.panel_width <= 0:
        raise ValueError("upscale/clip-len/panel-width must be positive")
    if args.dit_overlap < 0:
        raise ValueError("dit-overlap must be non-negative")
    if args.basiccnn_index_scale <= 0 or args.basiccnn_index_offset < 0:
        raise ValueError("BasicCNN index mapping is invalid")
    if args.auto_crops < 0 or args.auto_crop_size <= 0:
        raise ValueError("auto-crops must be non-negative and auto-crop-size positive")
    if not 0.0 <= args.auto_crop_iou_threshold < 1.0:
        raise ValueError("auto-crop-iou-threshold must be in [0,1)")

    output_root = (
        _resolve(args.output_dir)
        if args.output_dir is not None
        else _resolve(
            Path("outputs/custom") / f"{input_path.stem}_current_compare"
        )
    )
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    original = _require_path(
        args.original_checkpoint,
        "Original SwiftVR checkpoint",
    )
    base = _require_path(
        args.base_checkpoint,
        "prompt-free/no-time base checkpoint",
    )
    current = _require_path(
        args.current_checkpoint,
        "current experiment checkpoint",
    )
    decoder = _require_path(
        args.decoder_checkpoint,
        "M9-A1 decoder checkpoint",
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

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

    current_output = output_root / "current"
    if args.skip_existing and _has_pngs(current_output):
        print(f"[reuse] {args.current_label}: {current_output}", flush=True)
    else:
        _run_m8(
            input_path=input_path,
            output_path=current_output,
            base_checkpoint=base,
            transformer_checkpoint=current,
            decoder_checkpoint=decoder,
            args=args,
            env=env,
        )

    basiccnn = _prepare_basiccnn(
        source=args.basiccnn,
        output_root=output_root,
        fps=float(args.basiccnn_fps),
        normalize=not bool(args.no_basiccnn_fps_normalize),
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )

    comparison_dir = output_root / "comparison"
    if (
        not args.dry_run
        and comparison_dir.exists()
        and any(comparison_dir.iterdir())
    ):
        print(
            f"[rebuild] removing stale comparison output: {comparison_dir}",
            flush=True,
        )
        shutil.rmtree(comparison_dir)

    auto_crops = _auto_detail_crops(
        input_path,
        upscale=args.upscale,
        frame_indices=_parse_frame_indices(args.frame_indices),
        count=args.auto_crops,
        target_crop_size=args.auto_crop_size,
        iou_threshold=args.auto_crop_iou_threshold,
    )
    if auto_crops:
        print("\n[auto detail crops]", flush=True)
        for crop in auto_crops:
            print(f"  {crop}", flush=True)

    compare_command = [
        sys.executable,
        "tools/compare_720p3x_outputs.py",
        "--lq",
        str(input_path),
        "--method",
        f"Original SwiftVR={original_output}",
        "--method",
        f"{args.current_label}={current_output}",
        "--basiccnn",
        str(basiccnn),
        "--basiccnn-label",
        args.basiccnn_label,
        "--basiccnn-index-scale",
        str(args.basiccnn_index_scale),
        "--basiccnn-index-offset",
        str(args.basiccnn_index_offset),
        "--output-dir",
        str(comparison_dir),
        "--frame-indices",
        args.frame_indices,
        "--panel-width",
        str(args.panel_width),
    ]
    for crop in [*auto_crops, *args.crop]:
        compare_command.extend(["--crop", crop])

    _run(compare_command, dry_run=bool(args.dry_run))

    if args.dry_run:
        print("\nDry-run complete; no inference outputs were written.", flush=True)
        return 0

    summary = {
        "kind": "current_node_custom_input_compare_v2",
        "input": str(input_path),
        "output_root": str(output_root),
        "cuda_visible_devices": str(args.cuda_visible_devices),
        "original_checkpoint": str(original),
        "current_label": args.current_label,
        "current_checkpoint": str(current),
        "decoder_checkpoint": str(decoder),
        "basiccnn_source": str(_resolve(args.basiccnn)),
        "basiccnn_compared": str(basiccnn),
        "comparison_dir": str(comparison_dir),
        "auto_crops": auto_crops,
        "manual_crops": list(args.crop),
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"\nCustom comparison complete: {comparison_dir}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
