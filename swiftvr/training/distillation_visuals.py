"""Fixed validation visual exports for teacher-distillation runs."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from .perceptual_review import make_comparison_frame


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_REQUIRED_VIDEO_KEYS = ("lq_input", "target", "teacher_prediction", "student_prediction")


def _safe_name(value: object, fallback: str) -> str:
    result = _SAFE_NAME.sub("_", str(value)).strip("._-")
    return result or fallback


def _as_video(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"visual sample {name!r} must be a tensor")
    video = value.detach().float().cpu()
    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError(
            f"visual sample {name!r} must use [T,3,H,W], got {tuple(video.shape)}"
        )
    return video.clamp(0.0, 1.0)


def _pil_to_chw(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def _difference_panels(
    student: torch.Tensor,
    teacher: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: float,
) -> Image.Image:
    return make_comparison_frame(
        OrderedDict(
            (
                (
                    f"|Student-Teacher| x{scale:g}",
                    (student - teacher).abs().mul(scale).clamp(0.0, 1.0),
                ),
                (
                    f"|Student-GT| x{scale:g}",
                    (student - target).abs().mul(scale).clamp(0.0, 1.0),
                ),
                (
                    f"|Teacher-GT| x{scale:g}",
                    (teacher - target).abs().mul(scale).clamp(0.0, 1.0),
                ),
            )
        )
    )


def _write_video(path: Path, frames: Sequence[Image.Image], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path),
        fps=float(fps),
        codec="libx264",
        macro_block_size=1,
        quality=8,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))


def export_validation_visuals(
    samples: Sequence[Mapping[str, object]],
    *,
    output_root: str | Path,
    step: int,
    frame_indices: Sequence[int] = (0, 6, 12),
    fps: float = 8.0,
    difference_scale: float = 4.0,
    writer=None,
    write_videos: bool = True,
) -> dict[str, object]:
    """Write fixed PNG/MP4 comparisons and optional TensorBoard images.

    Each sample must contain ``lq_input``, ``target``, ``teacher_prediction`` and
    ``student_prediction`` tensors in ``[T,3,H,W]`` layout. Files are written
    under ``validation_visuals/step_XXXXXXXX`` and never depend on checkpoint
    retention.
    """

    if int(step) < 0:
        raise ValueError("step must be non-negative")
    if fps <= 0 or difference_scale <= 0:
        raise ValueError("fps and difference_scale must be positive")
    requested = tuple(dict.fromkeys(int(index) for index in frame_indices))
    if not requested or any(index < 0 for index in requested):
        raise ValueError("frame_indices must contain non-negative integers")

    root = Path(output_root).expanduser().resolve()
    step_dir = root / "validation_visuals" / f"step_{int(step):08d}"
    step_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "step": int(step),
        "samples": [],
        "video_errors": [],
    }

    for sample_index, sample in enumerate(samples):
        videos = {
            key: _as_video(sample.get(key), key)
            for key in _REQUIRED_VIDEO_KEYS
        }
        shapes = {tuple(video.shape) for video in videos.values()}
        if len(shapes) != 1:
            raise ValueError(f"visual sample videos have mismatched shapes: {shapes}")
        frames = int(videos["target"].shape[0])
        valid_indices = tuple(index for index in requested if index < frames)
        if not valid_indices:
            raise ValueError(
                f"No requested visual frame lies inside a {frames}-frame clip"
            )

        uid = _safe_name(sample.get("record_uid"), f"sample_{sample_index:03d}")
        sample_dir = step_dir / f"sample_{sample_index:03d}_{uid}"
        sample_dir.mkdir(parents=True)
        comparison_video: list[Image.Image] = []
        difference_video: list[Image.Image] = []

        for frame_index in range(frames):
            comparison = make_comparison_frame(
                OrderedDict(
                    (
                        ("LQ bicubic", videos["lq_input"][frame_index]),
                        ("GT", videos["target"][frame_index]),
                        ("Conditional teacher", videos["teacher_prediction"][frame_index]),
                        ("Prompt-free student", videos["student_prediction"][frame_index]),
                    )
                )
            )
            difference = _difference_panels(
                videos["student_prediction"][frame_index],
                videos["teacher_prediction"][frame_index],
                videos["target"][frame_index],
                scale=float(difference_scale),
            )
            comparison_video.append(comparison)
            difference_video.append(difference)
            if frame_index in valid_indices:
                comparison.save(sample_dir / f"comparison_frame_{frame_index:03d}.png")
                difference.save(sample_dir / f"difference_frame_{frame_index:03d}.png")
                if writer is not None:
                    writer.add_image(
                        f"validation_visuals/{uid}/comparison_frame_{frame_index:03d}",
                        _pil_to_chw(comparison),
                        int(step),
                    )
                    writer.add_image(
                        f"validation_visuals/{uid}/difference_frame_{frame_index:03d}",
                        _pil_to_chw(difference),
                        int(step),
                    )

        if write_videos:
            for filename, content in (
                ("comparison.mp4", comparison_video),
                ("differences.mp4", difference_video),
            ):
                try:
                    _write_video(sample_dir / filename, content, fps)
                except Exception as exc:  # PNG outputs remain usable without ffmpeg.
                    report["video_errors"].append(
                        {
                            "sample": uid,
                            "file": filename,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

        report["samples"].append(
            {
                "record_uid": str(sample.get("record_uid", uid)),
                "directory": str(sample_dir.relative_to(root)),
                "frames": frames,
                "selected_frames": list(valid_indices),
            }
        )

    metadata = step_dir / "metadata.json"
    metadata.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if writer is not None:
        writer.flush()
    return report
