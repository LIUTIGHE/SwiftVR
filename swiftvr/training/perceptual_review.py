""Perceptual metrics and visual-review helpers for SwiftVR checkpoints.""

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

FULL_REFERENCE_METRICS = frozenset({"lpips", "dists"})
NO_REFERENCE_METRICS = frozenset({"musiq"})
SUPPORTED_METRICS = FULL_REFERENCE_METRICS | NO_REFERENCE_METRICS
_LABEL_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class StudentCheckpointSpec:
    label: str
    path: Path | None
    step: int | None = None


def sanitize_label(value: str) -> str:
    label = _LABEL_PATTERN.sub("_", value.strip()).strip("._-")
    if not label:
        raise ValueError(f"Invalid empty label derived from {value!r}")
    return label


def parse_student_checkpoint(value: str) -> StudentCheckpointSpec:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError(
            "student checkpoint must use LABEL=PATH, for example "
            "step300=/path/to/step_00000300"
        )
    return StudentCheckpointSpec(
        label=sanitize_label(label),
        path=Path(path).expanduser().resolve(),
    )


def parse_csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected at least one comma-separated integer")
    if any(item < 0 for item in result):
        raise ValueError(f"Frame indices must be non-negative, got {result}")
    return result


def parse_metric_names(value: str | Sequence[str]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    names = tuple(
        dict.fromkeys(str(item).strip().lower() for item in raw if str(item).strip())
    )
    if not names:
        raise ValueError("At least one perceptual metric is required")
    unsupported = sorted(set(names) - SUPPORTED_METRICS)
    if unsupported:
        raise ValueError(
            f"Unsupported perceptual metrics {unsupported}; "
            f"supported={sorted(SUPPORTED_METRICS)}"
        )
    return names


def ensure_video_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if value.ndim != 5 or value.shape[2] != 3:
        raise ValueError(
            "Expected RGB video tensor [B,T,3,H,W] or [T,3,H,W], got "
            f"{tuple(value.shape)}"
        )
    return value


def flatten_video_frames(value: torch.Tensor) -> torch.Tensor:
    video = ensure_video_batch(value)
    return video.reshape(-1, *video.shape[2:])


def restore_trainable_parameters(
    module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    current = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(state):
        missing = sorted(set(state) - set(current))
        unexpected = sorted(set(current) - set(state))
        raise ValueError(
            "Trainable parameter set changed while restoring the base state: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    with torch.no_grad():
        for name, tensor in state.items():
            parameter = current[name]
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: model={tuple(parameter.shape)}, "
                    f"state={tuple(tensor.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def _normalize_metric_output(output: object, batch_size: int) -> list[float]:
    if isinstance(output, (float, int)) and not isinstance(output, bool):
        values = torch.tensor([float(output)])
    elif isinstance(output, torch.Tensor):
        values = output.detach().float().reshape(-1).cpu()
    else:
        try:
            values = torch.as_tensor(output, dtype=torch.float32).reshape(-1).cpu()
        except Exception as exc:
            raise TypeError(f"Unsupported IQA metric output type: {type(output)!r}") from exc
    if values.numel() == batch_size:
        return [float(item) for item in values.tolist()]
    if batch_size == 1 and values.numel() == 1:
        return [float(values.item())]
    raise ValueError(
        f"IQA metric returned {values.numel()} values for batch_size={batch_size}; "
        "use --metric-batch-size 1 for metrics that only return a batch average"
    )


class IQAMetricSuite:
    """Lazy pyiqa wrapper with explicit FR/NR semantics and framewise outputs."""

    def __init__(
        self,
        metric_names: Sequence[str],
        *,
        device: torch.device | str,
        batch_size: int = 1,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.metric_names = parse_metric_names(metric_names)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("metric batch size must be positive")
        if cache_dir is not None:
            os.environ.setdefault(
                "TORCH_HOME", str(Path(cache_dir).expanduser().resolve())
            )
        try:
            import pyiqa
        except Exception as exc:
            raise RuntimeError(
                "Perceptual review requires pyiqa. Install "
                "requirements-perceptual.txt before evaluation."
            ) from exc
        self.pyiqa_version = str(getattr(pyiqa, "__version__", "unknown"))
        self.metrics: dict[str, object] = {}
        self.lower_better: dict[str, bool] = {}
        for name in self.metric_names:
            metric = pyiqa.create_metric(name, device=self.device, as_loss=False)
            metric.eval()
            self.metrics[name] = metric
            self.lower_better[name] = bool(getattr(metric, "lower_better", False))

    @torch.inference_mode()
    def _score_chunks(
        self,
        name: str,
        distorted: torch.Tensor,
        reference: torch.Tensor | None,
    ) -> list[float]:
        distorted = flatten_video_frames(distorted).float().clamp(0.0, 1.0)
        reference_frames = None
        if reference is not None:
            reference_frames = flatten_video_frames(reference).float().clamp(0.0, 1.0)
            if reference_frames.shape != distorted.shape:
                raise ValueError(
                    f"IQA pair shape mismatch: {tuple(distorted.shape)} vs "
                    f"{tuple(reference_frames.shape)}"
                )
        metric = self.metrics[name]
        scores: list[float] = []
        for start in range(0, int(distorted.shape[0]), self.batch_size):
            end = min(start + self.batch_size, int(distorted.shape[0]))
            x = distorted[start:end].to(self.device, non_blocking=True)
            if name in FULL_REFERENCE_METRICS:
                if reference_frames is None:
                    raise ValueError(f"Full-reference metric {name!r} requires a target")
                y = reference_frames[start:end].to(self.device, non_blocking=True)
                output = metric(x, y)
            else:
                output = metric(x)
            scores.extend(_normalize_metric_output(output, end - start))
        return scores

    def score_metric(
        self,
        name: str,
        distorted: torch.Tensor,
        *,
        reference: torch.Tensor | None = None,
    ) -> dict[str, object]:
        if name not in self.metrics:
            raise KeyError(f"Metric {name!r} was not initialized")
        scores = self._score_chunks(name, distorted, reference)
        return {
            "mean": float(sum(scores) / len(scores)),
            "frames": scores,
            "lower_better": self.lower_better[name],
        }

    def score_video(
        self,
        distorted: torch.Tensor,
        *,
        reference: torch.Tensor | None = None,
    ) -> dict[str, dict[str, object]]:
        return {
            name: self.score_metric(name, distorted, reference=reference)
            for name in self.metric_names
        }

    def package_metadata(self) -> dict[str, object]:
        return {
            "backend": "pyiqa",
            "pyiqa_version": self.pyiqa_version,
            "metrics": {
                name: {
                    "kind": (
                        "full_reference"
                        if name in FULL_REFERENCE_METRICS
                        else "no_reference"
                    ),
                    "lower_better": self.lower_better[name],
                }
                for name in self.metric_names
            },
        }


def tensor_frame_to_uint8(frame: torch.Tensor) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f"Expected RGB frame [3,H,W], got {tuple(frame.shape)}")
    return (
        frame.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def _labelled_panel(
    frame: torch.Tensor,
    label: str,
    *,
    label_height: int = 28,
) -> Image.Image:
    array = tensor_frame_to_uint8(frame)
    height, width = array.shape[:2]
    panel = Image.new("RGB", (width, height + label_height), "white")
    panel.paste(Image.fromarray(array, mode="RGB"), (0, label_height))
    ImageDraw.Draw(panel).text((6, 7), label, fill="black")
    return panel


def make_comparison_frame(
    frames: Mapping[str, torch.Tensor],
    *,
    gap: int = 4,
    label_height: int = 28,
) -> Image.Image:
    if not frames:
        raise ValueError("At least one comparison frame is required")
    panels = [
        _labelled_panel(frame, label, label_height=label_height)
        for label, frame in frames.items()
    ]
    panel_height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, panel_height), "white")
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width + gap
    return canvas


def make_difference_frame(
    candidates: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    *,
    scale: float = 4.0,
    gap: int = 4,
    label_height: int = 28,
) -> Image.Image:
    differences = {
        f"|{label}-GT| x{scale:g}": (
            (frame.float() - target.float()).abs().mul(scale).clamp(0, 1)
        )
        for label, frame in candidates.items()
    }
    return make_comparison_frame(
        differences,
        gap=gap,
        label_height=label_height,
    )


def write_json_atomic(path: str | Path, value: object) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)


def write_jsonl_atomic(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(output)


def write_summary_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    temporary.replace(output)


def finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def write_tensorboard_curves(
    log_dir: str | Path,
    summary_rows: Sequence[Mapping[str, object]],
    *,
    reference_label: str = "conditional_reference",
    lq_label: str = "lq_bicubic",
) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        raise RuntimeError("TensorBoard is required for perceptual curves") from exc

    by_label = {str(row.get("label")): row for row in summary_rows}
    students = [row for row in summary_rows if isinstance(row.get("step"), int)]
    students.sort(key=lambda row: int(row["step"]))
    writer = SummaryWriter(log_dir=str(Path(log_dir).expanduser().resolve()))
    try:
        for row in students:
            step = int(row["step"])
            for key, raw in row.items():
                value = finite_or_none(raw)
                if value is None or key == "step":
                    continue
                if key.startswith("gt_"):
                    tag = "review/student_gt/" + key.removeprefix("gt_")
                elif key.startswith("reference_"):
                    tag = (
                        "review/student_reference/"
                        + key.removeprefix("reference_")
                    )
                elif key.startswith("nr_"):
                    tag = "review/no_reference/" + key.removeprefix("nr_")
                else:
                    continue
                writer.add_scalar(tag, value, step)

            for baseline_name, baseline_label in (
                ("conditional_reference", reference_label),
                ("lq_bicubic", lq_label),
            ):
                baseline = by_label.get(baseline_label)
                if baseline is None:
                    continue
                for key, raw in baseline.items():
                    value = finite_or_none(raw)
                    if value is None:
                        continue
                    if key.startswith("gt_") or key.startswith("nr_"):
                        writer.add_scalar(
                            f"review/baseline_{baseline_name}/{key}",
                            value,
                            step,
                        )
        writer.flush()
    finally:
        writer.close()


def build_html_report(
    output_path: str | Path,
    *,
    title: str,
    summary_rows: Sequence[Mapping[str, object]],
    sample_rows: Sequence[Mapping[str, object]],
    metric_metadata: Mapping[str, object],
) -> None:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in summary_rows:
        for key in row:
            if key not in headers:
                headers.append(str(key))

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;}"
        "table{border-collapse:collapse;margin-bottom:24px;}"
        "th,td{border:1px solid #ccc;padding:6px 9px;text-align:right;}"
        "th:first-child,td:first-child{text-align:left;}"
        "video,img{max-width:100%;height:auto;}"
        ".sample{margin:36px 0;padding-top:12px;border-top:2px solid #ddd;}"
        "code{background:#f5f5f5;padding:2px 4px;}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<p>Metric metadata: <code>"
        + html.escape(json.dumps(metric_metadata, sort_keys=True))
        + "</code></p>",
        "<h2>Aggregate metrics</h2><table><thead><tr>",
    ]
    parts.extend(f"<th>{html.escape(header)}</th>" for header in headers)
    parts.append("</tr></thead><tbody>")
    for row in summary_rows:
        parts.append("<tr>")
        for header in headers:
            value = row.get(header, "")
            text = f"{value:.6f}" if isinstance(value, float) else str(value)
            parts.append(f"<td>{html.escape(text)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    for sample in sample_rows:
        name = str(sample.get("record_uid", sample.get("key", "sample")))
        parts.append(f"<section class='sample'><h2>{html.escape(name)}</h2>")
        video = sample.get("comparison_video")
        if isinstance(video, str):
            parts.append(
                f"<video controls loop muted src='{html.escape(video)}'></video>"
            )
        diff_video = sample.get("difference_video")
        if isinstance(diff_video, str):
            parts.append(
                "<h3>Amplified absolute difference to GT</h3>"
                f"<video controls loop muted src='{html.escape(diff_video)}'></video>"
            )
        images = sample.get("comparison_images")
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
            parts.append("<h3>Selected frames</h3>")
            for image in images:
                parts.append(f"<img src='{html.escape(str(image))}'>")
        parts.append("</section>")
    parts.append("</body></html>")
    output.write_text("".join(parts), encoding="utf-8")
