"""TensorBoard telemetry helpers backed by SwiftVR JSONL logs.

JSONL remains the source of truth for resume and audit. TensorBoard is an
optional visualization layer that can be generated after a run or followed live
without changing trainer state or the exact-resume fingerprint.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Iterable, Mapping


TRAIN_SCALAR_TAGS = {
    "loss": "train/loss",
    "pixel_l1": "train/pixel_l1",
    "temporal_mse": "train/temporal_mse",
    "gradient_norm": "train/gradient_norm",
    "learning_rate": "train/learning_rate",
    "grad_scaler_scale": "train/grad_scaler_scale",
    "step_seconds": "train/step_seconds",
    "peak_allocated_gb": "train/peak_allocated_gb",
    "peak_allocated_gb_per_rank": "train/peak_allocated_gb_per_rank",
}

VAL_STUDENT_GT_FIELDS = (
    "loss",
    "pixel_l1",
    "temporal_mse",
    "psnr",
    "ssim",
    "mae",
    "mse",
    "rmse",
)


def _finite_scalar(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def scalar_tags_from_record(
    record: Mapping[str, object],
    *,
    stream: str,
) -> dict[str, float]:
    """Map one SwiftVR JSONL record to stable TensorBoard scalar tags."""

    if stream not in {"train", "val"}:
        raise ValueError(f"stream must be 'train' or 'val', got {stream!r}")
    tags: dict[str, float] = {}
    if stream == "train":
        for field, tag in TRAIN_SCALAR_TAGS.items():
            value = _finite_scalar(record.get(field))
            if value is not None:
                tags[tag] = value
        return tags

    # Existing Stage-3 logs are flat and always mean student-vs-GT.
    for field in VAL_STUDENT_GT_FIELDS:
        value = _finite_scalar(record.get(field))
        if value is not None:
            tags[f"val/student_gt/{field}"] = value

    # New offline/reference evaluators may write nested groups.
    for group in ("student_gt", "reference_gt", "student_reference", "gap"):
        payload = record.get(group)
        if not isinstance(payload, Mapping):
            continue
        for field, raw in payload.items():
            value = _finite_scalar(raw)
            if value is not None:
                tags[f"val/{group}/{field}"] = value

    eval_seconds = _finite_scalar(record.get("eval_seconds"))
    if eval_seconds is not None:
        tags["val/eval_seconds"] = eval_seconds
    return tags


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return []
    records: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected JSON object")
            records.append(value)
    return records


def _summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "TensorBoard support requires the 'tensorboard' package. "
            "Install it in the SwiftVR environment before running this tool."
        ) from exc
    return SummaryWriter(log_dir=str(log_dir))


def write_records(
    writer,
    records: Iterable[Mapping[str, object]],
    *,
    stream: str,
) -> int:
    written = 0
    for record in records:
        step = int(record.get("global_step", 0))
        for tag, value in scalar_tags_from_record(record, stream=stream).items():
            writer.add_scalar(tag, value, global_step=step)
            written += 1
    return written


def backfill_tensorboard(
    run_dir: str | Path,
    *,
    log_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    root = Path(run_dir).expanduser().resolve()
    output = (
        Path(log_dir).expanduser().resolve()
        if log_dir is not None
        else root / "tensorboard"
    )
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    train_records = read_jsonl(root / "train_log.jsonl")
    val_records = read_jsonl(root / "val_log.jsonl")
    writer = _summary_writer(output)
    try:
        train_scalars = write_records(writer, train_records, stream="train")
        val_scalars = write_records(writer, val_records, stream="val")
        writer.flush()
    finally:
        writer.close()
    return {
        "run_dir": str(root),
        "log_dir": str(output),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_scalars": train_scalars,
        "val_scalars": val_scalars,
    }


class JSONLTensorBoardFollower:
    """Follow append-only SwiftVR logs and mirror new records to TensorBoard."""

    def __init__(self, run_dir: str | Path, log_dir: str | Path | None = None):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.log_dir = (
            Path(log_dir).expanduser().resolve()
            if log_dir is not None
            else self.run_dir / "tensorboard_live"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = _summary_writer(self.log_dir)
        self.positions = {"train": 0, "val": 0}

    def _poll_file(self, stream: str) -> int:
        path = self.run_dir / f"{stream}_log.jsonl"
        if not path.is_file():
            return 0
        written = 0
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(self.positions[stream])
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    handle.seek(start)
                    break
                self.positions[stream] = handle.tell()
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}: expected JSON object")
                written += write_records(self.writer, [record], stream=stream)
        return written

    def poll(self) -> int:
        written = self._poll_file("train") + self._poll_file("val")
        if written:
            self.writer.flush()
        return written

    def follow(self, poll_seconds: float = 2.0) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        try:
            while True:
                self.poll()
                time.sleep(poll_seconds)
        finally:
            self.writer.flush()
            self.writer.close()
