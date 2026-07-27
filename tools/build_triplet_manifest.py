#!/usr/bin/env python3
"""Build deterministic manifests for aligned HR/HQ/LR media triplets.

The three dataset roots may contain encoded videos or image-frame sequences.
Frame sequences are grouped using regexes with named ``clip`` and ``frame``
groups. The default matches ``video1_comp2_000123.png``; per-root overrides
support asymmetric names such as clean HR/HQ frames and ``*_text.png`` LR.

For datasets with official train/validation/test directories, use
``--split-all``. Otherwise records are divided deterministically by sample ID.
This tool inspects paths and frame indices only; pixel alignment is audited by
``tools/audit_triplet_alignment.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Pattern

DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")
DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
DEFAULT_FRAME_REGEX = r"^(?P<clip>.+)_(?P<frame>\d{6})$"
VALID_MEDIA_MODES = ("auto", "video", "frames")
VALID_MATCH_MODES = ("relative_stem", "basename_stem")
VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TripletRecord:
    sample_id: str
    hr: str
    hq: str
    lr: str
    split: str
    media_type: str
    frame_start: int | None = None
    frame_end: int | None = None
    frame_count: int | None = None
    frame_digits: int | None = None
    frame_contiguous: bool | None = None
    frame_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FrameSequence:
    pattern: str
    indices: tuple[int, ...]
    frame_digits: int

    @property
    def contiguous(self) -> bool:
        return bool(self.indices) and self.indices == tuple(
            range(self.indices[0], self.indices[-1] + 1)
        )


@dataclass(frozen=True)
class FrameIndexResult:
    sequences: dict[str, FrameSequence]
    matched_file_count: int
    unmatched_file_count: int
    unmatched_examples: tuple[str, ...]


def parse_extensions(value: str | Iterable[str]) -> tuple[str, ...]:
    values = value.split(",") if isinstance(value, str) else list(value)
    normalized: list[str] = []
    for extension in values:
        extension = str(extension).strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.append(extension)
    if not normalized:
        raise ValueError("At least one media extension is required")
    return tuple(dict.fromkeys(normalized))


def compile_frame_regex(value: str) -> Pattern[str]:
    try:
        pattern = re.compile(value)
    except re.error as exc:
        raise ValueError(f"Invalid frame regex {value!r}: {exc}") from exc
    if not {"clip", "frame"}.issubset(pattern.groupindex):
        raise ValueError(
            "Frame regex must define named groups (?P<clip>...) and "
            "(?P<frame>...)"
        )
    return pattern


def logical_key(parent: Path, root: Path, clip: str, match_mode: str) -> str:
    if match_mode == "basename_stem":
        return clip
    if match_mode != "relative_stem":
        raise ValueError(f"Unsupported match mode: {match_mode}")
    relative_parent = parent.relative_to(root)
    return clip if relative_parent == Path(".") else (relative_parent / clip).as_posix()


def _raise_duplicates(
    root: Path,
    duplicates: dict[str, list[Path]],
    match_mode: str,
) -> None:
    preview = {
        key: [str(path) for path in paths]
        for key, paths in list(sorted(duplicates.items()))[:10]
    }
    raise ValueError(
        f"Duplicate match keys under {root} using mode={match_mode}: "
        f"{json.dumps(preview, indent=2)}"
    )


def index_videos(
    root: Path,
    *,
    extensions: tuple[str, ...],
    match_mode: str,
) -> dict[str, Path]:
    root = root.resolve()
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = logical_key(path.parent, root, path.stem, match_mode)
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(path.resolve())
        else:
            index[key] = path.resolve()
    if duplicates:
        _raise_duplicates(root, duplicates, match_mode)
    return index


def _frame_pattern(path: Path, match: re.Match[str], frame_digits: int) -> str:
    frame_start, frame_end = match.span("frame")
    stem = path.stem
    pattern_stem = (
        stem[:frame_start]
        + f"{{frame:0{frame_digits}d}}"
        + stem[frame_end:]
    )
    return str(path.parent / f"{pattern_stem}{path.suffix.lower()}")


def index_frame_sequences(
    root: Path,
    *,
    extensions: tuple[str, ...],
    match_mode: str,
    frame_regex: Pattern[str],
) -> FrameIndexResult:
    root = root.resolve()
    grouped: dict[str, list[tuple[int, str, Path]]] = {}
    unmatched: list[str] = []
    matched_file_count = 0
    unmatched_file_count = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        match = frame_regex.fullmatch(path.stem)
        if match is None:
            unmatched_file_count += 1
            if len(unmatched) < 20:
                unmatched.append(str(path.resolve()))
            continue
        matched_file_count += 1
        clip = match.group("clip")
        frame_text = match.group("frame")
        key = logical_key(path.parent, root, clip, match_mode)
        grouped.setdefault(key, []).append((int(frame_text), frame_text, path.resolve()))

    if not grouped:
        detail = f" unmatched_examples={unmatched}" if unmatched else ""
        raise ValueError(
            f"No frame sequences found under {root} using regex="
            f"{frame_regex.pattern!r}.{detail}"
        )

    sequences: dict[str, FrameSequence] = {}
    duplicate_frames: dict[str, list[Path]] = {}
    for key, rows in sorted(grouped.items()):
        widths = {len(frame_text) for _, frame_text, _ in rows}
        suffixes = {path.suffix.lower() for _, _, path in rows}
        if len(widths) != 1 or len(suffixes) != 1:
            raise ValueError(
                f"Sequence {key!r} mixes frame widths or extensions: "
                f"widths={sorted(widths)}, extensions={sorted(suffixes)}"
            )

        by_index: dict[int, Path] = {}
        for frame_index, _, path in rows:
            if frame_index in by_index:
                duplicate_frames.setdefault(key, [by_index[frame_index]]).append(path)
            else:
                by_index[frame_index] = path
        if key in duplicate_frames:
            continue

        first_path = rows[0][2]
        first_match = frame_regex.fullmatch(first_path.stem)
        assert first_match is not None
        digits = next(iter(widths))
        sequences[key] = FrameSequence(
            pattern=_frame_pattern(first_path, first_match, digits),
            indices=tuple(sorted(by_index)),
            frame_digits=digits,
        )

    if duplicate_frames:
        _raise_duplicates(root, duplicate_frames, match_mode)

    return FrameIndexResult(
        sequences=sequences,
        matched_file_count=matched_file_count,
        unmatched_file_count=unmatched_file_count,
        unmatched_examples=tuple(unmatched),
    )


def detect_root_mode(
    root: Path,
    *,
    video_extensions: tuple[str, ...],
    image_extensions: tuple[str, ...],
) -> str:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")

    has_video = False
    has_image = False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        has_video = has_video or suffix in video_extensions
        has_image = has_image or suffix in image_extensions
        if has_video and has_image:
            break

    if has_video and has_image:
        raise ValueError(
            f"Dataset root contains both videos and images: {root}. "
            "Pass --media-mode video or --media-mode frames explicitly."
        )
    if has_video:
        return "video"
    if has_image:
        return "frames"
    raise ValueError(f"No supported media files found under {root}")


def deterministic_split(
    sample_id: str,
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> str:
    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions must be non-negative")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def assign_split(
    sample_id: str,
    *,
    split_all: str | None,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> str:
    if split_all is not None:
        if split_all not in VALID_SPLITS:
            raise ValueError(
                f"Unsupported fixed split {split_all!r}; expected one of {VALID_SPLITS}"
            )
        return split_all
    return deterministic_split(
        sample_id,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )


def _resolve_frame_regexes(
    frame_regex: str,
    hr_frame_regex: str | None,
    hq_frame_regex: str | None,
    lr_frame_regex: str | None,
) -> dict[str, str]:
    return {
        "hr": hr_frame_regex or frame_regex,
        "hq": hq_frame_regex or frame_regex,
        "lr": lr_frame_regex or frame_regex,
    }


def build_manifest(
    *,
    hr_root: Path,
    hq_root: Path,
    lr_root: Path,
    video_extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS,
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
    media_mode: str = "auto",
    frame_regex: str = DEFAULT_FRAME_REGEX,
    hr_frame_regex: str | None = None,
    hq_frame_regex: str | None = None,
    lr_frame_regex: str | None = None,
    match_mode: str = "relative_stem",
    split_all: str | None = None,
    seed: int = 0,
    val_fraction: float = 0.05,
    test_fraction: float = 0.0,
    strict: bool = False,
) -> tuple[list[TripletRecord], dict[str, object]]:
    roots = {"hr": hr_root.resolve(), "hq": hq_root.resolve(), "lr": lr_root.resolve()}
    video_extensions = parse_extensions(video_extensions)
    image_extensions = parse_extensions(image_extensions)

    if media_mode not in VALID_MEDIA_MODES:
        raise ValueError(f"Unsupported media mode: {media_mode}")
    if match_mode not in VALID_MATCH_MODES:
        raise ValueError(f"Unsupported match mode: {match_mode}")
    if split_all is not None and split_all not in VALID_SPLITS:
        raise ValueError(
            f"Unsupported fixed split {split_all!r}; expected one of {VALID_SPLITS}"
        )
    if split_all is None:
        deterministic_split(
            "__split_validation__",
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

    if media_mode == "auto":
        detected = {
            name: detect_root_mode(
                root,
                video_extensions=video_extensions,
                image_extensions=image_extensions,
            )
            for name, root in roots.items()
        }
        if len(set(detected.values())) != 1:
            raise ValueError(f"Dataset roots use different media modes: {detected}")
        resolved_mode = next(iter(detected.values()))
    else:
        resolved_mode = media_mode
        for root in roots.values():
            if not root.is_dir():
                raise FileNotFoundError(f"Dataset root is not a directory: {root}")

    frame_regexes: dict[str, str] | None = None
    frame_scan: dict[str, FrameIndexResult] | None = None
    if resolved_mode == "video":
        indices: dict[str, dict[str, object]] = {
            name: index_videos(root, extensions=video_extensions, match_mode=match_mode)
            for name, root in roots.items()
        }
    else:
        frame_regexes = _resolve_frame_regexes(
            frame_regex, hr_frame_regex, hq_frame_regex, lr_frame_regex
        )
        compiled_regexes = {
            name: compile_frame_regex(value) for name, value in frame_regexes.items()
        }
        frame_scan = {
            name: index_frame_sequences(
                root,
                extensions=image_extensions,
                match_mode=match_mode,
                frame_regex=compiled_regexes[name],
            )
            for name, root in roots.items()
        }
        indices = {name: result.sequences for name, result in frame_scan.items()}

    if any(not index for index in indices.values()):
        counts = {name: len(index) for name, index in indices.items()}
        raise ValueError(f"One or more roots contain no usable clips: {counts}")

    all_keys = set().union(*(set(index) for index in indices.values()))
    common_keys = set.intersection(*(set(index) for index in indices.values()))
    missing = {name: sorted(all_keys - set(index)) for name, index in indices.items()}
    if strict and any(missing.values()):
        counts = {name: len(keys) for name, keys in missing.items()}
        raise ValueError(
            "Dataset roots do not contain identical triplet keys: "
            f"missing_counts={counts}"
        )

    records: list[TripletRecord] = []
    split_counts = {split: 0 for split in VALID_SPLITS}
    invalid_sequences: dict[str, str] = {}
    non_contiguous_sequences: dict[str, dict[str, object]] = {}
    frame_counts: list[int] = []

    for key in sorted(common_keys):
        frame_start = frame_end = frame_count = frame_digits = None
        frame_contiguous = None
        frame_indices = None
        if resolved_mode == "frames":
            sequences = {name: index[key] for name, index in indices.items()}
            assert all(isinstance(sequence, FrameSequence) for sequence in sequences.values())
            sequence_values = list(sequences.values())
            same_indices = len({sequence.indices for sequence in sequence_values}) == 1
            same_digits = len({sequence.frame_digits for sequence in sequence_values}) == 1
            if not same_indices or not same_digits:
                reason = f"same_indices={same_indices}, same_digits={same_digits}"
                invalid_sequences[key] = reason
                if strict:
                    raise ValueError(
                        f"Frame sequence alignment failed for {key!r}: {reason}"
                    )
                continue

            shared_indices = sequence_values[0].indices
            shared_contiguous = sequence_values[0].contiguous
            if not shared_contiguous and not strict:
                reason = (
                    "synchronized_non_contiguous=True; rerun with --strict to "
                    "preserve explicit frame_indices"
                )
                invalid_sequences[key] = reason
                continue
            frame_start = shared_indices[0]
            frame_end = shared_indices[-1]
            frame_count = len(shared_indices)
            frame_digits = sequence_values[0].frame_digits
            frame_contiguous = shared_contiguous
            if not frame_contiguous:
                frame_indices = shared_indices
                non_contiguous_sequences[key] = {
                    "frame_count": frame_count,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "index_preview": list(shared_indices[:20]),
                }
            frame_counts.append(frame_count)
            hr_value = sequences["hr"].pattern
            hq_value = sequences["hq"].pattern
            lr_value = sequences["lr"].pattern
        else:
            hr_value = str(indices["hr"][key])
            hq_value = str(indices["hq"][key])
            lr_value = str(indices["lr"][key])

        split = assign_split(
            key,
            split_all=split_all,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        split_counts[split] += 1
        records.append(
            TripletRecord(
                sample_id=key,
                hr=hr_value,
                hq=hq_value,
                lr=lr_value,
                split=split,
                media_type=resolved_mode,
                frame_start=frame_start,
                frame_end=frame_end,
                frame_count=frame_count,
                frame_digits=frame_digits,
                frame_contiguous=frame_contiguous,
                frame_indices=frame_indices,
            )
        )

    split_strategy = "fixed" if split_all is not None else "deterministic_hash"
    summary: dict[str, object] = {
        "manifest_version": 4,
        "hr_root": str(roots["hr"]),
        "hq_root": str(roots["hq"]),
        "lr_root": str(roots["lr"]),
        "media_mode": resolved_mode,
        "match_mode": match_mode,
        "video_extensions": list(video_extensions),
        "image_extensions": list(image_extensions),
        "frame_regex": frame_regex if resolved_mode == "frames" else None,
        "frame_regexes": frame_regexes,
        "split_strategy": split_strategy,
        "split_all": split_all,
        "seed": int(seed) if split_all is None else None,
        "val_fraction": float(val_fraction) if split_all is None else None,
        "test_fraction": float(test_fraction) if split_all is None else None,
        "indexed_counts": {name: len(index) for name, index in indices.items()},
        "triplet_count": len(records),
        "split_counts": split_counts,
        "missing_counts": {name: len(keys) for name, keys in missing.items()},
        "missing_examples": {name: keys[:20] for name, keys in missing.items()},
        "invalid_sequence_count": len(invalid_sequences),
        "invalid_sequence_examples": dict(list(sorted(invalid_sequences.items()))[:20]),
        "non_contiguous_sequence_count": len(non_contiguous_sequences),
        "non_contiguous_sequence_examples": dict(
            list(sorted(non_contiguous_sequences.items()))[:20]
        ),
    }
    if frame_scan is not None:
        summary["frame_scan"] = {
            name: {
                "matched_file_count": result.matched_file_count,
                "unmatched_file_count": result.unmatched_file_count,
                "unmatched_examples": list(result.unmatched_examples),
            }
            for name, result in frame_scan.items()
        }
    if frame_counts:
        summary["frame_count_stats"] = {
            "min": min(frame_counts),
            "max": max(frame_counts),
            "total": sum(frame_counts),
        }
    return records, summary


def write_manifest(
    records: list[TripletRecord],
    summary: dict[str, object],
    output: Path,
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                key: value for key, value in asdict(record).items() if value is not None
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr-root", type=Path, required=True)
    parser.add_argument("--hq-root", type=Path, required=True)
    parser.add_argument("--lr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--media-mode",
        choices=VALID_MEDIA_MODES,
        default="auto",
        help="Auto-detect encoded videos or image-frame sequences",
    )
    parser.add_argument(
        "--match-mode", choices=VALID_MATCH_MODES, default="relative_stem"
    )
    parser.add_argument(
        "--video-extensions",
        default=",".join(DEFAULT_VIDEO_EXTENSIONS),
        help="Comma-separated encoded-video extensions",
    )
    parser.add_argument(
        "--image-extensions",
        default=",".join(DEFAULT_IMAGE_EXTENSIONS),
        help="Comma-separated image extensions",
    )
    parser.add_argument(
        "--frame-regex",
        default=DEFAULT_FRAME_REGEX,
        help=(
            "Common regex applied to image stems. It must define named groups "
            "'clip' and 'frame'."
        ),
    )
    for root_name in ("hr", "hq", "lr"):
        parser.add_argument(
            f"--{root_name}-frame-regex",
            default=None,
            help=(
                f"Optional regex override for {root_name.upper()} image stems. "
                "It must define named groups 'clip' and 'frame'."
            ),
        )
    parser.add_argument(
        "--split-all",
        choices=VALID_SPLITS,
        default=None,
        help=(
            "Assign every record to one official split and bypass hash-based "
            "fractional splitting."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Hash-split validation fraction; ignored with --split-all",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Hash-split test fraction; ignored with --split-all",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail on missing triplets or mismatched HR/HQ/LR frame-index sets. "
            "Synchronized non-contiguous indices are preserved explicitly."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, summary = build_manifest(
        hr_root=args.hr_root,
        hq_root=args.hq_root,
        lr_root=args.lr_root,
        video_extensions=parse_extensions(args.video_extensions),
        image_extensions=parse_extensions(args.image_extensions),
        media_mode=args.media_mode,
        frame_regex=args.frame_regex,
        hr_frame_regex=args.hr_frame_regex,
        hq_frame_regex=args.hq_frame_regex,
        lr_frame_regex=args.lr_frame_regex,
        match_mode=args.match_mode,
        split_all=args.split_all,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        strict=args.strict,
    )
    if not records:
        raise RuntimeError("No complete aligned HR/HQ/LR triplets were found")
    write_manifest(records, summary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
