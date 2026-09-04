#!/usr/bin/env python3
"""Plan a source-aware SwiftVR teacher cache and DDP training schedule on CPU."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from swiftvr.data import read_triplet_manifests
from swiftvr.training.distillation_generalization import (
    SELECTION_MODES,
    SOURCE_IDENTITY_METHOD,
    record_source_uid,
    select_distillation_indices,
    selected_indices_sha256,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="train")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--views-per-record", type=int, default=1)
    parser.add_argument(
        "--selection-mode",
        choices=tuple(sorted(SELECTION_MODES)),
        default="all",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--duplicate-examples", type=int, default=12)
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def _optimizer_schedule(
    sample_count: int,
    *,
    world_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: float,
) -> dict[str, object]:
    if min(world_size, batch_size, gradient_accumulation_steps) <= 0:
        raise ValueError("world-size, batch-size, and accumulation must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    samples_per_rank_epoch = sample_count // world_size
    batches_per_rank_epoch = samples_per_rank_epoch // batch_size
    optimizer_steps_per_epoch = batches_per_rank_epoch // gradient_accumulation_steps
    if optimizer_steps_per_epoch <= 0:
        raise ValueError(
            "Selected dataset is too small for one optimizer step per DDP epoch"
        )
    consumed_per_epoch = (
        optimizer_steps_per_epoch
        * world_size
        * batch_size
        * gradient_accumulation_steps
    )
    dropped_per_epoch = sample_count - consumed_per_epoch
    max_steps = max(1, int(round(float(epochs) * optimizer_steps_per_epoch)))
    effective_epochs = max_steps / optimizer_steps_per_epoch
    return {
        "world_size": world_size,
        "local_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "global_effective_batch_size": (
            world_size * batch_size * gradient_accumulation_steps
        ),
        "samples_per_rank_epoch": samples_per_rank_epoch,
        "batches_per_rank_epoch": batches_per_rank_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "samples_consumed_per_epoch": consumed_per_epoch,
        "samples_dropped_per_epoch": dropped_per_epoch,
        "requested_epochs": epochs,
        "recommended_max_steps": max_steps,
        "effective_epochs": effective_epochs,
        "half_epoch_steps": max(1, optimizer_steps_per_epoch // 2),
        "quarter_epoch_steps": max(1, optimizer_steps_per_epoch // 4),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_length <= 0 or args.views_per_record <= 0:
        raise ValueError("clip-length and views-per-record must be positive")

    path_root = args.path_root.expanduser().resolve()
    records = read_triplet_manifests(
        args.manifest,
        split=args.split,
        path_root=path_root,
        verify_paths=args.verify_paths,
    )
    eligible = [record for record in records if record.frame_count >= args.clip_length]
    if not eligible:
        raise RuntimeError("No records remain after clip-length filtering")

    source_uids = tuple(record_source_uid(record) for record in eligible)
    source_to_records: dict[str, list[object]] = defaultdict(list)
    for source_uid, record in zip(source_uids, eligible):
        source_to_records[source_uid].append(record)

    full_dataset_length = len(eligible) * args.views_per_record
    selected_indices = select_distillation_indices(
        full_dataset_length,
        max_samples=args.max_samples,
        mode=args.selection_mode,
        seed=args.selection_seed,
        source_uids=source_uids,
        views_per_record=args.views_per_record,
    )
    selected_record_indices = {
        index // args.views_per_record for index in selected_indices
    }
    selected_source_uids = {
        source_uids[index] for index in selected_record_indices
    }
    selected_source_frequency = Counter(
        source_uids[index // args.views_per_record] for index in selected_indices
    )

    duplicate_sources = [
        (source_uid, grouped)
        for source_uid, grouped in source_to_records.items()
        if len(grouped) > 1
    ]
    duplicate_sources.sort(key=lambda item: (-len(item[1]), item[0]))
    duplicate_examples = []
    for source_uid, grouped in duplicate_sources[: args.duplicate_examples]:
        duplicate_examples.append(
            {
                "source_uid": source_uid,
                "record_count": len(grouped),
                "records": [
                    {
                        "record_uid": f"{record.variant}:{record.sample_id}",
                        "variant": record.variant,
                        "source_manifest": record.source_manifest,
                        "hr_first": record.hr_paths[0],
                    }
                    for record in grouped
                ],
            }
        )

    selection_frequency_histogram = Counter(selected_source_frequency.values())
    schedule = _optimizer_schedule(
        len(selected_indices),
        world_size=args.world_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
    )
    report: dict[str, object] = {
        "manifests": [str(path.expanduser().resolve()) for path in args.manifest],
        "path_root": str(path_root),
        "split": args.split,
        "clip_length": args.clip_length,
        "loaded_record_count": len(records),
        "eligible_record_count": len(eligible),
        "dropped_short_record_count": len(records) - len(eligible),
        "unique_hr_source_count": len(source_to_records),
        "duplicate_hr_source_count": len(duplicate_sources),
        "max_records_per_hr_source": max(len(grouped) for grouped in source_to_records.values()),
        "source_identity_method": SOURCE_IDENTITY_METHOD,
        "views_per_record": args.views_per_record,
        "full_dataset_length": full_dataset_length,
        "selection_mode": args.selection_mode,
        "selection_seed": args.selection_seed,
        "max_samples": args.max_samples,
        "selected_sample_count": len(selected_indices),
        "selected_record_count": len(selected_record_indices),
        "selected_unique_hr_source_count": len(selected_source_uids),
        "selected_source_coverage": len(selected_source_uids) / len(source_to_records),
        "selected_views_per_source_histogram": {
            str(key): value for key, value in sorted(selection_frequency_histogram.items())
        },
        "selected_indices_sha256": selected_indices_sha256(selected_indices),
        "training_schedule": schedule,
        "duplicate_source_examples": duplicate_examples,
    }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
