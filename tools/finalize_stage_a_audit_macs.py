#!/usr/bin/env python3
"""Merge steady-state streaming MACs into the Stage-A distillation audit.

Quality and fixed visuals come from ``audit_stage_a_distillation.py`` on val13.
Architecture compute comes from ``profile_stage_a_streaming_macs.py`` on a real
steady-state MIDDLE chunk. The prompt-free init/step992/long-run checkpoints share
one architecture, so they intentionally share the same canonical parameter/MAC
row while their quality metrics remain distinct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--streaming-macs-json", type=Path, required=True)
    return parser


def _metric(metrics: Mapping[str, object], name: str) -> str:
    value = metrics.get(name)
    if value is None:
        return "—"
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)


def _get_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _component_per_frame(record: Mapping[str, object], component: str) -> float:
    macs = _get_mapping(record.get("macs"), "record.macs")
    by_root = _get_mapping(
        macs.get("by_root_gmacs_per_output_frame"),
        "macs.by_root_gmacs_per_output_frame",
    )
    value = by_root.get(component)
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"Missing/invalid positive {component} GMACs/frame")
    return float(value)


def _total_gmac_per_frame(record: Mapping[str, object]) -> float:
    macs = _get_mapping(record.get("macs"), "record.macs")
    value = macs.get("gmacs_per_output_frame")
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError("Missing/invalid positive total GMACs/frame")
    return float(value)


def _canonical_params(record: Mapping[str, object]) -> int:
    params = _get_mapping(record.get("parameters"), "record.parameters")
    value = params.get("total_params")
    if not isinstance(value, (int, float)) or int(value) <= 0:
        raise ValueError("Missing/invalid total_params")
    return int(value)


def _format_compute_row(
    label: str,
    record: Mapping[str, object],
    *,
    teacher_total: float,
) -> str:
    encoder = _component_per_frame(record, "encoder")
    transformer = _component_per_frame(record, "transformer")
    decoder = _component_per_frame(record, "decoder")
    total = _total_gmac_per_frame(record)
    component_sum = encoder + transformer + decoder
    relative_error = abs(component_sum - total) / total
    if relative_error > 1e-9:
        raise ValueError(
            f"Component MAC sum mismatch for {label}: roots={component_sum}, total={total}"
        )
    ratio = total / teacher_total
    reduction = 100.0 * (1.0 - ratio)
    return (
        f"| {label} | {_canonical_params(record):,} | {encoder:.3f} | "
        f"{transformer:.3f} | {decoder:.3f} | {total:.3f} | {2.0 * total:.3f} | "
        f"{ratio:.4f}× | {reduction:.2f}% |"
    )


def write_markdown(
    audit: Mapping[str, object],
    compute: Mapping[str, object],
    path: Path,
) -> None:
    models = audit.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Stage-A audit has no models")
    teacher_quality = _get_mapping(audit.get("teacher"), "audit.teacher")
    teacher_metrics = _get_mapping(teacher_quality.get("metrics"), "teacher.metrics")
    teacher_compute = _get_mapping(compute.get("teacher"), "compute.teacher")
    student_compute = _get_mapping(compute.get("student"), "compute.student")
    teacher_total = _total_gmac_per_frame(teacher_compute)

    target_resolution = compute.get("target_resolution")
    internal_resolution = compute.get("internal_compute_resolution")
    clip_len = compute.get("clip_len")
    lines = [
        "# SwiftVR Stage-A Distillation Audit",
        "",
        "## Quality — deterministic val13",
        "",
        "| Model | Vel rel-L2 ↓ | Vel cosine ↑ | Teacher PSNR ↑ | Teacher SSIM ↑ | GT PSNR ↑ | GT SSIM ↑ | GT temporal MSE ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| Conditional teacher | 0 | 1 | — | — | {gpsnr} | {gssim} | {temp} |".format(
            gpsnr=_metric(teacher_metrics, "teacher_gt_psnr"),
            gssim=_metric(teacher_metrics, "teacher_gt_ssim"),
            temp=_metric(teacher_metrics, "teacher_gt_temporal_difference_mse"),
        ),
    ]
    for model in models:
        metrics = _get_mapping(model.get("metrics"), "model.metrics")
        lines.append(
            "| {label} | {rel} | {cos} | {rpsnr} | {rssim} | {gpsnr} | {gssim} | {temp} |".format(
                label=model.get("label", "student"),
                rel=_metric(metrics, "velocity_relative_l2"),
                cos=_metric(metrics, "velocity_cosine"),
                rpsnr=_metric(metrics, "student_teacher_psnr"),
                rssim=_metric(metrics, "student_teacher_ssim"),
                gpsnr=_metric(metrics, "student_gt_psnr"),
                gssim=_metric(metrics, "student_gt_ssim"),
                temp=_metric(metrics, "student_gt_temporal_difference_mse"),
            )
        )

    lines += [
        "",
        "## Compute — primary Stage-A comparison",
        "",
        (
            "Steady-state model MACs are measured on one warmed-up streaming MIDDLE "
            f"chunk at target resolution `{target_resolution}`, internal padded "
            f"resolution `{internal_resolution}`, clip length `{clip_len}`, SDPA, "
            "and `dit_overlap=0`."
        ),
        "",
        "Only executed Linear/Conv/attention matrix multiplications are included. Bias adds, elementwise operations, normalization, activation, RoPE, indexing/gather/scatter, interpolation, pixel shuffle/unshuffle and I/O are excluded from the MAC convention.",
        "",
        "`GFLOPs/frame*` uses the explicit convention **1 MAC = 2 FLOPs**.",
        "",
        "| Model | Canonical Params | Encoder GMAC/frame | DiT GMAC/frame | Decoder GMAC/frame | Total GMAC/frame | GFLOPs/frame* | MACs / Teacher | Reduction vs Teacher |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _format_compute_row(
            "Conditional teacher",
            teacher_compute,
            teacher_total=teacher_total,
        ),
    ]
    for model in models:
        lines.append(
            _format_compute_row(
                str(model.get("label", "student")),
                student_compute,
                teacher_total=teacher_total,
            )
        )

    teacher_calls = _get_mapping(
        _get_mapping(teacher_compute.get("macs"), "teacher.macs").get("calls_by_type"),
        "teacher.calls_by_type",
    )
    student_calls = _get_mapping(
        _get_mapping(student_compute.get("macs"), "student.macs").get("calls_by_type"),
        "student.calls_by_type",
    )
    lines += [
        "",
        "### Compute sanity checks",
        "",
        f"- Conditional teacher self-attention QK calls observed: `{teacher_calls.get('self_attn_qk', 0)}`.",
        f"- Conditional teacher cross-attention QK calls observed: `{teacher_calls.get('cross_attn_qk', 0)}`.",
        f"- Prompt-free student self-attention QK calls observed: `{student_calls.get('self_attn_qk', 0)}`.",
        f"- Prompt-free student cross-attention QK calls observed: `{student_calls.get('cross_attn_qk', 0)}` (expected 0).",
        "- Prompt-free init, step992 and long-run checkpoints intentionally share identical compute because Stage A changes weights, not the prompt-free/no-time architecture.",
        "",
        "## Supplementary hardware profile",
        "",
        "Wall-clock latency/FPS values remain available in `stage_a_audit.json` and the streaming MAC JSON, but they are not used as the primary method-complexity claim.",
        "",
        "## Visuals",
        "",
        "See `visuals/` for fixed LQ / GT / conditional-teacher / student comparisons and absolute-difference videos.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    audit_dir = args.audit_dir.expanduser().resolve()
    audit_path = audit_dir / "stage_a_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    compute_path = args.streaming_macs_json.expanduser().resolve()
    if not compute_path.is_file():
        raise FileNotFoundError(compute_path)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    compute = json.loads(compute_path.read_text(encoding="utf-8"))
    if compute.get("kind") != "swiftvr_stage_a_streaming_macs":
        raise ValueError(f"Unexpected compute report kind: {compute.get('kind')!r}")
    if compute.get("attention_backend") != "sdpa":
        raise ValueError("Canonical Stage-A MAC report must use SDPA")
    if int(compute.get("dit_overlap", -1)) != 0:
        raise ValueError("Canonical Stage-A MAC report must use dit_overlap=0")

    teacher_compute = _get_mapping(compute.get("teacher"), "compute.teacher")
    student_compute = _get_mapping(compute.get("student"), "compute.student")
    teacher_macs = _get_mapping(teacher_compute.get("macs"), "teacher.macs")
    student_macs = _get_mapping(student_compute.get("macs"), "student.macs")
    for label, macs in (("teacher", teacher_macs), ("student", student_macs)):
        errors = macs.get("count_errors")
        if not isinstance(errors, list) or errors:
            raise RuntimeError(f"{label} MAC counter diagnostics are not clean: {errors}")

    audit["stage_a_streaming_compute"] = compute
    teacher = audit.get("teacher")
    if isinstance(teacher, dict):
        teacher["canonical_parameters"] = teacher_compute["parameters"]
        teacher["streaming_compute"] = teacher_compute
    for model in audit.get("models", []):
        if isinstance(model, dict):
            model["canonical_parameters"] = student_compute["parameters"]
            model["streaming_compute"] = student_compute

    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = audit_dir / "stage_a_audit.md"
    write_markdown(audit, compute, markdown_path)

    teacher_total = _total_gmac_per_frame(teacher_compute)
    student_total = _total_gmac_per_frame(student_compute)
    print(f"Conditional teacher: {teacher_total:.3f} GMAC/frame")
    print(f"Prompt-free student: {student_total:.3f} GMAC/frame")
    print(f"Student / teacher : {student_total / teacher_total:.4f}x")
    print(f"Reduction         : {100.0 * (1.0 - student_total / teacher_total):.2f}%")
    print(f"Updated JSON      : {audit_path}")
    print(f"Updated Markdown  : {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
