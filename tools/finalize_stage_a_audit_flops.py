#!/usr/bin/env python3
"""Finalize a Stage-A audit with conditional-teacher FLOPs as the primary compute view.

This script augments ``stage_a_audit.json`` produced by
``tools/audit_stage_a_distillation.py``.  Student FLOPs are reused from the audit;
the original conditional teacher is reconstructed from immutable teacher-cache
metadata and profiled on the exact same deterministic validation batch.  The
Markdown report is then rewritten so FLOPs, rather than hardware-specific latency,
are the primary efficiency comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping

import torch
from safetensors.torch import load_file

try:
    import audit_stage_a_distillation as audit
except ModuleNotFoundError:
    from tools import audit_stage_a_distillation as audit

from swiftvr.models import ReAE
from swiftvr.models.transformer import WanTransformer3DModel
from swiftvr.training.distillation import TeacherVelocityCache
from swiftvr.training.forward import decode_reae_clip, encode_reae_clip, prepare_training_batch
from swiftvr.training.reference import expand_prompt_embedding, extract_transformer_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--split", default="val")
    parser.add_argument("--clip-length", type=int, default=13)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--views-per-record", type=int, default=1)
    parser.add_argument("--view-seed", type=int, default=9000001)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--allow-dtype-mismatch", action="store_true")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    return parser


def _parameter_summary(reae: torch.nn.Module, transformer: torch.nn.Module) -> dict[str, int]:
    parameters = list(reae.parameters()) + list(transformer.parameters())
    return {
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
    }


def _count_flops(fn: Callable[[], object]) -> tuple[int | None, str | None]:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as exc:
        return None, f"FlopCounterMode unavailable: {type(exc).__name__}: {exc}"
    try:
        with FlopCounterMode(display=False) as counter:
            fn()
        return int(counter.get_total_flops()), None
    except Exception as exc:
        return None, f"FLOP counting failed: {type(exc).__name__}: {exc}"


def _metadata_path(cache: TeacherVelocityCache, key: str, fallback: str | None = None) -> Path:
    value = cache.metadata.get(key, fallback)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Teacher cache metadata is missing {key!r}")
    return Path(value).expanduser().resolve()


def _load_teacher(
    cache: TeacherVelocityCache,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attention_backend: str,
    reae_filename: str,
    transformer_subfolder: str,
):
    reference_value = cache.metadata.get("reference_checkpoint")
    if not isinstance(reference_value, str) or not reference_value:
        raise ValueError("Teacher cache metadata is missing reference_checkpoint")
    reference_root = Path(reference_value).expanduser().resolve()

    reae_path_value = cache.metadata.get("reae_file")
    reae_path = (
        Path(reae_path_value).expanduser().resolve()
        if isinstance(reae_path_value, str) and reae_path_value
        else reference_root / reae_filename
    )
    prompt_path_value = cache.metadata.get("prompt_embedding_file")
    prompt_path = (
        Path(prompt_path_value).expanduser().resolve()
        if isinstance(prompt_path_value, str) and prompt_path_value
        else reference_root / "prompt_embedding.safetensors"
    )
    prompt_key = str(cache.metadata.get("prompt_key", "prompt_emb"))
    timestep = float(cache.metadata.get("timestep", 1000.0))

    prompt_payload = load_file(str(prompt_path), device="cpu")
    if prompt_key not in prompt_payload:
        raise KeyError(f"{prompt_path} does not contain prompt key {prompt_key!r}")
    prompt_embedding = prompt_payload[prompt_key]

    reae = ReAE(str(reae_path)).to(device=device, dtype=dtype).eval()
    transformer = WanTransformer3DModel.from_pretrained(
        str(reference_root),
        subfolder=transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype).eval()
    transformer.prepare_for_inference(attention_backend=attention_backend)
    for module in (reae, transformer):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return reference_root, reae, transformer, prompt_embedding, prompt_key, timestep


def _profile_teacher(
    *,
    reae: torch.nn.Module,
    transformer: torch.nn.Module,
    prompt_embedding: torch.Tensor,
    timestep: float,
    batch_cpu: Mapping[str, object],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    batch = audit.gate.move_video_batch(batch_cpu, device=device, dtype=dtype)
    prepared = prepare_training_batch(batch)
    lq_input = prepared["lq_input"]
    target = prepared["target"]
    if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("Teacher profiling batch is missing lq_input/target")
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        z_ntchw = encode_reae_clip(reae, lq_input, require_4k_plus_1=True)
        z_lq = z_ntchw.permute(0, 2, 1, 3, 4).contiguous()
        prompt = expand_prompt_embedding(prompt_embedding, int(z_lq.shape[0])).to(
            device=device, dtype=dtype
        )
        timesteps = torch.full(
            (int(z_lq.shape[0]),),
            timestep,
            device=device,
            dtype=torch.float32,
        )
        velocity = extract_transformer_sample(
            transformer(z_lq, timesteps, prompt, return_dict=True)
        )

    def encode_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            return encode_reae_clip(reae, lq_input, require_4k_plus_1=True)

    def transformer_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            return extract_transformer_sample(
                transformer(z_lq, timesteps, prompt, return_dict=True)
            )

    def decode_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            z_teacher = z_lq - velocity
            return decode_reae_clip(
                reae,
                z_teacher.permute(0, 2, 1, 3, 4).contiguous(),
                output_frames=int(target.shape[1]),
                clamp=False,
            )

    def end_to_end_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            z_ntchw_local = encode_reae_clip(reae, lq_input, require_4k_plus_1=True)
            z_local = z_ntchw_local.permute(0, 2, 1, 3, 4).contiguous()
            prompt_local = expand_prompt_embedding(
                prompt_embedding, int(z_local.shape[0])
            ).to(device=device, dtype=dtype)
            timesteps_local = torch.full(
                (int(z_local.shape[0]),),
                timestep,
                device=device,
                dtype=torch.float32,
            )
            velocity_local = extract_transformer_sample(
                transformer(z_local, timesteps_local, prompt_local, return_dict=True)
            )
            return decode_reae_clip(
                reae,
                (z_local - velocity_local).permute(0, 2, 1, 3, 4).contiguous(),
                output_frames=int(target.shape[1]),
                clamp=False,
            )

    flops: dict[str, object] = {}
    for name, fn in (
        ("encoder", encode_fn),
        ("transformer", transformer_fn),
        ("decoder", decode_fn),
        ("end_to_end", end_to_end_fn),
    ):
        count, error = _count_flops(fn)
        flops[name] = {"reported_flops": count, "error": error}
    return {
        "input_shape": list(lq_input.shape),
        "latent_shape": list(z_lq.shape),
        "output_frames": int(target.shape[1]),
        "flops": flops,
    }


def _flops_value(profile: Mapping[str, object], component: str) -> int | None:
    flops = profile.get("flops")
    if not isinstance(flops, Mapping):
        return None
    item = flops.get(component)
    if not isinstance(item, Mapping):
        return None
    value = item.get("reported_flops")
    return int(value) if isinstance(value, (int, float)) else None


def _tflops(value: int | None) -> str:
    return "—" if value is None else f"{value / 1e12:.3f}"


def _gflops_per_frame(profile: Mapping[str, object]) -> str:
    total = _flops_value(profile, "end_to_end")
    frames_value = profile.get("output_frames")
    if total is None or not isinstance(frames_value, int) or frames_value <= 0:
        input_shape = profile.get("input_shape")
        if isinstance(input_shape, list) and len(input_shape) >= 2:
            frames_value = int(input_shape[1])
        else:
            return "—"
    return f"{total / frames_value / 1e9:.3f}"


def _ratio_to_teacher(value: int | None, teacher: int | None) -> tuple[str, str]:
    if value is None or teacher is None or teacher <= 0:
        return "—", "—"
    ratio = value / teacher
    reduction = 1.0 - ratio
    return f"{ratio:.4f}×", f"{100.0 * reduction:.2f}%"


def _metric(metrics: Mapping[str, object], name: str) -> str:
    value = metrics.get(name)
    if value is None:
        return "—"
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)


def _latency_median(profile: Mapping[str, object], name: str) -> str:
    value = profile.get(name)
    if not isinstance(value, Mapping):
        return "—"
    median = value.get("median_ms")
    return "—" if not isinstance(median, (int, float)) else f"{float(median):.3f}"


def write_markdown(report: Mapping[str, object], path: Path) -> None:
    models = report.get("models")
    if not isinstance(models, list):
        raise TypeError("audit report models must be a list")
    teacher = report.get("teacher")
    if not isinstance(teacher, Mapping):
        raise TypeError("audit report is missing teacher")

    lines = [
        "# SwiftVR Stage-A Distillation Audit",
        "",
        "## Quality",
        "",
        "| Model | Vel rel-L2 ↓ | Vel cosine ↑ | Teacher PSNR ↑ | Teacher SSIM ↑ | GT PSNR ↑ | GT SSIM ↑ | GT temporal MSE ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        metrics = model.get("metrics", {})
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
    teacher_metrics = teacher.get("metrics", {})
    lines.append(
        "| Conditional teacher | 0 | 1 | — | — | {gpsnr} | {gssim} | {temp} |".format(
            gpsnr=_metric(teacher_metrics, "teacher_gt_psnr"),
            gssim=_metric(teacher_metrics, "teacher_gt_ssim"),
            temp=_metric(teacher_metrics, "teacher_gt_temporal_difference_mse"),
        )
    )

    teacher_profile = teacher.get("profile")
    if not isinstance(teacher_profile, Mapping):
        raise TypeError("teacher profile is missing")
    teacher_params = teacher.get("parameters")
    if not isinstance(teacher_params, Mapping):
        raise TypeError("teacher parameter summary is missing")
    teacher_e2e = _flops_value(teacher_profile, "end_to_end")

    lines += [
        "",
        "## Compute — primary comparison",
        "",
        "FLOPs are measured on the exact same deterministic validation clip and are the primary compute metric for Stage A. Latency is reported only as supplementary hardware information.",
        "",
        "| Model | Params | Encoder TFLOPs | DiT TFLOPs | Decoder TFLOPs | E2E TFLOPs | GFLOPs/frame | FLOPs / Teacher | Reduction vs Teacher |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def append_compute_row(label: str, params: Mapping[str, object], profile: Mapping[str, object]):
        total = _flops_value(profile, "end_to_end")
        ratio, reduction = _ratio_to_teacher(total, teacher_e2e)
        lines.append(
            "| {label} | {params:,} | {enc} | {dit} | {dec} | {e2e} | {per_frame} | {ratio} | {reduction} |".format(
                label=label,
                params=int(params.get("total_parameters", 0)),
                enc=_tflops(_flops_value(profile, "encoder")),
                dit=_tflops(_flops_value(profile, "transformer")),
                dec=_tflops(_flops_value(profile, "decoder")),
                e2e=_tflops(total),
                per_frame=_gflops_per_frame(profile),
                ratio=ratio,
                reduction=reduction,
            )
        )

    append_compute_row("Conditional teacher", teacher_params, teacher_profile)
    for model in models:
        params = model.get("parameters")
        profile = model.get("profile")
        if not isinstance(params, Mapping) or not isinstance(profile, Mapping):
            raise TypeError(f"Student compute profile is missing for {model.get('label')}")
        append_compute_row(str(model.get("label", "student")), params, profile)

    lines += [
        "",
        "### FLOP-counting note",
        "",
        "The table uses PyTorch operator-reported FLOPs. The same profiler, input shape, dtype and attention backend are used for teacher and students. If a fused/custom operator is unsupported by the counter, its FLOPs can be absent; therefore the report also records component-level counter errors in JSON. Do not use a row for quantitative claims if its DiT or E2E FLOP count is missing/zero.",
        "",
        "Stage-A student checkpoints share one prompt-free/no-time architecture, so init, step992 and long-run should have identical FLOPs. Their quality differences measure distillation progress rather than architectural compute changes.",
        "",
        "## Supplementary hardware profile",
        "",
        "| Model | Encoder ms | DiT ms | Decoder ms | E2E ms | Effective FPS | Peak GB |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| Conditional teacher | — | — | — | — | — | — |",
    ]
    for model in models:
        profile = model.get("profile", {})
        fps = profile.get("effective_fps") if isinstance(profile, Mapping) else None
        peak = profile.get("peak_allocated_gb") if isinstance(profile, Mapping) else None
        lines.append(
            "| {label} | {enc} | {dit} | {dec} | {e2e} | {fps} | {peak} |".format(
                label=model.get("label", "student"),
                enc=_latency_median(profile, "encoder_latency"),
                dit=_latency_median(profile, "transformer_latency"),
                dec=_latency_median(profile, "decoder_latency"),
                e2e=_latency_median(profile, "end_to_end_latency"),
                fps=("—" if not isinstance(fps, (int, float)) else f"{float(fps):.3f}"),
                peak=("—" if not isinstance(peak, (int, float)) else f"{float(peak):.3f}"),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_student_flops(report: Mapping[str, object]) -> None:
    models = report.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Audit report contains no student models")
    for model in models:
        profile = model.get("profile")
        if not isinstance(profile, Mapping):
            raise ValueError(f"Missing profile for {model.get('label')}")
        for component in ("encoder", "transformer", "decoder", "end_to_end"):
            value = _flops_value(profile, component)
            if value is None or value <= 0:
                raise ValueError(
                    f"Student {model.get('label')} lacks a usable {component} FLOP count. "
                    "Re-run audit_stage_a_distillation.py with --profile-flops before finalizing."
                )


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-A FLOP finalizer requires CUDA")
    audit_dir = args.audit_dir.expanduser().resolve()
    json_path = audit_dir / "stage_a_audit.json"
    markdown_path = audit_dir / "stage_a_audit.md"
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("Stage-A audit JSON must contain an object")
    _validate_student_flops(report)

    cache_value = report.get("teacher_cache")
    if not isinstance(cache_value, str):
        raise ValueError("Audit JSON does not contain teacher_cache")
    cache = TeacherVelocityCache(cache_value)

    device = torch.device("cuda")
    base_checkpoint_value = report.get("base_checkpoint")
    if not isinstance(base_checkpoint_value, str):
        raise ValueError("Audit JSON does not contain base_checkpoint")
    folded_config = audit.gate.validate_folded_checkpoint(
        Path(base_checkpoint_value),
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
    )
    dtype = audit.gate.resolve_runtime_dtype(
        args.dtype,
        folded_config,
        device,
        allow_mismatch=args.allow_dtype_mismatch,
    )

    dataset_args = argparse.Namespace(
        manifest=args.manifest,
        split=args.split,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        scale=args.scale,
        views_per_record=args.views_per_record,
        view_seed=args.view_seed,
        path_root=args.path_root,
        verify_paths=args.verify_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    dataset = audit.build_validation_dataset(dataset_args, cache)
    first_batch = next(iter(audit.build_loader(dataset, dataset_args)))

    reference_root, reae, transformer, prompt_embedding, prompt_key, timestep = _load_teacher(
        cache,
        device=device,
        dtype=dtype,
        attention_backend=args.attention_backend,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
    )
    print(f"profiling conditional teacher from {reference_root}", flush=True)
    profile = _profile_teacher(
        reae=reae,
        transformer=transformer,
        prompt_embedding=prompt_embedding,
        timestep=timestep,
        batch_cpu=first_batch,
        device=device,
        dtype=dtype,
    )
    for component in ("encoder", "transformer", "decoder", "end_to_end"):
        value = _flops_value(profile, component)
        if value is None or value <= 0:
            error = profile.get("flops", {}).get(component, {}).get("error")  # type: ignore[union-attr]
            raise RuntimeError(
                f"Conditional teacher {component} FLOP count is unusable: value={value}, error={error}"
            )

    teacher = report.get("teacher")
    if not isinstance(teacher, dict):
        teacher = {"kind": "cached_conditional_teacher", "metrics": {}}
        report["teacher"] = teacher
    teacher["reference_checkpoint"] = str(reference_root)
    teacher["prompt_key"] = prompt_key
    teacher["timestep"] = timestep
    teacher["parameters"] = _parameter_summary(reae, transformer)
    teacher["profile"] = profile
    teacher.pop("runtime_profile", None)

    report["compute_primary_metric"] = "operator_reported_flops"
    report["compute_input_contract"] = {
        "split": args.split,
        "clip_length": args.clip_length,
        "crop_size": args.crop_size,
        "scale": args.scale,
        "views_per_record": args.views_per_record,
        "view_seed": args.view_seed,
        "batch_size": args.batch_size,
        "dtype": str(dtype).removeprefix("torch."),
        "attention_backend": args.attention_backend,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, markdown_path)

    teacher_e2e = _flops_value(profile, "end_to_end")
    print(f"teacher E2E TFLOPs: {teacher_e2e / 1e12:.6f}", flush=True)
    for model in report["models"]:
        student_e2e = _flops_value(model["profile"], "end_to_end")
        ratio = student_e2e / teacher_e2e
        print(
            f"{model['label']}: E2E TFLOPs={student_e2e / 1e12:.6f}, "
            f"teacher_ratio={ratio:.6f}, reduction={(1-ratio)*100:.3f}%",
            flush=True,
        )
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
