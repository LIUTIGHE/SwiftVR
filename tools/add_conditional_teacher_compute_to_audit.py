#!/usr/bin/env python3
"""Add a measured conditional-teacher compute row to a Stage-A audit.

The Stage-A audit obtains teacher quality from the immutable offline velocity cache,
but cached targets cannot describe teacher runtime.  This companion profiler loads
the original conditional teacher, reconstructs the exact deterministic validation
view contract from the audit/cache metadata, measures the same encoder/DiT/decoder
and end-to-end paths used for student profiling, writes the results back into
``stage_a_audit.json``, and rewrites the Markdown compute table with a measured
``Conditional teacher`` row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

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
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help=(
            "Original conditional-teacher root. Defaults to the reference_checkpoint "
            "recorded in the immutable teacher cache metadata."
        ),
    )
    parser.add_argument("--teacher-transformer-subfolder", default="transformer")
    parser.add_argument("--path-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("audit", "float16", "bfloat16", "float32"),
        default="audit",
        help="Use the Stage-A audit dtype by default.",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Defaults to the backend recorded in the Stage-A audit.",
    )
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=10)
    parser.add_argument("--profile-flops", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _resolve_cache_file(
    value: object,
    *,
    teacher_root: Path,
    fallback_name: str,
) -> Path:
    if isinstance(value, str) and value:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = teacher_root / path
        path = path.resolve()
        if path.is_file():
            return path
        basename_fallback = (teacher_root / Path(value).name).resolve()
        if basename_fallback.is_file():
            return basename_fallback
    fallback = (teacher_root / fallback_name).resolve()
    if not fallback.is_file():
        raise FileNotFoundError(f"Unable to resolve teacher file: {fallback}")
    return fallback


def _resolve_dtype(report: Mapping[str, object], requested: str) -> torch.dtype:
    name = str(report.get("runtime_dtype", "bfloat16")) if requested == "audit" else requested
    if name not in audit.DTYPE_NAMES:
        raise ValueError(f"Unsupported runtime dtype in audit: {name!r}")
    return audit.DTYPE_NAMES[name]


def _validation_args(
    report: Mapping[str, object],
    cache: TeacherVelocityCache,
    *,
    path_root: Path | None,
) -> SimpleNamespace:
    metadata = cache.metadata
    manifests = report.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("Stage-A audit JSON is missing manifests")
    root_value = path_root if path_root is not None else Path(str(metadata.get("path_root", ".")))
    return SimpleNamespace(
        manifest=[Path(str(value)).expanduser().resolve() for value in manifests],
        split=str(metadata.get("split", "val")),
        clip_length=int(metadata.get("clip_length", 13)),
        crop_size=int(metadata.get("crop_size", 128)),
        scale=int(metadata.get("scale", 3)),
        views_per_record=int(metadata.get("views_per_record", 1)),
        view_seed=int(metadata.get("view_seed", 9000001)),
        path_root=root_value.expanduser().resolve(),
        verify_paths=False,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )


def _teacher_parameter_summary(reae: torch.nn.Module, transformer: torch.nn.Module) -> dict[str, int]:
    parameters = list(reae.parameters()) + list(transformer.parameters())
    return {
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": 0,
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
    }


def profile_conditional_teacher(
    report: Mapping[str, object],
    cache: TeacherVelocityCache,
    *,
    teacher_root: Path,
    transformer_subfolder: str,
    path_root: Path | None,
    device: torch.device,
    dtype: torch.dtype,
    attention_backend: str,
    latency_warmup: int,
    latency_repeats: int,
    profile_flops: bool,
) -> dict[str, object]:
    if latency_warmup < 0 or latency_repeats <= 0:
        raise ValueError("latency warmup must be non-negative and repeats positive")
    if device.type != "cuda":
        raise ValueError("Conditional-teacher profiling currently requires CUDA")

    validation_args = _validation_args(report, cache, path_root=path_root)
    dataset = audit.build_validation_dataset(validation_args, cache)
    first_batch = next(iter(audit.build_loader(dataset, validation_args)))
    batch = audit.gate.move_video_batch(first_batch, device=device, dtype=dtype)
    prepared = prepare_training_batch(batch)
    lq_input = prepared["lq_input"]
    target = prepared["target"]
    if not isinstance(lq_input, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("Profiling batch is missing lq_input/target")

    reae_path = _resolve_cache_file(
        cache.metadata.get("reae_file"),
        teacher_root=teacher_root,
        fallback_name="reae.safetensors",
    )
    prompt_path = _resolve_cache_file(
        cache.metadata.get("prompt_embedding_file"),
        teacher_root=teacher_root,
        fallback_name="prompt_embedding.safetensors",
    )
    prompt_key = str(cache.metadata.get("prompt_key", "prompt_emb"))
    timestep = float(cache.metadata.get("timestep", 1000.0))

    prompt_payload = load_file(str(prompt_path), device="cpu")
    if prompt_key not in prompt_payload:
        raise KeyError(f"{prompt_path} does not contain key {prompt_key!r}")
    prompt_embedding = prompt_payload[prompt_key].to(device=device, dtype=dtype)

    reae = ReAE(str(reae_path)).to(device=device, dtype=dtype).eval()
    transformer = WanTransformer3DModel.from_pretrained(
        str(teacher_root),
        subfolder=transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype).eval()
    transformer.prepare_for_inference(attention_backend=attention_backend)
    for module in (reae, transformer):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        z_ntchw = encode_reae_clip(reae, lq_input, require_4k_plus_1=True)
        z_lq = z_ntchw.permute(0, 2, 1, 3, 4).contiguous()
        prompt = expand_prompt_embedding(prompt_embedding, int(z_lq.shape[0])).to(
            device=device, dtype=dtype
        )
        timesteps = torch.full(
            (int(z_lq.shape[0]),), timestep, device=device, dtype=torch.float32
        )
        teacher_velocity = extract_transformer_sample(
            transformer(z_lq, timesteps, prompt, return_dict=True)
        )

    def encode_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            return encode_reae_clip(reae, lq_input, require_4k_plus_1=True)

    def transformer_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            current_prompt = expand_prompt_embedding(
                prompt_embedding, int(z_lq.shape[0])
            ).to(device=device, dtype=dtype)
            current_timesteps = torch.full(
                (int(z_lq.shape[0]),), timestep, device=device, dtype=torch.float32
            )
            return extract_transformer_sample(
                transformer(
                    z_lq,
                    current_timesteps,
                    current_prompt,
                    return_dict=True,
                )
            )

    def decode_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            z_teacher = z_lq - teacher_velocity
            return decode_reae_clip(
                reae,
                z_teacher.permute(0, 2, 1, 3, 4).contiguous(),
                output_frames=int(target.shape[1]),
                clamp=False,
            )

    def end_to_end_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            current_z_ntchw = encode_reae_clip(reae, lq_input, require_4k_plus_1=True)
            current_z = current_z_ntchw.permute(0, 2, 1, 3, 4).contiguous()
            current_prompt = expand_prompt_embedding(
                prompt_embedding, int(current_z.shape[0])
            ).to(device=device, dtype=dtype)
            current_timesteps = torch.full(
                (int(current_z.shape[0]),), timestep, device=device, dtype=torch.float32
            )
            current_velocity = extract_transformer_sample(
                transformer(
                    current_z,
                    current_timesteps,
                    current_prompt,
                    return_dict=True,
                )
            )
            current_z_teacher = current_z - current_velocity
            return decode_reae_clip(
                reae,
                current_z_teacher.permute(0, 2, 1, 3, 4).contiguous(),
                output_frames=int(target.shape[1]),
                clamp=False,
            )

    profile: dict[str, object] = {
        "input_shape": list(lq_input.shape),
        "latent_shape": list(z_lq.shape),
        "encoder_latency": audit._benchmark_cuda(
            encode_fn, warmup=latency_warmup, repeats=latency_repeats
        ),
        "transformer_latency": audit._benchmark_cuda(
            transformer_fn, warmup=latency_warmup, repeats=latency_repeats
        ),
        "decoder_latency": audit._benchmark_cuda(
            decode_fn, warmup=latency_warmup, repeats=latency_repeats
        ),
        "end_to_end_latency": audit._benchmark_cuda(
            end_to_end_fn, warmup=latency_warmup, repeats=latency_repeats
        ),
    }
    median_ms = float(profile["end_to_end_latency"]["median_ms"])  # type: ignore[index]
    frames = int(target.shape[1])
    profile["effective_fps"] = frames * 1000.0 / max(median_ms, 1e-9)

    torch.cuda.reset_peak_memory_stats(device)
    end_to_end_fn()
    torch.cuda.synchronize()
    profile["peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3

    if profile_flops:
        flops: dict[str, object] = {}
        for name, fn in (
            ("encoder", encode_fn),
            ("transformer", transformer_fn),
            ("decoder", decode_fn),
            ("end_to_end", end_to_end_fn),
        ):
            count, error = audit._count_flops(fn)
            flops[name] = {"reported_flops": count, "error": error}
        flops["note"] = (
            "Operator-reported FLOPs only; unsupported/custom/fused kernels can be absent. "
            "Use latency as the deployment-facing source of truth."
        )
        profile["flops"] = flops

    return {
        "kind": "measured_conditional_teacher",
        "checkpoint": str(teacher_root),
        "reae_file": str(reae_path),
        "prompt_embedding_file": str(prompt_path),
        "prompt_key": prompt_key,
        "timestep": timestep,
        "attention_backend": attention_backend,
        "parameters": _teacher_parameter_summary(reae, transformer),
        "runtime_profile": profile,
    }


def _format_flops(profile: Mapping[str, object]) -> str:
    flops = profile.get("flops")
    if not isinstance(flops, Mapping):
        return "—"
    end_to_end = flops.get("end_to_end")
    if not isinstance(end_to_end, Mapping):
        return "—"
    value = end_to_end.get("reported_flops")
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value) / 1e12:.3f}"


def _compute_row(label: str, parameters: Mapping[str, object], profile: Mapping[str, object], *, teacher: bool) -> str:
    trainable = "—" if teacher else f"{int(parameters['trainable_parameters']):,}"
    return (
        "| {label} | {params:,} | {trainable} | {enc:.3f} | {dit:.3f} | {dec:.3f} | "
        "{e2e:.3f} | {fps:.3f} | {peak:.3f} | {flops} |"
    ).format(
        label=label,
        params=int(parameters["total_parameters"]),
        trainable=trainable,
        enc=float(profile["encoder_latency"]["median_ms"]),  # type: ignore[index]
        dit=float(profile["transformer_latency"]["median_ms"]),  # type: ignore[index]
        dec=float(profile["decoder_latency"]["median_ms"]),  # type: ignore[index]
        e2e=float(profile["end_to_end_latency"]["median_ms"]),  # type: ignore[index]
        fps=float(profile["effective_fps"]),
        peak=float(profile["peak_allocated_gb"]),
        flops=_format_flops(profile),
    )


def rewrite_markdown_with_teacher_compute(report: Mapping[str, object], path: Path) -> None:
    """Reuse the quality table, then replace compute with student+teacher rows."""
    audit.write_markdown(report, path)
    text = path.read_text(encoding="utf-8")
    marker = "## Compute\n"
    if marker not in text:
        raise RuntimeError("Audit Markdown does not contain a Compute section")
    prefix = text.split(marker, 1)[0]

    models = report.get("models")
    teacher = report.get("teacher")
    if not isinstance(models, list) or not isinstance(teacher, Mapping):
        raise TypeError("Audit report is missing models/teacher")
    teacher_parameters = teacher.get("parameters")
    teacher_profile = teacher.get("runtime_profile")
    if not isinstance(teacher_parameters, Mapping) or not isinstance(teacher_profile, Mapping):
        raise TypeError("Teacher compute profile is missing")

    lines = [
        prefix.rstrip(),
        "",
        "## Compute",
        "",
        "| Model | Params | Trainable | Encoder ms | DiT ms | Decoder ms | E2E ms | Effective FPS | Peak GB | E2E TFLOPs* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        if not isinstance(model, Mapping):
            raise TypeError("Invalid model record in audit")
        parameters = model.get("parameters")
        profile = model.get("profile")
        if not isinstance(parameters, Mapping) or not isinstance(profile, Mapping):
            raise TypeError("Student compute profile is missing")
        lines.append(
            _compute_row(str(model.get("label", "student")), parameters, profile, teacher=False)
        )
    lines.append(
        _compute_row("Conditional teacher", teacher_parameters, teacher_profile, teacher=True)
    )
    lines += [
        "",
        "*E2E TFLOPs are operator-reported diagnostics. Fused/custom kernels may be under-counted; wall-clock latency is the deployment-facing source of truth.*",
        "",
        "All Stage-A student checkpoints share one prompt-free/no-time architecture, so their compute differences should be measurement noise unless structure changes.",
        "",
        "Conditional-teacher compute is measured by actually loading the original conditional Wan transformer and ReAE on the same deterministic validation batch; it is not inferred from the offline velocity cache.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    audit_json = args.audit_json.expanduser().resolve()
    report = _read_json(audit_json)
    cache_root = Path(str(report.get("teacher_cache", ""))).expanduser().resolve()
    cache = TeacherVelocityCache(cache_root)

    reference = cache.metadata.get("reference_checkpoint")
    if args.teacher_checkpoint is not None:
        teacher_root = args.teacher_checkpoint.expanduser().resolve()
    elif isinstance(reference, str) and reference:
        teacher_root = Path(reference).expanduser().resolve()
    else:
        raise ValueError(
            "Teacher cache lacks reference_checkpoint; pass --teacher-checkpoint explicitly"
        )
    if not teacher_root.is_dir():
        raise FileNotFoundError(
            f"Conditional teacher root does not exist: {teacher_root}; "
            "pass --teacher-checkpoint if the cache was moved between servers"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = _resolve_dtype(report, args.dtype)
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} does not support BF16")
    attention_backend = str(
        args.attention_backend
        if args.attention_backend is not None
        else report.get("attention_backend", cache.metadata.get("attention_backend", "sdpa"))
    )

    measured = profile_conditional_teacher(
        report,
        cache,
        teacher_root=teacher_root,
        transformer_subfolder=args.teacher_transformer_subfolder,
        path_root=args.path_root,
        device=device,
        dtype=dtype,
        attention_backend=attention_backend,
        latency_warmup=args.latency_warmup,
        latency_repeats=args.latency_repeats,
        profile_flops=args.profile_flops,
    )

    existing_teacher = report.get("teacher")
    teacher_record: dict[str, object] = dict(existing_teacher) if isinstance(existing_teacher, Mapping) else {}
    teacher_record.update(measured)
    report["teacher"] = teacher_record
    report["format_version"] = max(int(report.get("format_version", 1)), 2)

    temporary = audit_json.with_suffix(audit_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(audit_json)

    markdown = audit_json.with_name("stage_a_audit.md")
    rewrite_markdown_with_teacher_compute(report, markdown)
    print(
        json.dumps(
            {
                "audit_json": str(audit_json),
                "markdown": str(markdown),
                "teacher_parameters": teacher_record["parameters"],
                "teacher_profile": teacher_record["runtime_profile"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
