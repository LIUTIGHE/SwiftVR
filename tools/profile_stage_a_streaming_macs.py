#!/usr/bin/env python3
"""Profile steady-state Stage-A SwiftVR compute on one streaming MIDDLE chunk.

The profiler uses the real SwiftVR streaming ReAE path and counts the actually
executed Linear/Conv/attention MACs. Conditional-teacher and prompt-free/no-time
student architectures are measured on the same input chunks and output geometry.
Student delta checkpoints do not change architecture/MACs, so the prompt-free
architecture is profiled once and reused for init/step992/long-run audit rows.

Primary reporting convention:
  * GMACs per emitted output frame;
  * optional FLOP conversion uses 1 MAC = 2 FLOPs.

The canonical Stage-A setting uses SDPA and ``dit_overlap=0`` so attention MACs are
backend-independent and directly comparable across teacher/student architectures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr import SwiftVRPipeline
from swiftvr.io import (
    crop_spatial_padding_ntchw,
    get_video_info,
    iter_video_clips_fixed_scheme,
    preprocess_clip_uint8,
)
from swiftvr.models import ReAE
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.streaming import StreamingTAE
from swiftvr.streaming.chunk import ChunkType
from swiftvr.training.reference import extract_transformer_sample
from tools.runtime_macs import RuntimeMacCounter


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_resolution(value: str) -> tuple[int, int]:
    width, separator, height = value.lower().partition("x")
    if not separator:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    try:
        width_i, height_i = int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must contain integers") from exc
    if width_i <= 0 or height_i <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width_i, height_i


def aligned_pad(size: int, multiple: int = 32) -> int:
    return (multiple - int(size) % multiple) % multiple


def count_params(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def canonical_parameter_summary(reae: ReAE, transformer: torch.nn.Module) -> dict[str, int]:
    # Call this BEFORE prepare_for_inference(): Wan attention preparation creates
    # fused projection copies while retaining the serialized Q/K/V modules.
    encoder = count_params(reae.encoder)
    decoder = count_params(reae.decoder)
    dit = count_params(transformer)
    return {
        "encoder_params": encoder,
        "transformer_params": dit,
        "decoder_params": decoder,
        "total_params": encoder + dit + decoder,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--student-checkpoint", required=True, type=Path)
    parser.add_argument("--resolution", type=parse_resolution, default=(1920, 1080))
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=24)
    parser.add_argument("--dit-overlap", type=int, default=0)
    parser.add_argument("--warmup-middle", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa",),
        default="sdpa",
        help="Canonical MAC counting requires SDPA so self-attention is observable.",
    )
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/stage_a_streaming_macs.json"),
    )
    return parser


def _prepare_video_geometry(args: argparse.Namespace) -> dict[str, object]:
    raw_total, lq_h, lq_w, src_fps = get_video_info(args.input, fallback_fps=30)
    total_frames = 4 * ((int(raw_total) - 1) // 4) + 1
    out_w, out_h = args.resolution
    pad_h = aligned_pad(out_h, 32)
    pad_w = aligned_pad(out_w, 32)
    return {
        "raw_total_frames": int(raw_total),
        "total_frames": int(total_frames),
        "lq_h": int(lq_h),
        "lq_w": int(lq_w),
        "source_fps": float(src_fps),
        "out_w": int(out_w),
        "out_h": int(out_h),
        "pad_w": int(pad_w),
        "pad_h": int(pad_h),
        "compute_w": int(out_w + pad_w),
        "compute_h": int(out_h + pad_h),
    }


def _validate_counter_summary(
    summary: dict[str, object],
    *,
    expect_cross_attention: bool,
    label: str,
) -> None:
    if int(summary.get("total_macs", 0)) <= 0:
        raise RuntimeError(f"{label} produced zero MACs")
    errors = summary.get("count_errors")
    if isinstance(errors, list) and errors:
        raise RuntimeError(f"{label} MAC counting diagnostics are non-empty: {errors}")
    calls = summary.get("calls_by_type")
    if not isinstance(calls, dict):
        raise RuntimeError(f"{label} MAC summary lacks calls_by_type")
    if int(calls.get("self_attn_qk", 0)) <= 0:
        raise RuntimeError(f"{label} self-attention was not observed by the MAC counter")
    cross_calls = int(calls.get("cross_attn_qk", 0))
    if expect_cross_attention and cross_calls <= 0:
        raise RuntimeError(
            "Conditional teacher cross-attention was not observed; refusing to "
            "report an under-counted Stage-A teacher MAC value"
        )
    if not expect_cross_attention and cross_calls != 0:
        raise RuntimeError(
            f"Prompt-free student unexpectedly executed {cross_calls} cross-attention calls"
        )
    by_root = summary.get("by_root_gmacs")
    if not isinstance(by_root, dict):
        raise RuntimeError(f"{label} MAC summary lacks by_root_gmacs")
    for root in ("encoder", "transformer", "decoder"):
        if float(by_root.get(root, 0.0)) <= 0.0:
            raise RuntimeError(f"{label} did not record positive {root} MACs")


def _run_until_counted_middle(
    args: argparse.Namespace,
    geometry: dict[str, object],
    step_fn: Callable[[torch.Tensor], torch.Tensor | None],
    counter: RuntimeMacCounter,
    *,
    label: str,
    expect_cross_attention: bool,
) -> dict[str, object]:
    middle_seen = 0
    for spec, lq_chunk in iter_video_clips_fixed_scheme(
        args.input,
        clip_len=args.clip_len,
        total_frames=int(geometry["total_frames"]),
        crop_h=int(geometry["lq_h"]),
        crop_w=int(geometry["lq_w"]),
    ):
        is_middle = spec.ctype == ChunkType.MIDDLE
        if is_middle and middle_seen >= args.warmup_middle:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            with counter.count(reset=True):
                output = step_fn(lq_chunk)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            emitted = 0 if output is None else int(output.shape[1])
            if emitted <= 0:
                raise RuntimeError(f"{label} counted MIDDLE chunk emitted no frames")
            summary = counter.summary(emitted_frames=emitted)
            _validate_counter_summary(
                summary,
                expect_cross_attention=expect_cross_attention,
                label=label,
            )
            return {
                "label": label,
                "counted_chunk_type": spec.ctype.value,
                "counted_clip_idx": int(spec.clip_idx),
                "frame_start": int(spec.frame_start),
                "input_frames": int(lq_chunk.shape[0]),
                "output_frames": emitted,
                "step_seconds": elapsed,
                "step_fps": emitted / max(elapsed, 1e-12),
                "macs": summary,
                "by_module_gmacs": {
                    key: value / 1e9
                    for key, value in sorted(counter.macs_by_name.items())
                },
            }

        output = step_fn(lq_chunk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        emitted = 0 if output is None else int(output.shape[1])
        print(
            f"[{label}] warmup {spec.ctype.value:6s} clip={spec.clip_idx} "
            f"in={int(lq_chunk.shape[0])}f out={emitted}f",
            flush=True,
        )
        if is_middle:
            middle_seen += 1

    raise RuntimeError(
        f"{label}: no suitable MIDDLE chunk found; use a longer input or "
        "--warmup-middle 0"
    )


def _profile_teacher(
    args: argparse.Namespace,
    geometry: dict[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    pipe = SwiftVRPipeline.from_pretrained(
        args.teacher_checkpoint,
        reae_filename=args.reae_filename,
        transformer_subfolder=args.transformer_subfolder,
    )
    parameters = canonical_parameter_summary(pipe.reae, pipe.transformer)
    pipe.to(
        device,
        dtype=dtype,
        attention_backend=args.attention_backend,
        torch_compile=False,
    )
    session = pipe.stream(
        clip_len=args.clip_len,
        resolution=args.resolution,
        upscale=args.upscale,
        dit_overlap=args.dit_overlap,
    )
    counter = RuntimeMacCounter()
    counter.add_module("encoder", pipe.reae.encoder)
    counter.add_module("transformer", pipe.transformer)
    counter.add_module("decoder", pipe.reae.decoder)
    try:
        result = _run_until_counted_middle(
            args,
            geometry,
            session.step,
            counter,
            label="conditional_teacher",
            expect_cross_attention=True,
        )
    finally:
        counter.close()
    result["parameters"] = parameters
    return result


class PromptFreeMiddleSession:
    """Minimal streaming wrapper used only for architecture MAC profiling.

    ReAE state follows the real StreamingTAE path. The prompt-free/no-time DiT is
    applied once to each emitted latent chunk. With ``dit_overlap=0`` its MACs are
    identical to a dedicated streaming wrapper; the temporal RoPE offset changes
    values but not operation count, so no quality claim is made from this helper.
    """

    def __init__(
        self,
        reae: ReAE,
        transformer: WanTransformer3DModelPromptFreeNoTime,
        *,
        device: torch.device,
        dtype: torch.dtype,
        out_w: int,
        out_h: int,
        pad_w: int,
        pad_h: int,
        upscale_mode: str = "bilinear",
    ):
        self.reae = reae
        self.transformer = transformer
        self.tae = StreamingTAE(reae)
        self.device = device
        self.dtype = dtype
        self.out_w = out_w
        self.out_h = out_h
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.upscale_mode = upscale_mode
        self.tae.reset()

    @torch.inference_mode()
    def step(self, frames_uint8: torch.Tensor) -> torch.Tensor | None:
        frames = frames_uint8.to(self.device)
        clip = preprocess_clip_uint8(
            frames,
            self.out_h,
            self.out_w,
            self.upscale_mode,
            self.pad_h,
            self.pad_w,
            self.dtype,
        )
        z = self.tae.encode_chunk(clip)
        if z is None:
            return None
        z_bcfhw = z.permute(0, 2, 1, 3, 4).contiguous()
        velocity = extract_transformer_sample(
            self.transformer(z_bcfhw, return_dict=True)
        )
        if velocity.shape != z_bcfhw.shape:
            raise RuntimeError(
                f"Prompt-free velocity shape mismatch: {tuple(velocity.shape)} vs "
                f"{tuple(z_bcfhw.shape)}"
            )
        z_den = (z_bcfhw - velocity).permute(0, 2, 1, 3, 4).contiguous()
        rgb = self.tae.decode_chunk(z_den)
        return crop_spatial_padding_ntchw(rgb, self.pad_h, self.pad_w)


def _profile_student(
    args: argparse.Namespace,
    geometry: dict[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    root = args.student_checkpoint.expanduser().resolve()
    reae = ReAE(str(root / args.reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    parameters = canonical_parameter_summary(reae, transformer)
    reae.to(device=device, dtype=dtype).eval()
    transformer.to(device=device, dtype=dtype).eval()
    transformer.prepare_for_inference(
        attention_backend=args.attention_backend,
        use_torch_compile=False,
    )
    session = PromptFreeMiddleSession(
        reae,
        transformer,
        device=device,
        dtype=dtype,
        out_w=int(geometry["out_w"]),
        out_h=int(geometry["out_h"]),
        pad_w=int(geometry["pad_w"]),
        pad_h=int(geometry["pad_h"]),
    )
    counter = RuntimeMacCounter()
    counter.add_module("encoder", reae.encoder)
    counter.add_module("transformer", transformer)
    counter.add_module("decoder", reae.decoder)
    try:
        result = _run_until_counted_middle(
            args,
            geometry,
            session.step,
            counter,
            label="prompt_free_no_time_student",
            expect_cross_attention=False,
        )
    finally:
        counter.close()
    result["parameters"] = parameters
    return result


def _print_model_summary(record: dict[str, object]) -> None:
    macs = record["macs"]
    assert isinstance(macs, dict)
    roots = macs["by_root_gmacs_per_output_frame"]
    assert isinstance(roots, dict)
    print(f"\n========== {record['label']} ==========", flush=True)
    print(f"Output frames          : {record['output_frames']}")
    print(f"Encoder GMAC/frame     : {float(roots['encoder']):.3f}")
    print(f"DiT GMAC/frame         : {float(roots['transformer']):.3f}")
    print(f"Decoder GMAC/frame     : {float(roots['decoder']):.3f}")
    print(f"Total GMAC/frame       : {float(macs['gmacs_per_output_frame']):.3f}")
    print(
        "GFLOPs/frame (2/MAC) : "
        f"{float(macs['gflops_per_output_frame_if_1mac_2flops']):.3f}"
    )
    print(f"Params                 : {int(record['parameters']['total_params']):,}")
    print("==========================================", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_len <= 0 or args.clip_len % 4 != 0:
        raise ValueError(f"--clip-len must be a positive multiple of 4, got {args.clip_len}")
    if args.upscale <= 0:
        raise ValueError("--upscale must be positive")
    if args.dit_overlap != 0:
        raise ValueError(
            "Canonical Stage-A MAC comparison currently requires --dit-overlap 0"
        )
    if args.warmup_middle < 0:
        raise ValueError("--warmup-middle must be non-negative")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")

    args.input = args.input.expanduser().resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    args.student_checkpoint = args.student_checkpoint.expanduser().resolve()
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    geometry = _prepare_video_geometry(args)

    print("========== Stage-A streaming MAC profile ==========")
    print(f"Input               : {args.input}")
    print(f"Teacher checkpoint  : {args.teacher_checkpoint}")
    print(f"Student checkpoint  : {args.student_checkpoint}")
    print(f"Target resolution   : {geometry['out_w']}x{geometry['out_h']}")
    print(f"Internal compute    : {geometry['compute_w']}x{geometry['compute_h']}")
    print(f"Clip length         : {args.clip_len}")
    print(f"Warmup MIDDLE       : {args.warmup_middle}")
    print(f"Attention backend   : {args.attention_backend}")
    print(f"MAC convention      : 1 MAC = multiply-accumulate; FLOP view uses 2/MAC")
    print("===================================================", flush=True)

    teacher = _profile_teacher(args, geometry, device=device, dtype=dtype)
    _print_model_summary(teacher)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    student = _profile_student(args, geometry, device=device, dtype=dtype)
    _print_model_summary(student)

    teacher_macs = float(teacher["macs"]["gmacs_per_output_frame"])
    student_macs = float(student["macs"]["gmacs_per_output_frame"])
    ratio = student_macs / teacher_macs
    reduction = 1.0 - ratio

    report = {
        "format_version": 1,
        "kind": "swiftvr_stage_a_streaming_macs",
        "input": str(args.input),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "student_checkpoint": str(args.student_checkpoint),
        "target_resolution": [int(geometry["out_w"]), int(geometry["out_h"])],
        "internal_compute_resolution": [
            int(geometry["compute_w"]),
            int(geometry["compute_h"]),
        ],
        "clip_len": int(args.clip_len),
        "dit_overlap": int(args.dit_overlap),
        "warmup_middle": int(args.warmup_middle),
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "geometry": geometry,
        "teacher": teacher,
        "student": student,
        "student_over_teacher_macs": ratio,
        "student_macs_reduction_percent": 100.0 * reduction,
        "reporting_note": (
            "Primary compute metric is steady-state model GMACs per emitted MIDDLE-chunk "
            "output frame. GFLOPs are shown only under the explicit 1 MAC = 2 FLOPs "
            "convention. Latency is supplementary and hardware-dependent."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== Stage-A reduction ==========")
    print(f"Teacher GMAC/frame      : {teacher_macs:.3f}")
    print(f"Student GMAC/frame      : {student_macs:.3f}")
    print(f"Student / Teacher       : {ratio:.4f}x")
    print(f"MAC reduction           : {100.0 * reduction:.2f}%")
    print(f"Saved                   : {args.output_json}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
