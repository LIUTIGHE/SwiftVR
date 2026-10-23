#!/usr/bin/env python3
"""Profile Stage-B1 SwiftVR with the Tiny Conditional Decoder at steady state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profile_stage_a_streaming_macs import (
    _prepare_video_geometry,
    _run_until_counted_middle,
    count_params,
    parse_resolution,
)
from swiftvr.io import crop_spatial_padding_ntchw, preprocess_clip_uint8
from swiftvr.models import ReAE
from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.streaming import StreamingDiTPromptFreeNoTime, StreamingTAE
from swiftvr.streaming.tiny_decoder import StreamingTinyConditionalDecoder
from tools.runtime_macs import RuntimeMacCounter


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--tiny-decoder", type=Path, required=True)
    parser.add_argument("--resolution", type=parse_resolution, default=(1920, 1080))
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=24)
    parser.add_argument("--dit-overlap", type=int, default=0)
    parser.add_argument("--warmup-middle", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("sdpa",), default="sdpa")
    parser.add_argument("--reae-filename", default="reae.safetensors")
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument(
        "--stage-a-macs-json",
        type=Path,
        default=None,
        help="Optional Stage-A profile used to report decoder/total reduction.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/stage_b1_tiny_decoder_macs_1080p.json"),
    )
    return parser


class TinyDecoderStreamingSession:
    def __init__(
        self,
        *,
        reae: ReAE,
        transformer: WanTransformer3DModelPromptFreeNoTime,
        decoder: TinyConditionalDecoder,
        device: torch.device,
        dtype: torch.dtype,
        out_w: int,
        out_h: int,
        pad_w: int,
        pad_h: int,
        dit_overlap: int,
    ) -> None:
        self.reae_stream = StreamingTAE(reae)
        self.dit_stream = StreamingDiTPromptFreeNoTime(transformer, overlap=dit_overlap)
        self.decoder_stream = StreamingTinyConditionalDecoder(decoder)
        self.device = device
        self.dtype = dtype
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.pad_w = int(pad_w)
        self.pad_h = int(pad_h)
        self.reae_stream.reset()
        self.dit_stream.reset()
        self.decoder_stream.reset()

    @torch.inference_mode()
    def step(self, frames_uint8: torch.Tensor) -> torch.Tensor | None:
        frames = frames_uint8.to(self.device)
        condition = preprocess_clip_uint8(
            frames,
            self.out_h,
            self.out_w,
            "bilinear",
            self.pad_h,
            self.pad_w,
            self.dtype,
        )
        z_lq = self.reae_stream.encode_chunk(condition)
        if z_lq is None:
            return None
        z_bcfhw = z_lq.permute(0, 2, 1, 3, 4).contiguous()
        z_sr_bcfhw = self.dit_stream.denoise(z_bcfhw)
        z_sr = z_sr_bcfhw.permute(0, 2, 1, 3, 4).contiguous()
        rgb = self.decoder_stream.decode_chunk(z_sr, condition)
        return crop_spatial_padding_ntchw(rgb, self.pad_h, self.pad_w)


def _baseline_summary(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    student = payload.get("student")
    if not isinstance(student, dict):
        raise ValueError("Stage-A MAC JSON does not contain a student record")
    macs = student.get("macs")
    if not isinstance(macs, dict):
        raise ValueError("Stage-A student record does not contain MAC summary")
    roots = macs.get("by_root_gmacs_per_output_frame")
    if not isinstance(roots, dict):
        raise ValueError("Stage-A MAC JSON lacks per-root GMAC/frame")
    return {
        "encoder_gmacs_per_frame": float(roots["encoder"]),
        "transformer_gmacs_per_frame": float(roots["transformer"]),
        "decoder_gmacs_per_frame": float(roots["decoder"]),
        "total_gmacs_per_frame": float(macs["gmacs_per_output_frame"]),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.clip_len % 4:
        raise ValueError("clip-len must be divisible by four")
    if args.dit_overlap != 0:
        raise ValueError("Canonical Stage-B MAC comparison currently requires dit-overlap=0")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    geometry = _prepare_video_geometry(args)
    root = args.student_checkpoint.expanduser().resolve()

    reae = ReAE(str(root / args.reae_filename))
    transformer = WanTransformer3DModelPromptFreeNoTime.from_pretrained(
        str(root),
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tiny = TinyConditionalDecoder.from_pretrained(args.tiny_decoder, device="cpu")

    parameters = {
        "encoder_params": count_params(reae.encoder),
        "transformer_params": count_params(transformer),
        "decoder_params": count_params(tiny),
    }
    parameters["total_params"] = sum(parameters.values())

    reae.to(device=device, dtype=dtype).eval()
    transformer.to(device=device, dtype=dtype).eval()
    tiny.to(device=device, dtype=dtype).eval()
    transformer.prepare_for_inference(
        attention_backend=args.attention_backend,
        use_torch_compile=False,
    )

    session = TinyDecoderStreamingSession(
        reae=reae,
        transformer=transformer,
        decoder=tiny,
        device=device,
        dtype=dtype,
        out_w=int(geometry["out_w"]),
        out_h=int(geometry["out_h"]),
        pad_w=int(geometry["pad_w"]),
        pad_h=int(geometry["pad_h"]),
        dit_overlap=args.dit_overlap,
    )
    counter = RuntimeMacCounter()
    counter.add_module("encoder", reae.encoder)
    counter.add_module("transformer", transformer)
    counter.add_module("decoder", tiny)
    try:
        result = _run_until_counted_middle(
            args,
            geometry,
            session.step,
            counter,
            label="stage_b1_tiny_decoder",
            expect_cross_attention=False,
        )
    finally:
        counter.close()
    result["parameters"] = parameters

    macs = result["macs"]
    roots = macs["by_root_gmacs_per_output_frame"]
    baseline = _baseline_summary(args.stage_a_macs_json)
    comparison = None
    if baseline is not None:
        new_decoder = float(roots["decoder"])
        new_total = float(macs["gmacs_per_output_frame"])
        comparison = {
            "stage_a": baseline,
            "decoder_ratio": new_decoder / baseline["decoder_gmacs_per_frame"],
            "decoder_reduction_percent": 100.0
            * (1.0 - new_decoder / baseline["decoder_gmacs_per_frame"]),
            "total_ratio": new_total / baseline["total_gmacs_per_frame"],
            "total_reduction_percent": 100.0
            * (1.0 - new_total / baseline["total_gmacs_per_frame"]),
        }

    report = {
        "kind": "swiftvr_stage_b1_tiny_decoder_streaming_macs",
        "geometry": geometry,
        "student_checkpoint": str(root),
        "tiny_decoder": str(args.tiny_decoder.expanduser().resolve()),
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "mac_convention": "1 MAC = one multiply-accumulate; FLOP view uses 2 FLOPs/MAC",
        "profile": result,
        "comparison": comparison,
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\n========== Stage-B1 Tiny Decoder MACs ==========", flush=True)
    print(f"Encoder GMAC/frame : {float(roots['encoder']):.3f}")
    print(f"DiT GMAC/frame     : {float(roots['transformer']):.3f}")
    print(f"TinyDec GMAC/frame : {float(roots['decoder']):.3f}")
    print(f"Total GMAC/frame   : {float(macs['gmacs_per_output_frame']):.3f}")
    print(
        "GFLOPs/frame       : "
        f"{float(macs['gflops_per_output_frame_if_1mac_2flops']):.3f}"
    )
    print(f"Total params       : {int(parameters['total_params']):,}")
    if comparison is not None:
        print(f"Decoder reduction  : {comparison['decoder_reduction_percent']:.2f}%")
        print(f"Total reduction    : {comparison['total_reduction_percent']:.2f}%")
    print(f"Saved              : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
