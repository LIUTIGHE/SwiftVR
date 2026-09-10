#!/usr/bin/env python3
"""Analytical MAC and optional CUDA latency profile for ReAE slim decoders."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftvr.models.reae_slim_decoder import (
    AGGRESSIVE_CHANNELS,
    M8_DECODER76_CHANNELS,
    SLIM100_CHANNELS,
    TEACHER_CHANNELS,
    SlimReAEDecoder,
)
from swiftvr.streaming import StreamingTAE


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

GROUP_COMPONENTS = {
    "input_conv": ("input_conv",),
    "stage0_memblocks": ("stage0_memblocks",),
    "transition01": ("tgrow01", "conv01"),
    "stage1_memblocks": ("stage1_memblocks",),
    "transition12": ("tgrow12", "conv12"),
    "stage2_memblocks": ("stage2_memblocks",),
    "transition23": ("tgrow23", "conv23"),
    "output_head": ("output_head",),
}

# Top-level decoder Sequential indices covered by each measured CUDA interval.
RUNTIME_GROUPS = {
    "input_conv": (0, 2),
    "stage0_memblocks": (3, 5),
    "transition01": (6, 8),
    "stage1_memblocks": (9, 11),
    "transition12": (12, 14),
    "stage2_memblocks": (15, 17),
    "transition23": (18, 20),
    "output_head": (21, 22),
}


def _parse_channels(value: str) -> tuple[int, int, int, int]:
    try:
        channels = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "channels must be four comma-separated integers"
        ) from exc
    if len(channels) != 4 or any(item <= 0 for item in channels):
        raise argparse.ArgumentTypeError("channels must contain four positive integers")
    return channels  # type: ignore[return-value]


def estimate_reae_decoder_macs(
    channels,
    *,
    output_height: int = 1088,
    output_width: int = 1920,
    latent_channels: int = 48,
    patch_size: int = 2,
):
    c0, c1, c2, c3 = (int(value) for value in channels)
    if output_height % 16 or output_width % 16:
        raise ValueError("output geometry must be divisible by 16")
    h0, w0 = output_height // 16, output_width // 16
    h1, w1 = output_height // 8, output_width // 8
    h2, w2 = output_height // 4, output_width // 4
    h3, w3 = output_height // 2, output_width // 2

    values = {
        "input_conv": h0 * w0 * latent_channels * c0 * 9 * 0.25,
        "stage0_memblocks": h0 * w0 * 108 * c0 * c0 * 0.25,
        "tgrow01": h1 * w1 * c0 * c0 * 0.25,
        "conv01": h1 * w1 * 9 * c0 * c1 * 0.25,
        "stage1_memblocks": h1 * w1 * 108 * c1 * c1 * 0.25,
        "tgrow12": h2 * w2 * 3 * c1 * c1 * 0.5,
        "conv12": h2 * w2 * 9 * c1 * c2 * 0.5,
        "stage2_memblocks": h2 * w2 * 108 * c2 * c2 * 0.5,
        "tgrow23": h3 * w3 * 3 * c2 * c2,
        "conv23": h3 * w3 * 9 * c2 * c3,
        "output_head": h3 * w3 * 9 * c3 * (3 * patch_size**2),
    }
    total = sum(values.values())
    components = {key: value / 1e9 for key, value in values.items()}
    grouped = {
        group: sum(values[name] for name in names) / 1e9
        for group, names in GROUP_COMPONENTS.items()
    }
    return {
        "channels": [c0, c1, c2, c3],
        "components_gmac": components,
        "components_percent": {
            key: 100.0 * value / total for key, value in values.items()
        },
        "groups_gmac": grouped,
        "groups_percent": {
            key: 100.0 * value / (total / 1e9) for key, value in grouped.items()
        },
        "total_gmac": total / 1e9,
        "total_gflops_2flop_per_mac": 2.0 * total / 1e9,
    }


class _GroupTimer:
    def __init__(self, decoder: torch.nn.Sequential):
        self.active = False
        self.starts = {}
        self.pairs = {name: [] for name in RUNTIME_GROUPS}
        self.handles = []
        for name, (first, last) in RUNTIME_GROUPS.items():
            self.handles.append(
                decoder[first].register_forward_pre_hook(self._pre(name))
            )
            self.handles.append(
                decoder[last].register_forward_hook(self._post(name))
            )

    def _pre(self, name):
        def hook(_module, _inputs):
            if self.active:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.starts[name] = event
        return hook

    def _post(self, name):
        def hook(_module, _inputs, _output):
            if self.active:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.pairs[name].append((self.starts.pop(name), end))
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

    def means_ms(self):
        return {
            name: statistics.mean(start.elapsed_time(end) for start, end in pairs)
            for name, pairs in self.pairs.items()
        }


@torch.inference_mode()
def profile_decoder_latency(
    checkpoint: Path,
    *,
    output_height: int,
    output_width: int,
    clip_len: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("runtime decoder latency profiling requires CUDA")
    if clip_len <= 0 or clip_len % 4:
        raise ValueError("clip-len must be a positive multiple of 4")
    if output_height % 16 or output_width % 16:
        raise ValueError("runtime output geometry must be divisible by 16")

    model = SlimReAEDecoder.from_pretrained(checkpoint, device="cpu")
    model.to(device=device, dtype=dtype).eval()
    z = torch.randn(
        1,
        clip_len // 4,
        model.latent_channels,
        output_height // 16,
        output_width // 16,
        device=device,
        dtype=dtype,
    )
    stream = StreamingTAE(model)
    stream.reset()

    # Establish boundary state and consume the first-decode trim before timing.
    stream.decode_chunk(z)
    for _ in range(warmup):
        stream.decode_chunk(z)
    torch.cuda.synchronize(device)

    timer = _GroupTimer(model.decoder)
    totals = []
    try:
        timer.active = True
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = stream.decode_chunk(z)
            end.record()
            totals.append((start, end, int(output.shape[1])))
        timer.active = False
        torch.cuda.synchronize(device)
    finally:
        timer.close()

    total_ms = [start.elapsed_time(end) for start, end, _ in totals]
    emitted = [frames for _, _, frames in totals]
    if len(set(emitted)) != 1:
        raise RuntimeError(f"inconsistent output frames across repeats: {emitted}")
    frames = emitted[0]
    group_ms = timer.means_ms()
    mean_total = statistics.mean(total_ms)
    group_sum = sum(group_ms.values())
    return {
        "checkpoint": str(checkpoint),
        "channels": list(model.channels),
        "dtype": str(dtype),
        "clip_len": clip_len,
        "latent_frames": clip_len // 4,
        "emitted_frames_per_repeat": frames,
        "state_prefill_chunks": 1,
        "warmup": warmup,
        "repeat": repeat,
        "mean_ms_per_chunk": mean_total,
        "mean_ms_per_output_frame": mean_total / frames,
        "fps_decoder_only": 1000.0 * frames / mean_total,
        "groups_ms_per_chunk": group_ms,
        "groups_ms_per_output_frame": {
            key: value / frames for key, value in group_ms.items()
        },
        "groups_percent_of_decoder_latency": {
            key: 100.0 * value / mean_total for key, value in group_ms.items()
        },
        "unattributed_wrapper_ms_per_chunk": mean_total - group_sum,
        "note": (
            "Group CUDA events cover grouped top-level decoder intervals. "
            "Unattributed time includes state setup at interval entries, "
            "PixelShuffle/clamp and outer Python wrapper work."
        ),
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--height", type=int, default=1088)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument(
        "--channels",
        type=_parse_channels,
        action="append",
        default=[],
        help="Optional custom C0,C1,C2,C3 widths; repeat for several candidates.",
    )
    p.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=None,
        help="Optional SlimReAEDecoder checkpoint for measured CUDA latency.",
    )
    p.add_argument("--clip-len", type=int, default=24)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >=0 and --repeat must be >0")

    result = {
        "teacher": estimate_reae_decoder_macs(
            TEACHER_CHANNELS, output_height=args.height, output_width=args.width
        ),
        "slim100": estimate_reae_decoder_macs(
            SLIM100_CHANNELS, output_height=args.height, output_width=args.width
        ),
        "aggressive": estimate_reae_decoder_macs(
            AGGRESSIVE_CHANNELS, output_height=args.height, output_width=args.width
        ),
        "m8_decoder76": estimate_reae_decoder_macs(
            M8_DECODER76_CHANNELS,
            output_height=args.height,
            output_width=args.width,
        ),
    }
    if args.channels:
        result["custom"] = [
            estimate_reae_decoder_macs(
                channels, output_height=args.height, output_width=args.width
            )
            for channels in args.channels
        ]

    if args.decoder_checkpoint is not None:
        runtime = profile_decoder_latency(
            args.decoder_checkpoint.expanduser().resolve(),
            output_height=args.height,
            output_width=args.width,
            clip_len=args.clip_len,
            device=torch.device(args.device),
            dtype=DTYPES[args.dtype],
            warmup=args.warmup,
            repeat=args.repeat,
        )
        result["runtime"] = runtime
        result["runtime_analytic"] = estimate_reae_decoder_macs(
            runtime["channels"],
            output_height=args.height,
            output_width=args.width,
        )

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        path = args.output_json.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
