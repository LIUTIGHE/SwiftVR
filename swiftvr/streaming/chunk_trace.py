"""Opt-in observations of the canonical fixed-chunk runner; no alternate forward.

Used only for paired chunk-length diagnostics. CPU copies and file writes make
traced runs unsuitable for latency benchmarking. A complete trace is published
only after restore_video has successfully written all PNG frames.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def _pngs(path: Path) -> list[Path]:
    def key(p):
        return (0, int(p.stem)) if p.stem.isdigit() else (1, p.name)
    return sorted(path.glob("*.png"), key=key)


def _input_digest(path: Path) -> str:
    """Bind runs to identical input bytes/order, not just the same directory name."""
    files = _pngs(path) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No input PNGs: {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8") + b"\0")
        with file.open("rb") as handle:
            for data in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(data)
    return digest.hexdigest()


class ChunkTrace:
    def __init__(self, pipe, directory, args):
        if not args.png or args.dit_overlap != 0:
            raise ValueError("--chunk-trace requires --png and --dit-overlap 0")
        self.root = Path(directory).expanduser().resolve()
        self.output = Path(args.output).expanduser().resolve()
        # Reject stale frames before any model forward.
        for path in (self.root, self.output):
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f"Use a fresh diagnostic directory: {path}")
        self.source = Path(args.input).expanduser().resolve()
        self.digest = _input_digest(self.source)
        self.args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.latent_cursor = self.output_cursor = 0
        self.pending = None
        self.trim = int(pipe.reae.frames_to_trim)
        tae = pipe.tae_stream
        encode, decode = tae.encode_chunk_fixed, tae.decode_chunk_fixed

        def traced_encode(x, spec):
            if self.pending is not None:
                raise RuntimeError("Unpaired encode/decode in chunk trace")
            z = encode(x, spec)
            if not isinstance(z, torch.Tensor) or z.ndim != 5:
                raise ValueError("Expected encoder latent [N,T,C,H,W]")
            self.pending = (int(spec.clip_idx), z.detach().cpu().clone(), list(x.shape))
            return z

        def traced_decode(z, spec):
            if self.pending is None or self.pending[0] != int(spec.clip_idx):
                raise RuntimeError("Decode does not match traced encoder chunk")
            _, z_lq, input_shape = self.pending
            z_sr = z.detach().cpu().clone()
            if z_lq.shape != z_sr.shape:
                raise ValueError("Encoder/restored latent shape mismatch")
            rgb = decode(z, spec)
            count = 0 if rgb is None else int(rgb.shape[1])
            latent_count = int(z.shape[1])
            filename = f"chunk_{int(spec.clip_idx):05d}.safetensors"
            save_file({"z_lq": z_lq.contiguous(), "z_sr": z_sr.contiguous()}, str(self.root / filename))
            row = {
                "chunk_index": int(spec.clip_idx), "chunk_type": spec.ctype.value,
                "input_start": int(spec.frame_start), "input_count": int(spec.frame_count),
                "input_shape": input_shape, "latent_start": self.latent_cursor,
                "latent_count": latent_count, "latent_shape": list(z.shape),
                "output_start": self.output_cursor, "output_count": count,
                "output_shape_before_spatial_crop": None if rgb is None else list(rgb.shape),
                "is_first_decode": bool(spec.is_first_decode), "tensor_file": filename,
            }
            self.rows.append(row)
            self.latent_cursor += latent_count
            self.output_cursor += count
            self.pending = None
            print(f"[chunk-trace] {row['chunk_type']} input=[{row['input_start']},"
                  f"{row['input_start'] + row['input_count']}) latent=[{row['latent_start']},"
                  f"{self.latent_cursor}) output=[{row['output_start']},{self.output_cursor})", flush=True)
            return rgb

        tae.encode_chunk_fixed = traced_encode
        tae.decode_chunk_fixed = traced_decode

    def finish(self, stats):
        paths = _pngs(self.output)
        if self.pending is not None or not self.rows:
            raise RuntimeError("Incomplete trace")
        if len(paths) != self.output_cursor or int(stats["frames"]) != self.output_cursor:
            raise RuntimeError("Written PNG count does not match actual decoder emissions")
        input_count = sum(row["input_count"] for row in self.rows)
        if input_count != self.output_cursor:
            raise RuntimeError("Fixed-chunk input/output frame counts differ")
        report = {
            "kind": "swiftvr_chunk_trace_v1", "complete": True,
            "input": str(self.source), "input_sha256": self.digest,
            "output": str(self.output), "png_files": [p.name for p in paths],
            "arguments": self.args, "frames_to_trim": self.trim,
            "frames": self.output_cursor, "latent_frames": self.latent_cursor,
            "chunks": self.rows,
            "notes": ["Observed canonical encode/decode calls; no replacement forward.",
                      "Input origin and 4-frame grouping must match across compared runs.",
                      "Trace CPU copies/I/O invalidate inference timing comparisons.",
                      "PNG filenames preserve canonical output order, not a PTS re-alignment."],
        }
        tmp = self.root / "trace.json.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.root / "trace.json")
        return report
