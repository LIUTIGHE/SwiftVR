from __future__ import annotations

import contextlib
import functools
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _numel_shape(shape) -> int:
    value = 1
    for item in shape:
        value *= int(item)
    return int(value)


class RuntimeMacCounter:
    """Runtime multiply-accumulate counter for SwiftVR inference.

    The counter follows the actually executed prepared inference graph rather than
    the serialized parameter tree. It counts the dominant dense neural-network
    operations used by SwiftVR:

    * ``nn.Linear``;
    * ``nn.Conv1d`` / ``nn.Conv2d`` / ``nn.Conv3d``;
    * PyTorch SDPA used by SwiftVR shifted-window self-attention;
    * SwiftVR ``dispatch_attention_fn`` used by conditional cross-attention.

    The following are deliberately excluded from the MAC convention: bias adds,
    elementwise arithmetic, normalization, activations, RoPE, indexing/gather/
    scatter, reshape/concat, interpolation, pixel shuffle/unshuffle, image/video
    I/O and CPU preprocessing.

    ``RuntimeMacCounter`` is intended for single-process profiling. Its attention
    instrumentation temporarily monkey-patches process-global Python call sites,
    so concurrent counters in the same process are unsupported.
    """

    def __init__(self):
        self.macs_by_name = defaultdict(int)
        self.macs_by_type = defaultdict(int)
        self.calls_by_type = defaultdict(int)
        self.handles = []
        self._seen_modules = set()
        self._orig_sdpa = None
        self._swiftvr_transformer_module = None
        self._orig_dispatch_attention = None
        self._dispatch_depth = 0
        self.enabled = False
        self.count_errors: list[str] = []

    def reset(self) -> None:
        self.macs_by_name.clear()
        self.macs_by_type.clear()
        self.calls_by_type.clear()
        self.count_errors.clear()

    def add_module(self, root_name: str, module: nn.Module) -> None:
        """Register Linear/Conv hooks recursively below one reporting root.

        Register disjoint roots such as ``encoder``, ``transformer`` and
        ``decoder``. Re-registering an already seen module object is ignored so a
        module cannot be double counted accidentally.
        """

        normalized_root = str(root_name).strip()
        if not normalized_root:
            raise ValueError("root_name must be non-empty")

        for name, child in module.named_modules():
            if id(child) in self._seen_modules:
                continue
            self._seen_modules.add(id(child))
            full_name = normalized_root if name == "" else f"{normalized_root}.{name}"

            if isinstance(child, nn.Linear):
                self.handles.append(
                    child.register_forward_hook(
                        functools.partial(self._linear_hook, full_name)
                    )
                )
            elif isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self.handles.append(
                    child.register_forward_hook(
                        functools.partial(self._conv_hook, full_name)
                    )
                )

    def _add(self, name: str, op_type: str, macs: int) -> None:
        if not self.enabled:
            return
        macs = int(macs)
        if macs < 0:
            raise ValueError(f"MAC count must be non-negative, got {macs}")
        self.macs_by_name[name] += macs
        self.macs_by_type[op_type] += macs
        self.calls_by_type[op_type] += 1

    def _linear_hook(self, name: str, module: nn.Linear, inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            self.count_errors.append(f"Linear output is not a tensor: {name}")
            return
        # One MAC per input feature for every output element. Bias adds are not
        # part of the MAC convention.
        self._add(name, "linear", output.numel() * int(module.in_features))

    def _conv_hook(self, name: str, module: nn.modules.conv._ConvNd, inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            self.count_errors.append(f"Conv output is not a tensor: {name}")
            return

        batch = int(output.shape[0])
        out_channels = int(output.shape[1])
        output_spatial = _numel_shape(output.shape[2:])
        kernel_spatial = _numel_shape(module.kernel_size)
        in_channels_per_group = int(module.in_channels) // int(module.groups)
        macs = (
            batch
            * out_channels
            * output_spatial
            * in_channels_per_group
            * kernel_spatial
        )

        if isinstance(module, nn.Conv1d):
            op_type = "conv1d"
        elif isinstance(module, nn.Conv2d):
            op_type = "conv2d"
        elif isinstance(module, nn.Conv3d):
            op_type = "conv3d"
        else:
            op_type = "conv"
        self._add(name, op_type, macs)

    def _count_sdpa(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        # PyTorch SDPA layout is [..., H, L, D] in SwiftVR after the explicit
        # transpose in _dense_attn(). More generally all leading dimensions
        # before L,D are independent attention batches/heads.
        q_shape = tuple(query.shape)
        k_shape = tuple(key.shape)
        v_shape = tuple(value.shape)
        if len(q_shape) < 3 or len(k_shape) < 3 or len(v_shape) < 3:
            raise ValueError(
                f"Unexpected SDPA shapes: q={q_shape}, k={k_shape}, v={v_shape}"
            )
        length_q = int(q_shape[-2])
        length_k = int(k_shape[-2])
        dim_qk = int(q_shape[-1])
        dim_v = int(v_shape[-1])
        batch_heads = _numel_shape(q_shape[:-2])
        self._add(
            "transformer.self_attention.qk",
            "self_attn_qk",
            batch_heads * length_q * length_k * dim_qk,
        )
        self._add(
            "transformer.self_attention.av",
            "self_attn_av",
            batch_heads * length_q * length_k * dim_v,
        )

    def _count_swiftvr_dispatch(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        # WanAttnProcessor passes [B, L, H, D] tensors to diffusers'
        # dispatch_attention_fn for conditional cross-attention.
        q_shape = tuple(query.shape)
        k_shape = tuple(key.shape)
        v_shape = tuple(value.shape)
        if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
            raise ValueError(
                "SwiftVR dispatch attention expects [B,L,H,D]: "
                f"q={q_shape}, k={k_shape}, v={v_shape}"
            )
        batch = int(q_shape[0])
        length_q = int(q_shape[1])
        heads = int(q_shape[2])
        length_k = int(k_shape[1])
        dim_qk = int(q_shape[3])
        dim_v = int(v_shape[3])
        self._add(
            "transformer.cross_attention.qk",
            "cross_attn_qk",
            batch * heads * length_q * length_k * dim_qk,
        )
        self._add(
            "transformer.cross_attention.av",
            "cross_attn_av",
            batch * heads * length_q * length_k * dim_v,
        )

    def _patch_attention(self) -> None:
        if self._orig_sdpa is not None or self._orig_dispatch_attention is not None:
            raise RuntimeError("RuntimeMacCounter attention hooks are already installed")

        self._orig_sdpa = F.scaled_dot_product_attention

        def counted_sdpa(query, key, value, *args, **kwargs):
            if self.enabled and self._dispatch_depth == 0:
                try:
                    self._count_sdpa(query, key, value)
                except Exception as exc:  # counting must not break model inference
                    self.count_errors.append(
                        f"SDPA count failed: {type(exc).__name__}: {exc}"
                    )
            return self._orig_sdpa(query, key, value, *args, **kwargs)

        F.scaled_dot_product_attention = counted_sdpa

        # Conditional Wan cross-attention does not use SwiftVR's _dense_attn;
        # WanAttnProcessor calls the module-global dispatch_attention_fn imported
        # from diffusers. Patch that exact global so the mathematical attention
        # MACs are counted regardless of the backend selected internally. The
        # dispatch-depth guard prevents double counting if diffusers eventually
        # implements that dispatch by calling F.scaled_dot_product_attention.
        try:
            import swiftvr.models.transformer as transformer_ops

            original_dispatch = transformer_ops.dispatch_attention_fn
            self._swiftvr_transformer_module = transformer_ops
            self._orig_dispatch_attention = original_dispatch

            def counted_dispatch(query, key, value, *args, **kwargs):
                if self.enabled:
                    try:
                        self._count_swiftvr_dispatch(query, key, value)
                    except Exception as exc:
                        self.count_errors.append(
                            f"dispatch attention count failed: {type(exc).__name__}: {exc}"
                        )
                self._dispatch_depth += 1
                try:
                    return original_dispatch(query, key, value, *args, **kwargs)
                finally:
                    self._dispatch_depth -= 1

            transformer_ops.dispatch_attention_fn = counted_dispatch
        except Exception as exc:
            # Prompt-free models do not need dispatch attention, so keep this as
            # an auditable diagnostic instead of making the generic counter fail.
            self.count_errors.append(
                f"SwiftVR dispatch patch unavailable: {type(exc).__name__}: {exc}"
            )

    def _unpatch_attention(self) -> None:
        if self._orig_sdpa is not None:
            F.scaled_dot_product_attention = self._orig_sdpa
            self._orig_sdpa = None
        if (
            self._swiftvr_transformer_module is not None
            and self._orig_dispatch_attention is not None
        ):
            self._swiftvr_transformer_module.dispatch_attention_fn = (
                self._orig_dispatch_attention
            )
        self._swiftvr_transformer_module = None
        self._orig_dispatch_attention = None
        self._dispatch_depth = 0

    @contextlib.contextmanager
    def count(self, *, reset: bool = False):
        if self.enabled:
            raise RuntimeError("Nested RuntimeMacCounter.count() contexts are unsupported")
        if reset:
            self.reset()
        self.enabled = True
        self._patch_attention()
        try:
            yield self
        finally:
            self.enabled = False
            self._unpatch_attention()

    def total_macs(self) -> int:
        return int(sum(self.macs_by_type.values()))

    def summary(self, emitted_frames: Optional[int] = None) -> Dict:
        total = self.total_macs()
        by_root = defaultdict(int)
        for name, macs in self.macs_by_name.items():
            by_root[name.split(".", 1)[0]] += int(macs)

        result: Dict = {
            "total_macs": total,
            "total_gmacs": total / 1e9,
            "by_type_gmacs": {
                key: value / 1e9 for key, value in sorted(self.macs_by_type.items())
            },
            "by_root_gmacs": {
                key: value / 1e9 for key, value in sorted(by_root.items())
            },
            "calls_by_type": dict(sorted(self.calls_by_type.items())),
            "count_errors": list(self.count_errors),
            "mac_convention": (
                "Linear/Conv/attention matmul MACs; bias, elementwise, norm, activation, "
                "RoPE, indexing, interpolation and I/O excluded"
            ),
        }
        if emitted_frames is not None and emitted_frames > 0:
            frames = int(emitted_frames)
            result["emitted_frames"] = frames
            result["gmacs_per_output_frame"] = total / frames / 1e9
            result["gflops_per_output_frame_if_1mac_2flops"] = (
                2.0 * total / frames / 1e9
            )
            result["by_root_gmacs_per_output_frame"] = {
                key: value / frames / 1e9 for key, value in sorted(by_root.items())
            }
        return result

    def print_summary(self, emitted_frames: Optional[int] = None, topk: int = 30) -> None:
        summary = self.summary(emitted_frames=emitted_frames)
        print("\n================ MACs Summary ================")
        print(f"Total MACs : {summary['total_macs']:,}")
        print(f"Total GMACs: {summary['total_gmacs']:.3f}")
        if emitted_frames is not None and emitted_frames > 0:
            print(f"Output frames: {emitted_frames}")
            print(f"GMACs / output frame: {summary['gmacs_per_output_frame']:.3f}")
            print(
                "GFLOPs / output frame, if 1 MAC = 2 FLOPs: "
                f"{summary['gflops_per_output_frame_if_1mac_2flops']:.3f}"
            )
        print("\nBy root:")
        for key, value in summary["by_root_gmacs"].items():
            print(f"  {key:16s}: {value:.3f} GMACs")
        print("\nBy operator type:")
        for key, value in summary["by_type_gmacs"].items():
            print(f"  {key:16s}: {value:.3f} GMACs")
        print(f"\nTop {topk} modules:")
        for name, macs in sorted(
            self.macs_by_name.items(), key=lambda item: item[1], reverse=True
        )[:topk]:
            print(f"  {macs / 1e9:10.3f} GMACs  {name}")
        if summary["count_errors"]:
            print("\nCounting diagnostics:")
            for error in summary["count_errors"]:
                print(f"  WARNING: {error}")
        print("==============================================\n")

    def save_json(self, path: Union[str, Path], emitted_frames: Optional[int] = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = self.summary(emitted_frames=emitted_frames)
        data["by_module_gmacs"] = {
            key: value / 1e9 for key, value in sorted(self.macs_by_name.items())
        }
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved MAC report to: {output}")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._seen_modules.clear()
        self.enabled = False
        self._unpatch_attention()
