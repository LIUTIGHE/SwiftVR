from __future__ import annotations

import tempfile
import unittest

import torch

from swiftvr.models.tiny_conditional_decoder import TinyConditionalDecoder
from swiftvr.models.tiny_decoder_sparsity import (
    CompactMemBlock,
    StructuredSparseMemBlock,
    convert_dense_decoder_to_sparse,
    materialize_sparse_decoder,
    stage_internal_widths,
    structured_sparsity_penalty,
)
from swiftvr.streaming.tiny_decoder import StreamingTinyConditionalDecoder


def make_dense() -> TinyConditionalDecoder:
    return TinyConditionalDecoder(
        latent_channels=6,
        condition_channels=4,
        channels=(16, 16, 8, 8),
        blocks_per_stage=(1, 1, 1, 1),
        temporal_factor=4,
        spatial_factor=16,
        patch_size=2,
        frames_to_trim=3,
    )


class TinyDecoderSparsityTests(unittest.TestCase):
    def test_dense_to_sparse_is_exact_at_unit_gates(self):
        torch.manual_seed(11)
        dense = make_dense().eval()
        sparse = convert_dense_decoder_to_sparse(dense).eval()
        self.assertEqual(sparse.block_mode, "sparse")
        self.assertTrue(any(isinstance(m, StructuredSparseMemBlock) for m in sparse.modules()))
        z = torch.randn(1, 2, 6, 2, 2)
        cond = torch.randn(1, 5, 3, 32, 32)
        with torch.no_grad():
            expected = dense(z, cond, output_frames=5)
            actual = sparse(z, cond, output_frames=5)
        self.assertTrue(torch.equal(expected, actual))

    def test_gate_penalty_backpropagates(self):
        torch.manual_seed(12)
        sparse = convert_dense_decoder_to_sparse(make_dense())
        for parameter in sparse.parameters():
            parameter.requires_grad_(False)
        gates = []
        for module in sparse.modules():
            if isinstance(module, StructuredSparseMemBlock):
                module.channel_gate.requires_grad_(True)
                gates.append(module.channel_gate)
        penalty = structured_sparsity_penalty(sparse)
        penalty.backward()
        self.assertTrue(gates)
        for gate in gates:
            self.assertIsNotNone(gate.grad)
            self.assertTrue(torch.isfinite(gate.grad).all())

    def test_stage_internal_widths_are_hardware_aligned(self):
        self.assertEqual(
            stage_internal_widths((192, 128, 64, 32), keep_ratio=0.75, multiple=8),
            (144, 96, 48, 24),
        )
        self.assertEqual(
            stage_internal_widths((192, 128, 64, 32), keep_ratio=0.40, multiple=8),
            (80, 48, 24, 16),
        )

    def test_materialization_reduces_parameters_and_preserves_shape(self):
        torch.manual_seed(13)
        sparse = convert_dense_decoder_to_sparse(make_dense())
        # Make channel ranking deterministic and non-uniform.
        for module in sparse.modules():
            if isinstance(module, StructuredSparseMemBlock):
                with torch.no_grad():
                    module.channel_gate.copy_(
                        torch.linspace(0.1, 1.0, module.channel_gate.numel())
                    )
        compact, manifest = materialize_sparse_decoder(
            sparse, keep_ratio=0.5, multiple=4
        )
        self.assertEqual(compact.block_mode, "compact")
        self.assertEqual(len(manifest), sum(compact.blocks_per_stage))
        self.assertTrue(any(isinstance(m, CompactMemBlock) for m in compact.modules()))
        sparse_params = sum(p.numel() for p in sparse.parameters())
        compact_params = sum(p.numel() for p in compact.parameters())
        self.assertLess(compact_params, sparse_params)
        z = torch.randn(1, 2, 6, 2, 2)
        cond = torch.randn(1, 5, 3, 32, 32)
        output = compact(z, cond, output_frames=5)
        self.assertEqual(tuple(output.shape), (1, 5, 3, 32, 32))
        output.float().mean().backward()
        grads = [p.grad for p in compact.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))
        self.assertTrue(all(g is None or torch.isfinite(g).all() for g in grads))

    def test_compact_streaming_and_checkpoint_roundtrip(self):
        torch.manual_seed(14)
        sparse = convert_dense_decoder_to_sparse(make_dense())
        compact, _ = materialize_sparse_decoder(sparse, keep_ratio=0.5, multiple=4)
        compact.eval()
        stream = StreamingTinyConditionalDecoder(compact)
        with torch.no_grad():
            first = stream.decode_chunk(
                torch.randn(1, 2, 6, 2, 2),
                torch.randn(1, 8, 3, 32, 32),
            )
            middle = stream.decode_chunk(
                torch.randn(1, 1, 6, 2, 2),
                torch.randn(1, 4, 3, 32, 32),
            )
        self.assertEqual(tuple(first.shape), (1, 5, 3, 32, 32))
        self.assertEqual(tuple(middle.shape), (1, 4, 3, 32, 32))

        z = torch.randn(1, 2, 6, 2, 2)
        cond = torch.randn(1, 5, 3, 32, 32)
        with torch.no_grad():
            reference = compact(z, cond, output_frames=5)
        with tempfile.TemporaryDirectory() as directory:
            compact.save_pretrained(directory)
            loaded = TinyConditionalDecoder.from_pretrained(directory).eval()
            self.assertEqual(loaded.block_mode, "compact")
            self.assertEqual(loaded.block_internal_channels, compact.block_internal_channels)
            with torch.no_grad():
                actual = loaded(z, cond, output_frames=5)
        self.assertTrue(torch.equal(reference, actual))


if __name__ == "__main__":
    unittest.main()
