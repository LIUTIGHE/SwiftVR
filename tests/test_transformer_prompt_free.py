"""Minimal CPU tests for the prompt-free SwiftVR transformer.

Run from the repository root with:

    python -m unittest tests.test_transformer_prompt_free
"""

import unittest

import torch

from swiftvr.models.transformer_prompt_free import (
    PromptFreeResidualAdapter,
    WanTransformer3DModelPromptFree,
)


class PromptFreeTransformerTest(unittest.TestCase):
    def test_adapter_is_exact_zero_at_initialization(self):
        torch.manual_seed(0)
        adapter = PromptFreeResidualAdapter(dim=32, bottleneck_dim=8)
        hidden_states = torch.randn(2, 11, 32)

        with torch.no_grad():
            residual = adapter(hidden_states)

        self.assertEqual(residual.shape, hidden_states.shape)
        self.assertEqual(torch.count_nonzero(residual).item(), 0)

    def test_model_has_no_text_or_cross_attention_parameters(self):
        model = self._build_tiny_model()
        parameter_names = tuple(name for name, _ in model.named_parameters())
        module_names = tuple(name for name, _ in model.named_modules())

        self.assertFalse(any("text_embedder" in name for name in parameter_names))
        self.assertFalse(any("attn2" in name for name in module_names))
        self.assertTrue(
            any("prompt_free_adapter" in name for name in parameter_names)
        )

    def test_model_preserves_latent_shape(self):
        torch.manual_seed(0)
        model = self._build_tiny_model().eval()
        hidden_states = torch.randn(1, 8, 2, 8, 8)
        timestep = torch.tensor([1000.0])

        with torch.no_grad():
            output = model(hidden_states, timestep).sample

        self.assertEqual(output.shape, hidden_states.shape)

    @staticmethod
    def _build_tiny_model():
        return WanTransformer3DModelPromptFree(
            patch_size=(1, 2, 2),
            num_attention_heads=4,
            attention_head_dim=8,
            in_channels=8,
            out_channels=8,
            freq_dim=16,
            ffn_dim=64,
            num_layers=2,
            rope_max_seq_len=64,
            self_attn_window_hw=(4, 4),
            adapter_dim=8,
        )


if __name__ == "__main__":
    unittest.main()
