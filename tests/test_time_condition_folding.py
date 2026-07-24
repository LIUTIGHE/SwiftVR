"""CPU tests for fixed timestep folding and the no-time SwiftVR student."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from swiftvr.models.transformer_prompt_free import WanTransformer3DModelPromptFree
from swiftvr.models.transformer_prompt_free_no_time import (
    WanTransformer3DModelPromptFreeNoTime,
)
from swiftvr.streaming.dit_prompt_free_no_time import (
    StreamingDiTPromptFreeNoTime,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "fold_time_condition.py"
_SPEC = importlib.util.spec_from_file_location("fold_time_condition", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FOLDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FOLDER)


def _model_kwargs() -> dict[str, object]:
    return {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 4,
        "attention_head_dim": 8,
        "in_channels": 8,
        "out_channels": 8,
        "text_dim": 16,
        "freq_dim": 16,
        "ffn_dim": 64,
        "num_layers": 2,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "rope_max_seq_len": 64,
        "self_attn_window_hw": (4, 4),
        "adapter_dim": 8,
    }


def _build_folded_pair():
    torch.manual_seed(7)
    source = WanTransformer3DModelPromptFree(**_model_kwargs()).float().eval()

    # Exercise a non-zero adapter path rather than only the hard-removal state.
    with torch.no_grad():
        for block in source.blocks:
            torch.nn.init.normal_(block.prompt_free_adapter.up.weight, std=0.01)
            torch.nn.init.normal_(block.prompt_free_adapter.up.bias, std=0.01)

    config = dict(source.config)
    folded_state, report = _FOLDER.fold_state_dict(
        source.state_dict(),
        config,
        timestep=1000.0,
    )
    folded = WanTransformer3DModelPromptFreeNoTime(
        **_model_kwargs(),
        folded_timestep=1000.0,
        time_condition_folded=True,
    ).float().eval()
    folded.load_state_dict(folded_state, strict=True)
    return source, folded, report


class TimeConditionFoldingTest(unittest.TestCase):
    def test_folded_model_matches_prompt_free_model(self):
        source, folded, _ = _build_folded_pair()
        hidden_states = torch.randn(1, 8, 2, 8, 8)
        timestep = torch.tensor([1000.0])

        with torch.no_grad():
            expected = source(hidden_states.clone(), timestep).sample
            actual = folded(hidden_states.clone()).sample

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_folded_model_has_no_time_parameters_or_timestep_argument(self):
        _, folded, report = _build_folded_pair()
        names = tuple(name for name, _ in folded.named_parameters())

        self.assertFalse(any("condition_embedder" in name for name in names))
        self.assertFalse(any("time_embedder" in name for name in names))
        self.assertEqual(report["folded_tensor_count"], 3)
        self.assertGreater(report["dropped_tensor_count"], 0)

        hidden_states = torch.randn(1, 8, 2, 8, 8)
        with torch.no_grad():
            output = folded(hidden_states).sample
        self.assertEqual(output.shape, hidden_states.shape)

    def test_no_time_stream_matches_direct_forward_without_overlap(self):
        _, folded, _ = _build_folded_pair()
        stream = StreamingDiTPromptFreeNoTime(folded, overlap=0)
        latent = torch.randn(1, 8, 2, 8, 8)

        with torch.no_grad():
            expected = latent - folded(latent.clone()).sample
            actual = stream.denoise(latent.clone())

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(stream._g_off, latent.shape[2])

    def test_checkpoint_converter_writes_loadable_no_time_layout(self):
        source, _, _ = _build_folded_pair()
        config = dict(source.config)
        config["_class_name"] = "WanTransformer3DModelPromptFree"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            output_root = root / "output"
            transformer_dir = source_root / "transformer"
            transformer_dir.mkdir(parents=True)
            (transformer_dir / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            save_file(
                source.state_dict(),
                str(transformer_dir / "diffusion_pytorch_model.safetensors"),
            )
            (source_root / "reae.safetensors").write_bytes(b"dummy")

            report = _FOLDER.fold_checkpoint(
                source_root,
                output_root,
                timestep=1000.0,
            )

            output_config = json.loads(
                (output_root / "transformer" / "config.json").read_text(
                    encoding="utf-8"
                )
            )
            output_state = load_file(
                str(
                    output_root
                    / "transformer"
                    / "diffusion_pytorch_model.safetensors"
                )
            )

            self.assertEqual(
                output_config["_class_name"],
                "WanTransformer3DModelPromptFreeNoTime",
            )
            self.assertTrue(output_config["time_condition_folded"])
            self.assertEqual(output_config["folded_timestep"], 1000.0)
            self.assertFalse(
                any(key.startswith("condition_embedder.") for key in output_state)
            )
            self.assertTrue((output_root / "reae.safetensors").is_file())
            self.assertEqual(report["fixed_timestep"], 1000.0)


if __name__ == "__main__":
    unittest.main()
