"""CPU tests for tools/convert_prompt_free_checkpoint.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "convert_prompt_free_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("convert_prompt_free_checkpoint", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CONVERTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONVERTER)


def _tiny_config() -> dict[str, object]:
    return {
        "_class_name": "WanTransformer3DModel",
        "patch_size": [1, 2, 2],
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 16,
        "freq_dim": 8,
        "ffn_dim": 16,
        "num_layers": 2,
        "cross_attn_norm": True,
        "eps": 1e-6,
    }


def _tiny_teacher_state() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "patch_embedding.weight": torch.randn(8, 4, 1, 2, 2),
        "patch_embedding.bias": torch.randn(8),
        "condition_embedder.time_embedder.linear_1.weight": torch.randn(8, 8),
        "condition_embedder.time_embedder.linear_1.bias": torch.randn(8),
        "condition_embedder.time_proj.weight": torch.randn(48, 8),
        "condition_embedder.time_proj.bias": torch.randn(48),
        "condition_embedder.text_embedder.linear_1.weight": torch.randn(8, 16),
        "condition_embedder.text_embedder.linear_1.bias": torch.randn(8),
        "proj_out.weight": torch.randn(16, 8),
        "proj_out.bias": torch.randn(16),
        "scale_shift_table": torch.randn(1, 2, 8),
    }
    for layer_index in range(2):
        prefix = f"blocks.{layer_index}"
        state.update(
            {
                f"{prefix}.attn1.to_q.weight": torch.randn(8, 8),
                f"{prefix}.attn1.to_q.bias": torch.randn(8),
                f"{prefix}.attn1.to_qkv.weight": torch.randn(24, 8),
                f"{prefix}.attn2.to_q.weight": torch.randn(8, 8),
                f"{prefix}.attn2.to_k.weight": torch.randn(8, 8),
                f"{prefix}.norm2.weight": torch.randn(8),
                f"{prefix}.norm2.bias": torch.randn(8),
                f"{prefix}.ffn.net.0.proj.weight": torch.randn(16, 8),
                f"{prefix}.ffn.net.0.proj.bias": torch.randn(16),
                f"{prefix}.scale_shift_table": torch.randn(1, 6, 8),
            }
        )
    return state


class PromptFreeCheckpointConversionTest(unittest.TestCase):
    def test_drop_policy_is_narrow(self):
        self.assertTrue(
            _CONVERTER.should_drop_source_key(
                "condition_embedder.text_embedder.linear_1.weight"
            )
        )
        self.assertTrue(_CONVERTER.should_drop_source_key("blocks.0.attn2.to_q.weight"))
        self.assertTrue(_CONVERTER.should_drop_source_key("blocks.0.norm2.weight"))
        self.assertTrue(_CONVERTER.should_drop_source_key("blocks.0.attn1.to_qkv.weight"))
        self.assertFalse(_CONVERTER.should_drop_source_key("blocks.0.attn1.to_q.weight"))
        self.assertFalse(_CONVERTER.should_drop_source_key("blocks.0.ffn.net.0.proj.weight"))

    def test_convert_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            transformer_dir = source / "transformer"
            transformer_dir.mkdir(parents=True)

            config = _tiny_config()
            (transformer_dir / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            source_state = _tiny_teacher_state()
            save_file(
                source_state,
                str(transformer_dir / "diffusion_pytorch_model.safetensors"),
                metadata={"source": "unit-test"},
            )
            save_file({"dummy": torch.ones(1)}, str(source / "reae.safetensors"))
            save_file(
                {"prompt_emb": torch.ones(1, 2, 3)},
                str(source / "prompt_embedding.safetensors"),
            )

            report = _CONVERTER.convert_checkpoint(
                source,
                output,
                adapter_dim=3,
                seed=17,
            )

            converted = load_file(
                str(output / "transformer" / "diffusion_pytorch_model.safetensors")
            )
            output_config = json.loads(
                (output / "transformer" / "config.json").read_text(encoding="utf-8")
            )
            saved_report = json.loads(
                (output / "conversion_report.json").read_text(encoding="utf-8")
            )

            self.assertIn("blocks.0.attn1.to_q.weight", converted)
            self.assertIn("blocks.1.ffn.net.0.proj.weight", converted)
            self.assertNotIn("blocks.0.attn2.to_q.weight", converted)
            self.assertNotIn("blocks.0.norm2.weight", converted)
            self.assertNotIn("blocks.0.attn1.to_qkv.weight", converted)
            self.assertNotIn(
                "condition_embedder.text_embedder.linear_1.weight",
                converted,
            )

            for layer_index in range(2):
                prefix = f"blocks.{layer_index}.prompt_free_adapter"
                self.assertEqual(tuple(converted[f"{prefix}.down.weight"].shape), (3, 8))
                self.assertEqual(tuple(converted[f"{prefix}.up.weight"].shape), (8, 3))
                self.assertTrue(torch.equal(converted[f"{prefix}.norm.weight"], torch.ones(8)))
                self.assertTrue(torch.equal(converted[f"{prefix}.norm.bias"], torch.zeros(8)))
                self.assertTrue(torch.equal(converted[f"{prefix}.up.weight"], torch.zeros(8, 3)))
                self.assertTrue(torch.equal(converted[f"{prefix}.up.bias"], torch.zeros(8)))

            self.assertEqual(output_config["_class_name"], "WanTransformer3DModelPromptFree")
            self.assertEqual(output_config["adapter_dim"], 3)
            self.assertTrue((output / "reae.safetensors").is_file())
            self.assertFalse((output / "prompt_embedding.safetensors").exists())
            self.assertEqual(report["added_tensor_count"], 12)
            self.assertEqual(saved_report["omitted_files"], ["prompt_embedding.safetensors"])

    def test_conversion_is_deterministic_for_same_seed(self):
        config = _tiny_config()
        source_state = _tiny_teacher_state()
        first, _ = _CONVERTER.convert_state_dict(
            source_state,
            config,
            adapter_dim=3,
            seed=5,
        )
        second, _ = _CONVERTER.convert_state_dict(
            source_state,
            config,
            adapter_dim=3,
            seed=5,
        )
        self.assertTrue(
            torch.equal(
                first["blocks.1.prompt_free_adapter.down.weight"],
                second["blocks.1.prompt_free_adapter.down.weight"],
            )
        )


if __name__ == "__main__":
    unittest.main()
