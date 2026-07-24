"""CPU tests for the prompt-free SwiftVR pipeline.

Run from the repository root with:

    python -m unittest tests.test_pipeline_prompt_free
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from swiftvr.pipeline_prompt_free import (
    SwiftVRPromptFreePipeline,
    _RunnerCompatiblePromptFreeDiT,
)
from swiftvr.streaming.dit_prompt_free import StreamingDiTPromptFree


class _DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


class PromptFreePipelineTest(unittest.TestCase):
    def test_constructor_uses_prompt_free_stream_and_no_prompt_embedding(self):
        reae = _DummyModule()
        transformer = _DummyModule()

        pipe = SwiftVRPromptFreePipeline(reae, transformer)

        self.assertIs(pipe.reae, reae)
        self.assertIs(pipe.transformer, transformer)
        self.assertIsNone(pipe.prompt_emb)
        self.assertIsInstance(pipe.dit_stream, StreamingDiTPromptFree)

    @patch("swiftvr.pipeline_prompt_free.WanTransformer3DModelPromptFree")
    @patch("swiftvr.pipeline_prompt_free.ReAE")
    def test_from_pretrained_does_not_load_prompt_embedding(
        self,
        reae_cls,
        transformer_cls,
    ):
        reae = _DummyModule()
        transformer = _DummyModule()
        reae_cls.return_value = reae
        transformer_cls.from_pretrained.return_value = transformer

        checkpoint = Path("/tmp/prompt_free_checkpoint")
        pipe = SwiftVRPromptFreePipeline.from_pretrained(checkpoint)

        reae_cls.assert_called_once_with(
            str(checkpoint / "reae.safetensors")
        )
        transformer_cls.from_pretrained.assert_called_once_with(
            str(checkpoint),
            subfolder="transformer",
        )
        self.assertIs(pipe.reae, reae)
        self.assertIs(pipe.transformer, transformer)
        self.assertIsNone(pipe.prompt_emb)

    @patch.object(SwiftVRPromptFreePipeline, "to", autospec=True)
    @patch("swiftvr.pipeline_prompt_free.WanTransformer3DModelPromptFree")
    @patch("swiftvr.pipeline_prompt_free.ReAE")
    def test_from_pretrained_loads_transformer_directly_in_requested_dtype(
        self,
        reae_cls,
        transformer_cls,
        pipe_to,
    ):
        reae_cls.return_value = _DummyModule()
        transformer_cls.from_pretrained.return_value = _DummyModule()
        pipe_to.return_value = None

        checkpoint = Path("/tmp/prompt_free_checkpoint")
        pipe = SwiftVRPromptFreePipeline.from_pretrained(
            checkpoint,
            device="cuda",
            dtype="float16",
        )

        transformer_cls.from_pretrained.assert_called_once_with(
            str(checkpoint),
            subfolder="transformer",
            torch_dtype=torch.float16,
        )
        pipe_to.assert_called_once_with(pipe, "cuda", dtype="float16")

    def test_runner_adapter_discards_prompt_for_regular_chunk(self):
        transformer = MagicMock()
        stream = _RunnerCompatiblePromptFreeDiT(transformer)
        lq = torch.randn(1, 4, 2, 4, 4)
        expected = torch.randn_like(lq)

        with patch.object(
            StreamingDiTPromptFree,
            "denoise",
            return_value=expected,
        ) as base_denoise:
            output = stream.denoise(lq, prompt_emb=torch.randn(1))

        base_denoise.assert_called_once_with(lq)
        self.assertIs(output, expected)

    def test_runner_adapter_discards_prompt_for_last_chunk(self):
        transformer = MagicMock()
        stream = _RunnerCompatiblePromptFreeDiT(transformer)
        z_new = torch.randn(1, 2, 4, 4, 4)
        spec = MagicMock()
        prev = torch.randn(1, 4, 1, 4, 4)
        expected = torch.randn_like(z_new)
        device = torch.device("cpu")

        with patch.object(
            StreamingDiTPromptFree,
            "denoise_last_chunk",
            return_value=expected,
        ) as base_last:
            output = stream.denoise_last_chunk(
                z_new,
                spec,
                torch.randn(1),
                prev,
                6,
                device,
                torch.float32,
            )

        base_last.assert_called_once_with(
            z_new,
            spec,
            prev,
            6,
            device,
            torch.float32,
        )
        self.assertIs(output, expected)


if __name__ == "__main__":
    unittest.main()
