"""CPU integration tests for the time-folded prompt-free pipeline."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from swiftvr.pipeline_prompt_free_no_time import (
    SwiftVRPromptFreeNoTimePipeline,
    _RunnerCompatiblePromptFreeNoTimeDiT,
)
from swiftvr.streaming.dit_prompt_free_no_time import (
    StreamingDiTPromptFreeNoTime,
)


class _DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


class PromptFreeNoTimePipelineTest(unittest.TestCase):
    def test_constructor_has_no_prompt_and_uses_no_time_stream(self):
        reae = _DummyModule()
        transformer = _DummyModule()
        pipe = SwiftVRPromptFreeNoTimePipeline(reae, transformer)

        self.assertIsNone(pipe.prompt_emb)
        self.assertIs(pipe.reae, reae)
        self.assertIs(pipe.transformer, transformer)
        self.assertIsInstance(pipe.dit_stream, StreamingDiTPromptFreeNoTime)

    @patch.object(SwiftVRPromptFreeNoTimePipeline, "to", autospec=True)
    @patch(
        "swiftvr.pipeline_prompt_free_no_time."
        "WanTransformer3DModelPromptFreeNoTime"
    )
    @patch("swiftvr.pipeline_prompt_free_no_time.ReAE")
    def test_loader_uses_no_time_model_and_requested_dtype(
        self,
        reae_cls,
        transformer_cls,
        pipe_to,
    ):
        reae_cls.return_value = _DummyModule()
        transformer_cls.from_pretrained.return_value = _DummyModule()
        pipe_to.return_value = None

        checkpoint = Path("/tmp/prompt_free_no_time")
        pipe = SwiftVRPromptFreeNoTimePipeline.from_pretrained(
            checkpoint,
            device="cuda",
            dtype="float16",
        )

        reae_cls.assert_called_once_with(str(checkpoint / "reae.safetensors"))
        transformer_cls.from_pretrained.assert_called_once_with(
            str(checkpoint),
            subfolder="transformer",
            torch_dtype=torch.float16,
        )
        pipe_to.assert_called_once_with(pipe, "cuda", dtype="float16")

    def test_runner_adapter_discards_legacy_prompt_argument(self):
        stream = _RunnerCompatiblePromptFreeNoTimeDiT(MagicMock())
        latent = torch.randn(1, 4, 2, 4, 4)
        expected = torch.randn_like(latent)

        with patch.object(
            StreamingDiTPromptFreeNoTime,
            "denoise",
            return_value=expected,
        ) as base_denoise:
            actual = stream.denoise(latent, prompt_emb=torch.randn(1))

        base_denoise.assert_called_once_with(latent)
        self.assertIs(actual, expected)


if __name__ == "__main__":
    unittest.main()
