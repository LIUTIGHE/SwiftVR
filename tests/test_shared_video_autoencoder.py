from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from swiftvr.models.reae import ReAE
from swiftvr.models.reae_slim_decoder import SlimReAEDecoder
from swiftvr.models.shared_video_autoencoder import SharedVideoAutoencoder


class SharedVideoAutoencoderTest(unittest.TestCase):
    def _model(self) -> SharedVideoAutoencoder:
        base = ReAE(checkpoint_path=None, latent_channels=48)
        decoder = SlimReAEDecoder(
            channels=(8, 8, 4, 4),
            latent_channels=48,
            patch_size=2,
            frames_to_trim=3,
        )
        return SharedVideoAutoencoder(base.encoder, decoder)

    def test_shape_contract(self):
        model = self._model().eval()
        video = torch.rand(1, 5, 3, 16, 16)
        with torch.no_grad():
            latent = model.encode(video)
            recon = model.decode(latent, output_frames=5)
        self.assertEqual(tuple(latent.shape), (1, 2, 48, 1, 1))
        self.assertEqual(tuple(recon.shape), tuple(video.shape))
        self.assertFalse(model.config_dict["is_variational"])
        self.assertIsNone(model.config_dict["latent_scaling"])

    def test_rejects_non_4k1_clip(self):
        model = self._model()
        with self.assertRaisesRegex(ValueError, "T=4k\\+1"):
            model.encode(torch.rand(1, 4, 3, 16, 16))

    def test_frozen_decoder_keeps_input_gradient(self):
        model = self._model().freeze()
        latent = torch.randn(1, 2, 48, 1, 1, requires_grad=True)
        output = model.decode(latent, output_frames=5)
        output.mean().backward()
        self.assertIsNotNone(latent.grad)
        self.assertGreater(float(latent.grad.abs().sum().item()), 0.0)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_export_roundtrip(self):
        model = self._model().eval()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "codec"
            model.save_pretrained(root)
            loaded = SharedVideoAutoencoder.from_pretrained(root)
            self.assertEqual(loaded.decoder_channels, (8, 8, 4, 4))
            self.assertEqual(loaded.latent_channels, 48)
            self.assertEqual(loaded.frames_to_trim, 3)
            self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
