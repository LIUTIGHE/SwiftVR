"""CPU contracts for P2 video GAN; real checkpoint/DDP still needs server smoke."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GAN = load("m8_p2_gan_unit", "swiftvr/training/m8_p2_gan.py")
TRAIN = load("m8_p2_train_unit", "tools/train_m8_p2_video_gan.py")


class VideoGANContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_output_retains_full_temporal_resolution(self):
        d = GAN.TemporalPatchDiscriminator(width=8)
        frames = torch.rand(2, 13, 3, 32, 32)
        scores = d(frames)
        self.assertEqual(scores.shape[:3], (2, 1, 13))

    def test_rgb_and_temporal_differences_enter_discriminator(self):
        class Capture(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.x = None
            def forward(self, x):
                self.x = x; return x
        d = GAN.TemporalPatchDiscriminator(width=8)
        capture = Capture()
        d.net = capture
        frames = torch.zeros(1, 5, 3, 32, 32)
        frames[:, 2] = 0.4
        d(frames)
        features = capture.x
        self.assertEqual(features.shape, (1, 6, 5, 32, 32))
        self.assertGreater(float(features[:, 3:, 2].abs().mean()), 0)
        self.assertGreater(float(features[:, 3:, 3].abs().mean()), 0)
        self.assertEqual(float(features[:, 3:, 0].abs().max()), 0)

    def test_paired_crops_preserve_time_and_pixel_alignment(self):
        x = torch.arange(2*13*3*39*40, dtype=torch.float32).reshape(2,13,3,39,40) / 1e6
        a, b = GAN.paired_spatial_crop(x, x + 3, 32)
        self.assertEqual(tuple(a.shape), (2, 13, 3, 32, 32))
        torch.testing.assert_close(b - a, torch.full_like(a, 3))
        with self.assertRaises(ValueError):
            GAN.paired_spatial_crop(x, x[:1], 32)

    def test_hinge_loss_reacts_to_logits(self):
        yes = GAN.hinge_discriminator_loss(torch.tensor([2.]), torch.tensor([-2.]))
        no = GAN.hinge_discriminator_loss(torch.tensor([-2.]), torch.tensor([2.]))
        self.assertLess(float(yes), float(no))
        self.assertAlmostEqual(float(GAN.generator_adversarial_loss(torch.tensor([1.]))), -1)

    def test_critic_step_does_not_backprop_into_generator(self):
        gen = torch.nn.Conv3d(3, 3, 1)
        d = GAN.TemporalPatchDiscriminator(width=8)
        raw = torch.rand(1, 3, 5, 32, 32)
        fake = gen(raw).permute(0,2,1,3,4).contiguous()
        real = torch.rand_like(fake)
        logits = d(torch.cat([real, fake.detach()], dim=0))
        real_logit, fake_logit = logits.chunk(2, dim=0)
        GAN.hinge_discriminator_loss(real_logit, fake_logit).backward()
        self.assertTrue(all(p.grad is None for p in gen.parameters()))
        self.assertTrue(any(p.grad is not None for p in d.parameters()))

    def test_gan_step_reaches_generator_with_critic_frozen(self):
        gen = torch.nn.Conv3d(3,3,1)
        d = GAN.TemporalPatchDiscriminator(width=8)
        d.eval().requires_grad_(False)
        fake = gen(torch.rand(1,3,5,32,32)).permute(0,2,1,3,4).contiguous()
        GAN.generator_adversarial_loss(d(fake)).backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in gen.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in d.parameters()))

    def test_critic_safetensors_round_trip(self):
        d = GAN.TemporalPatchDiscriminator(width=8).eval()
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/"discriminator.safetensors"
            save_file({k:v.detach().contiguous() for k,v in d.state_dict().items()}, str(p))
            restored = GAN.TemporalPatchDiscriminator(width=8).eval()
            restored.load_state_dict(load_file(str(p)), strict=True)
            x = torch.rand(1,5,3,32,32)
            torch.testing.assert_close(d(x), restored(x))

    def test_warmup_ramp(self):
        self.assertAlmostEqual(GAN.adversarial_weight(25, weight=0.01, ramp_steps=100), 0.0025)
        self.assertAlmostEqual(GAN.adversarial_weight(250, weight=0.01, ramp_steps=100), 0.01)
        with self.assertRaises(ValueError):
            GAN.adversarial_weight(0, weight=0.01, ramp_steps=100)

    def test_learning_rate_and_defaults(self):
        args = TRAIN.build_parser().parse_args([
            "--base-checkpoint","E","--student-init","P1","--decoder-checkpoint","A1",
            "--legacy-view-cache","legacy","--val-view-cache","val",
            "--ultra-pairs","paired","--output-dir","P2"])
        self.assertEqual(args.max_steps,1000)
        self.assertEqual(args.gan_weight,0.01)
        self.assertEqual(args.validate_every,250)
        self.assertEqual(args.lpips_weight,0.1)
        self.assertEqual(args.disc_crop,128)

    def test_shell_and_cli(self):
        launcher=ROOT / "tools/run_m8_p2_gan.sh"
        subprocess.run(["bash","-n",str(launcher)],check=True)
        result=subprocess.run([sys.executable,str(ROOT/"tools/train_m8_p2_video_gan.py"),"--help"],
                              check=False,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == "__main__":
    unittest.main()
