"""R2 CPU contracts. Full pretrained-LPIPS check runs only with local weights."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

r2 = load("_m10_r2_perceptual_test", ROOT / "swiftvr/training/m10_perceptual.py")
m10 = load("_m10_r2_recovery_test", ROOT / "swiftvr/training/m10_recovery.py")
entry = load("_m10_r2_entry_test", ROOT / "tools/train_m10_backbone_recovery_ddp.py")


class ToyPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, 3, padding=1)
        self.requires_grad_(False).eval()

    def forward(self, x, y):
        return (self.conv(x) - self.conv(y)).square().mean((1, 2, 3), keepdim=True)


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(4, 3, 1)
        self.requires_grad_(False).eval()

    def forward(self, z, *, output_frames, clamp=False):
        x = self.conv(z.permute(0, 2, 1, 3, 4)).sigmoid().permute(0, 2, 1, 3, 4)
        return x[:, :output_frames]


class R2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.metric = ToyPerceptual()

    def test_lpips_microbatch_value_and_gradient_agree(self):
        gt = torch.rand(2, 5, 3, 8, 8, requires_grad=True)
        value, grad = [], []
        source = torch.rand_like(gt)
        for size in (1, 4, 16):
            pred = source.detach().clone().requires_grad_()
            y = r2.lpips_frame_values(self.metric, pred, gt, microbatch_frames=size)
            self.assertEqual(y.shape, (2, 5))
            y.mean().backward()
            value.append(y.detach()); grad.append(pred.grad)
        for i in (1, 2):
            torch.testing.assert_close(value[0], value[i])
            torch.testing.assert_close(grad[0], grad[i])
        self.assertIsNone(gt.grad)
        self.assertTrue(all(p.grad is None for p in self.metric.parameters()))

    def test_identical_dynamic_clip_has_zero_perceptual_loss(self):
        clip = torch.rand(1, 13, 3, 8, 8)
        y = r2.lpips_frame_values(self.metric, clip, clip)
        self.assertEqual(float(y.abs().sum()), 0)

    def test_input_normalization_matches_a1_semantics(self):
        pred = torch.linspace(-0.1, 1.1, 192).reshape(1, 1, 3, 8, 8)
        target = torch.rand_like(pred)
        actual = r2.lpips_frame_values(self.metric, pred, target)
        expected = self.metric(pred[0].clamp(0, 1) * 2 - 1, target[0].clamp(0, 1) * 2 - 1).mean()
        torch.testing.assert_close(actual.mean(), expected)

    def test_r2_frozen_decoder_input_gradients_and_probe(self):
        backbone, decoder = nn.Conv3d(4, 4, 1), ToyDecoder()
        z = torch.randn(1, 4, 4, 8, 8)
        velocity = backbone(z)
        rgb = m10.decode_student(decoder, z, velocity, output_frames=4)
        target = torch.rand_like(rgb, requires_grad=True)
        lpips = r2.lpips_frame_values(self.metric, rgb, target).mean()
        out = {"prediction": rgb, "teacher_prediction": target, "velocity": velocity,
               "router_balance_loss": velocity.sum() * 0}
        w = m10.RecoveryWeights(velocity_nmse=.01, velocity_cosine_loss=0, hf_l1=0,
                                hf_temporal_l1=0, lpips=.2)
        terms = m10.recovery_objective(out, torch.randn_like(velocity), weights=w,
                                      high_pass=m10.GaussianHighPass(), perceptual_loss=lpips)
        probe = r2.loss_gradient_probe(terms, velocity)
        self.assertGreater(probe["appearance_gradient_norm"], 0)
        self.assertFalse(probe["weights_automatically_changed"])
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        terms["loss"].backward()
        self.assertGreater(float(backbone.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(backbone.weight.grad).all())
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))

    def test_legacy_objective_unchanged_and_r2_selection(self):
        sv, tv = torch.randn(1, 4, 4, 2, 2), torch.randn(1, 4, 4, 2, 2)
        out = {"velocity": sv, "prediction": torch.rand(1, 4, 3, 8, 8),
               "teacher_prediction": torch.rand(1, 4, 3, 8, 8), "router_balance_loss": torch.tensor(1.2)}
        terms = m10.recovery_objective(out, tv, weights=m10.RecoveryWeights(), high_pass=m10.GaussianHighPass())
        self.assertNotIn("lpips", terms)
        expected = .25 * terms["velocity_nmse"] + .25 * terms["velocity_cosine_loss"] + terms["rgb_l1"] + terms["hf_l1"] + .01 * terms["router_balance"]
        torch.testing.assert_close(terms["loss"], expected)
        torch.testing.assert_close(terms["teacher_selection_score"], terms["rgb_l1"] + terms["hf_l1"] + terms["hf_temporal_l1"])
        w = m10.RecoveryWeights(velocity_nmse=.01, velocity_cosine_loss=0, hf_l1=0, lpips=.2)
        result = m10.recovery_objective(out, tv, weights=w, high_pass=m10.GaussianHighPass(), perceptual_loss=torch.tensor(.12))
        torch.testing.assert_close(result["teacher_selection_score"], result["rgb_l1"] + .2 * result["lpips"])
        torch.testing.assert_close(result["loss"], .01 * result["velocity_nmse"] + result["rgb_l1"] + .2 * result["lpips"] + .01 * result["router_balance"])
        with self.assertRaises(ValueError):
            m10.recovery_objective(out, tv, weights=w, high_pass=m10.GaussianHighPass())

    def test_phase_counts_balanced_and_gt_not_an_input(self):
        pred, target = torch.ones(2, 13, 3, 8, 8), torch.zeros(2, 13, 3, 8, 8)
        lpips = torch.arange(13, dtype=torch.float32).repeat(2, 1)
        sums = r2.phase_totals(pred, target, lpips)
        for phase in range(4):
            self.assertEqual(sums[f"phase{phase}_frames"], 6)
            self.assertEqual(sums[f"phase{phase}_rgb_l1"], 6)
        self.assertEqual(sums["phase0_lpips"], 2 * (1 + 5 + 9))
        self.assertEqual(sums["phase3_lpips"], 2 * (4 + 8 + 12))

    def test_probe_detects_opposed_gradients_without_modifying_grad(self):
        v = torch.ones(1, 3, requires_grad=True)
        terms = {"weighted_velocity_nmse": v.sum(), "weighted_velocity_cosine_loss": v.sum() * 0,
                 "weighted_rgb_l1": -2 * v.sum(), "weighted_lpips": -v.sum()}
        p = r2.loss_gradient_probe(terms, v)
        self.assertAlmostEqual(p["anchor_appearance_cosine"], -1, places=6)
        self.assertAlmostEqual(p["anchor_to_appearance_norm_ratio"], 1 / 3, places=6)
        self.assertIsNone(v.grad)

    def test_missing_local_weights_fail_without_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.hub, "get_dir", return_value=tmp):
            with self.assertRaises(FileNotFoundError):
                r2.resolve_alexnet_weights()
            path = Path(tmp) / "custom.pth"
            path.write_text("dummy")
            self.assertEqual(r2.resolve_alexnet_weights(path), path.resolve())

    def test_trunk_maps_every_pretrained_parameter(self):
        trunk = nn.Module()
        trunk.slice1 = nn.Sequential(nn.Conv2d(3, 4, 1))
        state = {"features.0.weight": torch.ones(4, 3, 1, 1), "features.0.bias": torch.ones(4)}
        r2._load_alexnet_trunk(trunk, state)
        self.assertTrue(all(bool((p == 1).all()) for p in trunk.parameters()))
        with self.assertRaises(ValueError):
            r2._load_alexnet_trunk(trunk, {})

    def test_offline_constructor_loads_heads_and_trunk(self):
        class FakeLPIPS(nn.Module):
            def __init__(self, **kw):
                super().__init__()
                self.options = kw
                self.net = nn.Module()
                self.net.slice1 = nn.Sequential(nn.Conv2d(3, 4, 1))
                for i in range(5):
                    head = nn.Module()
                    head.model = nn.Sequential(nn.Dropout(), nn.Conv2d(1, 1, 1, bias=False))
                    setattr(self, f"lin{i}", head)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / "weights/v0.1/alex.pth"
            calibration.parent.mkdir(parents=True)
            torch.save({f"lin{i}.model.1.weight": torch.ones(1, 1, 1, 1) * .25 for i in range(5)}, calibration)
            trunk = root / "alexnet.pth"
            torch.save({"features.0.weight": torch.ones(4, 3, 1, 1), "features.0.bias": torch.ones(4)}, trunk)
            with patch.dict(sys.modules, {"lpips": types.SimpleNamespace(LPIPS=FakeLPIPS)}), \
                    patch.object(r2.inspect, "getfile", return_value=str(root / "lpips.py")):
                metric, info = r2.load_local_lpips(trunk)
            self.assertTrue(metric.options["pnet_rand"])
            self.assertFalse(metric.options["pretrained"])
            self.assertFalse(metric.pnet_rand)
            self.assertTrue(all(not p.requires_grad for p in metric.parameters()))
            self.assertFalse(metric.training)
            self.assertEqual(float(metric.lin0.model[1].weight.item()), .25)
            self.assertEqual(info["alexnet_weights"], str(trunk))

    def test_legacy_cli_defaults_preserved(self):
        args = entry.build_parser().parse_args([
            "--base-checkpoint", "base", "--student-init", "s", "--decoder-checkpoint", "d",
            "--teacher-cache", "tr", "--val-teacher-cache", "va", "--output-dir", "out"])
        self.assertEqual(args.lpips_weight, 0)
        self.assertEqual(args.velocity_nmse_weight, .25)
        self.assertEqual(args.hf_weight, 1)
        self.assertEqual(args.visual_frame_indices, "0,6,12")

    def test_real_local_lpips_matches_reference(self):
        try:
            path = r2.resolve_alexnet_weights()
            import lpips
        except (ImportError, FileNotFoundError) as exc:
            self.skipTest(f"Pretrained local LPIPS not available: {exc}")
        # No download is allowed even in the reference construction.
        with patch.object(torch.hub, "download_url_to_file", side_effect=AssertionError("Network download forbidden")):
            local, _ = r2.load_local_lpips(path)
            reference = lpips.LPIPS(net="alex", verbose=False).eval().requires_grad_(False)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(local.state_dict()[key], value, rtol=0, atol=0)
        x, y = torch.rand(1, 2, 3, 64, 64), torch.rand(1, 2, 3, 64, 64)
        torch.testing.assert_close(r2.lpips_frame_values(local, x, y), r2.lpips_frame_values(reference, x, y))


if __name__ == "__main__":
    unittest.main()
