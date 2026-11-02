"""CPU contracts for M10; numerical losses do not require Diffusers/Hub access."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m10 = _load("_m10_loss_contract", ROOT / "swiftvr/training/m10_recovery.py")
entry = _load("_m10_entry_contract", ROOT / "tools/train_m10_backbone_recovery_ddp.py")


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(4, 3, 1)

    def forward(self, z, *, output_frames, clamp=False):
        x = self.proj(z.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        x = x[:, :output_frames]
        return x.clamp(0, 1) if clamp else x


class M10LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.hp = m10.GaussianHighPass()

    def test_constant_has_no_high_frequency(self):
        y = self.hp(torch.ones(2, 4, 3, 9, 9) * 0.4)
        torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-6, rtol=0)

    def test_filter_never_mixes_batch_time_or_channel(self):
        x = torch.zeros(2, 4, 3, 9, 9)
        x[1, 2, 0, 4, 4] = 1
        y = self.hp(x)
        self.assertGreater(float(y[1, 2, 0].abs().sum()), 0)
        y[1, 2, 0] = 0
        self.assertEqual(float(y.abs().sum()), 0)

    def test_equal_dynamic_teacher_is_not_penalized_for_motion(self):
        teacher = torch.randn(1, 4, 3, 8, 8)
        terms = m10.high_frequency_terms(teacher, teacher, self.hp)
        self.assertTrue(all(float(value) == 0 for value in terms.values()))

    def test_fixed_vs_flickering_high_frequency_error(self):
        pattern = (torch.arange(8)[None, :] + torch.arange(8)[:, None]) % 2
        pattern = pattern.float()[None, None, None].expand(1, 4, 3, 8, 8)
        teacher = torch.zeros_like(pattern)
        fixed = m10.high_frequency_terms(pattern, teacher, self.hp)
        flickering = pattern * torch.tensor([1, -1, 1, -1]).view(1, 4, 1, 1, 1)
        changing = m10.high_frequency_terms(flickering, teacher, self.hp)
        self.assertGreater(float(fixed["hf_l1"]), 0)
        self.assertEqual(float(fixed["hf_temporal_l1"]), 0)
        self.assertGreater(float(changing["hf_temporal_l1"]), 0.1)

    def test_teacher_and_gt_do_not_receive_gradients(self):
        student = torch.randn(1, 4, 3, 8, 8, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        sv = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
        tv = torch.randn_like(sv, requires_grad=True)
        gt = torch.randn_like(student, requires_grad=True)
        out = {"prediction": student, "teacher_prediction": teacher, "velocity": sv,
               "router_balance_loss": sv.sum() * 0, "target": gt}
        terms = m10.recovery_objective(out, tv, weights=m10.RecoveryWeights(hf_temporal_l1=1), high_pass=self.hp)
        terms["loss"].backward()
        self.assertGreater(float(student.grad.abs().sum()), 0)
        self.assertGreater(float(sv.grad.abs().sum()), 0)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(tv.grad)
        self.assertIsNone(gt.grad)
        out["target"] = gt.detach() * 100
        again = m10.recovery_objective(out, tv, weights=m10.RecoveryWeights(hf_temporal_l1=1), high_pass=self.hp)
        torch.testing.assert_close(again["loss"], terms["loss"])

    def test_spatial_and_temporal_gates_differ_only_by_temporal_term(self):
        sv, tv = torch.randn(1, 4, 2, 2, 2), torch.randn(1, 4, 2, 2, 2)
        out = {"prediction": torch.randn(1, 4, 3, 8, 8), "teacher_prediction": torch.randn(1, 4, 3, 8, 8),
               "velocity": sv, "router_balance_loss": torch.tensor(1.1)}
        a = m10.recovery_objective(out, tv, weights=m10.RecoveryWeights(), high_pass=self.hp)
        b = m10.recovery_objective(out, tv, weights=m10.RecoveryWeights(hf_temporal_l1=1), high_pass=self.hp)
        torch.testing.assert_close(b["loss"] - a["loss"], a["hf_temporal_l1"])
        torch.testing.assert_close(a["teacher_selection_score"], b["teacher_selection_score"])

    def test_single_frame_temporal_term_is_zero(self):
        x = torch.randn(1, 1, 3, 8, 8, requires_grad=True)
        terms = m10.high_frequency_terms(x, torch.zeros_like(x), self.hp)
        self.assertEqual(float(terms["hf_temporal_l1"].detach()), 0)
        terms["hf_temporal_l1"].backward()
        self.assertIsNotNone(x.grad)

    def test_frozen_decoder_preserves_transformer_gradient_with_checkpoint(self):
        decoder = ToyDecoder().requires_grad_(False).eval()
        transformer = nn.Conv3d(4, 4, 1)
        z = torch.randn(1, 4, 3, 8, 8)
        gradients = []
        for recompute in (False, True):
            transformer.zero_grad(set_to_none=True)
            rgb = m10.decode_student(decoder, z, transformer(z), output_frames=3, checkpointing=recompute)
            terms = m10.high_frequency_terms(rgb, torch.zeros_like(rgb), self.hp)
            (terms["hf_l1"] + terms["hf_temporal_l1"]).backward()
            gradients.append(transformer.weight.grad.clone())
        torch.testing.assert_close(gradients[0], gradients[1])
        self.assertGreater(float(gradients[1].abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))

    def test_bad_weights_and_shapes_rejected(self):
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                m10.RecoveryWeights(hf_l1=weight)
        with self.assertRaises(ValueError):
            m10.high_frequency_terms(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8), self.hp)

    def test_teacher_cache_lineage_validation(self):
        good = {"kind": m10.STAGE_A_CACHE_KIND, "teacher_delta_step": 200000,
                "reae_sha256": "encoder", "teacher_delta_weights_sha256": "stagea"}
        m10.validate_teacher_metadata(good, good, reae_sha256="encoder")
        for key, value in (("kind", "swiftvr_b2b_d1536_ta_velocity"),
                           ("teacher_delta_step", 24000), ("reae_sha256", "other"),
                           ("teacher_delta_weights_sha256", "other")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                m10.validate_teacher_metadata(good, {**good, key: value}, reae_sha256="encoder")

    def test_real_a1_input_backward(self):
        try:
            from swiftvr.models.m9_factorized_decoder import M9A1FactorizedReAEDecoder, CausalTemporalAdapter
        except ModuleNotFoundError as exc:
            self.skipTest(f"Full SwiftVR dependencies not installed: {exc}")
        decoder = M9A1FactorizedReAEDecoder()
        for layer in decoder.modules():
            if isinstance(layer, CausalTemporalAdapter):
                nn.init.normal_(layer.conv2.weight, std=0.01)
        decoder.requires_grad_(False).eval()
        z = torch.randn(1, 48, 4, 2, 2)
        velocity = torch.randn_like(z, requires_grad=True)
        rgb = m10.decode_student(decoder, z, velocity, output_frames=13)
        self.assertEqual(tuple(rgb.shape), (1, 13, 3, 32, 32))
        terms = m10.high_frequency_terms(rgb, torch.zeros_like(rgb), self.hp)
        (terms["rgb_l1"] + terms["hf_l1"] + terms["hf_temporal_l1"]).backward()
        self.assertTrue(bool(torch.isfinite(velocity.grad).all()))
        self.assertGreater(float(velocity.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))


class M10EntrypointTests(unittest.TestCase):
    def test_local_inputs_and_immutable_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("base/reae.safetensors", "student/transformer/config.json", "student/transformer/model.safetensors",
                         "decoder/config.json", "decoder/model.safetensors", "train/metadata.json", "val/metadata.json"):
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("{}")
            argv = []
            for flag, value in (("--base-checkpoint", "base"), ("--student-init", "student"),
                                ("--decoder-checkpoint", "decoder"), ("--teacher-cache", "train"),
                                ("--val-teacher-cache", "val"), ("--output-dir", "out")):
                argv += [flag, str(root / value)]
            args = entry.build_parser().parse_args(argv)
            entry._check_args(args)
            args.output_dir = root / "student/new_run"
            with self.assertRaises(ValueError):
                entry._check_args(args)
            args.output_dir = root / "out"
            args.output_dir.mkdir()
            (args.output_dir / "keep.txt").write_text("immutable")
            with self.assertRaises(FileExistsError):
                entry._check_args(args)
            self.assertEqual((args.output_dir / "keep.txt").read_text(), "immutable")


if __name__ == "__main__":
    unittest.main()
