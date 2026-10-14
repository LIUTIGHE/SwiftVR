"""CPU-only T1 contracts: exact global-index pairing, cache identity, loss gradients.

Real checkpoint, LPIPS, Original ReAE, DDP, and custom video smoke must run on
the user's CUDA server. Tests never download weights or open 4K videos.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_m8_t1_teacher_kd import (  # noqa: E402
    TeacherPairedView, validate_ultra_teacher_views, teacher_band_energy_loss,
    build_parser,
)


def make_identity(*, idx, materialized=None):
    return {
        "sample_id":"source:0", "record_uid":"record_0", "variant":"train",
        "frame_indices":torch.arange(13), "crop_top":2, "crop_left":5,
        "horizontal_flip":False, "vertical_flip":False,
        "distillation_index":idx, "distillation_view_index":3,
        "distillation_view_seed":2026,
        "scale":3,
        "lr":torch.ones(13,3,4,4),
        "materialized_index":idx if materialized is None else materialized,
    }


class T1Contracts(unittest.TestCase):
    def test_legacy_keeps_view_identity_and_hr(self):
        row=make_identity(idx=12)
        row["hr"]=torch.ones(13,3,12,12)
        parent=SimpleNamespace(dataset=[row],domain="legacy",hr_rows=None)
        pair=TeacherPairedView(parent)[0]
        self.assertEqual(pair["distillation_index"],12)
        self.assertEqual(pair["teacher_cache_domain"],"legacy")
        self.assertEqual(tuple(pair["frame_indices"].shape),(13,))
        torch.testing.assert_close(pair["hr"],row["hr"])
        self.assertEqual(pair["pair_id"],"legacy:record_0:12")

    def test_ultra_uses_global_not_local_index(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"hr.safetensors"
            save_file({"hr":torch.full((13,3,12,12),128,dtype=torch.uint8)},str(path))
            row=make_identity(idx=0,materialized=37)
            hr_row={"materialized_index":37,"hr_bundle_path":str(path),
                    "hr_bundle_key":"hr","lr_row_sha256":"sha-row-37"}
            parent=SimpleNamespace(dataset=[row],domain="ultravideo",hr_rows=[hr_row])
            pair=TeacherPairedView(parent)[0]
            self.assertEqual(pair["distillation_index"],37)
            self.assertEqual(pair["teacher_cache_domain"],"ultravideo")
            self.assertEqual(pair["pair_id"],"sha-row-37")
            self.assertAlmostEqual(float(pair["hr"].mean()),128/255,places=6)
            parent.hr_rows=[{**hr_row,"materialized_index":38}]
            with self.assertRaises(ValueError):
                TeacherPairedView(parent)[0]

    def test_full_metadata_alignment_strict(self):
        row=make_identity(idx=0)
        item={k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in row.items()}
        item["view_index"]=item.pop("distillation_view_index")
        item["view_seed"]=item.pop("distillation_view_seed")
        item.pop("lr")
        materialized=dict(item)
        cache=SimpleNamespace(
            metadata={"dataset_kind":"ultravideo_materialized_lr_v1",
                      "materialized_manifest_sha256":"hash"},
            samples_by_index={0:dict(item)})
        shard=SimpleNamespace(dataset=SimpleNamespace(rows=[materialized]))
        meta=[{"source_manifest_sha256":"hash"}]
        validate_ultra_teacher_views(cache,[shard],meta,expected_count=1)
        cache.samples_by_index[0]["crop_left"]+=1
        with self.assertRaisesRegex(ValueError,"crop_left"):
            validate_ultra_teacher_views(cache,[shard],meta,expected_count=1)
        with self.assertRaises(ValueError):
            validate_ultra_teacher_views(cache,[shard],
                                         [{"source_manifest_sha256":"other"}],
                                         expected_count=1)

    def test_teacher_texture_grad_and_alignment(self):
        torch.manual_seed(47)
        x=(torch.rand(1,3,3,32,32)*0.15).requires_grad_(True)
        teacher=torch.rand_like(x)
        gt=torch.rand_like(x)
        loss=teacher_band_energy_loss(x,teacher,gt)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(x.grad.abs().sum()),0)
        with self.assertRaises(ValueError):
            teacher_band_energy_loss(x,teacher[:,:2],gt)
        torch.testing.assert_close(teacher_band_energy_loss(gt,gt,gt),
                                   torch.zeros((),dtype=torch.float32),atol=1e-7,rtol=0)

    def test_cli_contract_and_no_new_inference_module(self):
        a=build_parser().parse_args([
            "--base-checkpoint","BASE","--student-init","P1P5000",
            "--decoder-checkpoint","A1","--legacy-view-cache","L",
            "--ultra-teacher-cache","U","--val-view-cache","V",
            "--ultra-pairs","S0","--output-dir","T1",
            "--recipe","teacher_perceptual"])
        self.assertEqual(a.teacher_lpips_weight,0.2)
        self.assertEqual(a.teacher_texture_weight,0.1)
        self.assertEqual(a.max_steps,5000)
        self.assertEqual(a.recipe,"teacher_perceptual")
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--recipe","wrong"])

    def test_compiles(self):
        import py_compile
        py_compile.compile(str(ROOT/"tools/train_m8_t1_teacher_kd.py"),doraise=True)


if __name__=="__main__":
    unittest.main()
