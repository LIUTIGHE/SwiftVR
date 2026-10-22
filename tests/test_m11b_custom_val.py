"""CPU guardrails for reusable cached four-way custom video validation.

No GPU inference in this test module. Tests path provenance, source-frame
identity contracts and lossless five-way A/B native ROI assembly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.m11b_custom_val import _png_files,load_bank
from tools.train_with_custom_val import cli_arguments,milestones,verified_checkpoint
from tools.compare_m11b_t1b_ab import run as ab_run


class CustomValContracts(unittest.TestCase):
    def test_milestones_and_non_mutable_train_args(self):
        self.assertEqual(milestones("125,250,500"),[125,250,500])
        for bad in ("250,125","0,250","250,250","-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                milestones(bad)
        self.assertEqual(cli_arguments({"variant":"A","gradient_accumulation_steps":8,
                                        "gt_lpips_weight":0}),
                         ["--variant","A","--gradient-accumulation-steps","8",
                          "--gt-lpips-weight","0"])
        for bad in ({"resume":"any"},{"max_steps":250},{"output_dir":"somewhere"},
                    {"smoke_steps":0}):
            with self.assertRaisesRegex(ValueError,"controlled"):
                cli_arguments(bad)

    def test_strict_current_png_numbering(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i in range(8):
                Image.new("RGB",(12,12)).save(root/f"{i:05d}.png")
            self.assertEqual(len(_png_files(root,8)),8)
            (root/"00004.png").unlink()
            with self.assertRaisesRegex(ValueError,"contiguous"):
                _png_files(root,8)

    def test_reference_bank_changed_file_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            sources=[]
            for name in ("original","input","basic"):
                path=root/name
                path.write_bytes(b"hello")
                stat=path.stat()
                sources.append(dict(path=str(path),size=stat.st_size,
                                    mtime_ns=stat.st_mtime_ns,sha256="placeholder"))
            plan={"kind":"m8_task_custom_reference_v1",
                  "inference":dict(upscale=3,clip_len=24,dit_overlap=0,dtype="bfloat16",
                                   attention_backend="sdpa"),
                  "original_files":[sources[0]],
                  "clips":[dict(name=name,input=sources[1],basic=sources[2],
                                original=str(root/"png"),frames=249,roi=[0,0,4,4])
                           for name in ("Wetland1","Mangrove1")]}
            (root/"reference.json").write_text(json.dumps(plan))
            self.assertEqual(len(load_bank(root)["clips"]),2)
            with (root/"input").open("ab") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError,"changed"):
                load_bank(root)

    def test_checkpoint_must_be_complete(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            ckpt=run/"checkpoints"/"step_00000125"
            (ckpt/"tiny_decoder").mkdir(parents=True)
            (ckpt/"metadata.json").write_text(json.dumps({"step":125}))
            (ckpt/"tiny_decoder"/"model.safetensors").write_bytes(b"fake")
            with self.assertRaisesRegex(ValueError,"Incomplete"):
                verified_checkpoint(run,125)
            (ckpt/"training_state.pt").write_bytes(b"fake")
            self.assertEqual(verified_checkpoint(run,125),ckpt)

    def test_ab_native_roi_fiveway_and_phase_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            a,b=root/"a",root/"b"
            roi=[0,0,4,4]
            phase={
                str(p):{"hf_energy_current":0.1+p/100,
                        "hf_energy_e99":0.2,
                        "hf_energy_original":0.22,
                        "hf_energy_tiny":0.15,
                        "hf_mae_current_vs_e99":0.01,
                        "luma_delta_current_vs_e99":0.001}
                for p in range(4)
            }
            for directory in (a,b):
                for scene in ("Wetland1","Mangrove1"):
                    output=directory/scene
                    cmp=output/"comparison"
                    cmp.mkdir(parents=True)
                    provenance={"source":{"sha256":"same"},
                                "m11b_transformer_sha256":"same",
                                "reae_sha256":"same",
                                "full_origin_frame_count":249,
                                "inference":{"clip_len":24}}
                    summary={"frames":249,"roi":roi,
                             "phase_formula":"(output_frame+3)%4",
                             "by_phase":phase,
                             "hf_ratio_current_vs_e99_phase_means":{
                                 str(p):phase[str(p)]["hf_energy_current"]/0.2
                                 for p in range(4)}}
                    (output/"provenance.json").write_text(json.dumps(provenance))
                    (cmp/"summary.json").write_text(json.dumps(summary))
                    frames=(245,248) if scene=="Wetland1" else (12,124,245,248)
                    for frame in frames:
                        Image.new("RGB",(8,64),"red").save(
                            cmp/f"native_fourway_{frame:06d}_p{(frame+3)%4}.png")
            result=ab_run(argparse.Namespace(root_a=a,root_b=b,output=root/"out"))
            self.assertEqual(len(result["scenes"]),2)
            target=root/"out"/"Wetland1_global_000245_p0_fiveway.png"
            self.assertTrue(target.is_file())
            self.assertEqual(Image.open(target).size,(20,32))


if __name__=="__main__":
    unittest.main()
