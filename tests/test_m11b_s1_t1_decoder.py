"""CPU contracts for the M11B-S1-T1 fixed-latent decoder training entry."""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools import train_m11b_s1_t1_decoder as trainer
from swiftvr.training.m11b_decoder_pairs import (
    M11BPairedLatentDataset, verified_cache_metadata,
)


class PhaseT1Tests(unittest.TestCase):
    def test_balanced_sampler_determinism_and_resume(self):
        args=dict(legacy=11,ultra=17,batch_size=2,rank=0,world=2,seed=20261021)
        s=list(trainer.BalancedBatchSampler(start=0,stop=13,**args))
        recovered=list(trainer.BalancedBatchSampler(start=5,stop=13,**args))
        self.assertEqual(s[5:],recovered)
        self.assertEqual(len(s),13)
        for indices in s:
            self.assertEqual(len(indices),2)
            self.assertEqual(sum(i<11 for i in indices),1)
            self.assertEqual(sum(i>=11 for i in indices),1)

    def test_one_item_ranks_cover_both_domains(self):
        args=dict(legacy=11,ultra=17,batch_size=1,world=2,seed=15)
        a=list(trainer.BalancedBatchSampler(start=0,stop=4,rank=0,**args))
        b=list(trainer.BalancedBatchSampler(start=0,stop=4,rank=1,**args))
        self.assertTrue(all(x[0]<11 for x in a))
        self.assertTrue(all(x[0]>=11 for x in b))

    def test_schedule_not_restarted_during_continuation(self):
        a=argparse.Namespace(
            warmup_steps=100,schedule_steps=6000,learning_rate=1e-5,min_lr_ratio=.1)
        self.assertAlmostEqual(trainer.lr_for_update(1,a),1e-7)
        self.assertAlmostEqual(trainer.lr_for_update(100,a),1e-5)
        self.assertLess(trainer.lr_for_update(1000,a),1e-5)
        self.assertAlmostEqual(trainer.lr_for_update(6000,a),1e-6)

    def test_no_m11b_model_or_encoder_forward_in_trainer(self):
        # T1 should consume cached z_SR and never load a trainable Transformer.
        tree=ast.parse(inspect.getsource(trainer))
        used={node.id for node in ast.walk(tree) if isinstance(node,ast.Name)}
        self.assertNotIn("WanTransformer3DModelPromptFreeNoTimeMoE",used)
        self.assertIn("decode_reae_clip",used)
        self.assertIn("M9A1FactorizedReAEDecoder",used)
        self.assertIn("M11BPairedLatentDataset",used)

    def test_cache_rejects_changed_source_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            base=root/"base";(base).mkdir()
            model=root/"model";model.mkdir()
            cache=root/"cache";cache.mkdir()
            file=root/"weights"
            file.write_bytes(b"initial")
            from tools.m11b_s1_d1_cache import sha256
            metadata={
                "kind":"M11B-S1-D1-restored-latent-v1","phase":"M11B-S1-D1",
                "complete":True,"samples":[{}],"sample_count":1,
                "dataset_count":1,"selected_count":1,"pilot":False,"partial_ultra":False,
                "source_checkpoint":str(model.resolve()),"base_checkpoint":str(base.resolve()),
                "source_files":{str(file):sha256(file)}
            }
            plan={k:v for k,v in metadata.items() if k not in ("complete","samples","sample_count","kind")}\n            plan["kind"]="M11B-S1-D1-restored-latent-plan-v1"
            (cache/"metadata.json").write_text(json.dumps(metadata))
            (cache/"plan.json").write_text(json.dumps(plan))
            self.assertEqual(verified_cache_metadata(cache,model_root=model,reae_root=base)["sample_count"],1)
            file.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError,"hash changed"):
                verified_cache_metadata(cache,model_root=model,reae_root=base)

    def test_loader_checks_same_pair_and_shape(self):
        class FakeDataset:
            def __getitem__(self,index):
                return {
                    "lr":torch.zeros(13,3,128,128),
                    "hr":torch.zeros(13,3,384,384),
                    "pair_id":"identity0","domain":"legacy",
                }
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"samples").mkdir()
            latent=torch.ones(4,48,24,24,dtype=torch.float16)
            save_file({"z_sr":latent},str(root/"samples"/"a.safetensors"))
            obj=M11BPairedLatentDataset.__new__(M11BPairedLatentDataset)
            obj.root=root
            obj.dataset=FakeDataset()
            obj.meta={"domain":"legacy"}
            obj.rows=[{
                "pair_id":"identity0","domain":"legacy","file":"samples/a.safetensors",
                "lr_shape":[13,3,128,128],"gt_shape":[13,3,384,384],
                "z_sr_shape":[4,48,24,24]
            }]
            result=obj[0]
            self.assertEqual(tuple(result["z_sr"].shape),(4,48,24,24))
            obj.rows[0]["pair_id"]="wrong"
            with self.assertRaisesRegex(ValueError,"identity mismatch"):
                obj[0]


if __name__=="__main__":
    unittest.main()
