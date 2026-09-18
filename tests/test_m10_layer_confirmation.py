"""CPU contracts for expanded two-mask confirmation and exact continuation."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import m10_layer_confirmation as confirm
from tools import select_m10_layers as ls
from tools.m10_layer_search_core import training_order


OLD = [0,3,4,5,6,7,8,11,13,16,17,18,20,23,24,25,26,27,28,29]
NEW = [0,1,3,4,5,6,7,8,13,16,17,18,19,20,23,24,25,27,28,29]


def selection(root):
    candidates = [{"id": cid, "steps": 250, "kept_source_blocks": mask,
                   "init_checkpoint": str(root / cid / "init"),
                   "checkpoint": str(root / cid / "step_00000250")}
                  for cid, mask in (("heuristic", OLD), ("swap_bd14890ed632", NEW))]
    return {"selected": candidates[1], "candidates": candidates}


class Packets:
    def __init__(self):
        self.data = [{"z": torch.tensor([[v, 1.0]]), "v": torch.tensor([[2*v + .5]])}
                     for v in (.1, .2, .3, .4, .5)]
        self.calls = []

    def get(self, i):
        self.calls.append(i)
        return self.data[i]


def toy_velocity(model, z, dtype, grad=False):
    # Include dropout to exercise RNG restoration, rather than only deterministic weights.
    v = model(z)
    return v, v.sum() * 0


class ConfirmationTests(unittest.TestCase):
    def test_only_two_locked_init_paths_not_short_recovery(self):
        s = selection(Path('/tmp/source'))
        s['candidates'].append({'id': 'unselected_other'})
        pair = confirm.locked_pair(s)
        self.assertEqual([r['role'] for r in pair], ['heuristic', 'selected'])
        self.assertEqual(pair[1]['id'], 'swap_bd14890ed632')
        self.assertTrue(all(Path(r['init_checkpoint']).name == 'init' for r in pair))
        self.assertEqual(pair[0]['kept_source_blocks'], OLD)
        self.assertEqual(pair[1]['kept_source_blocks'], NEW)
        s['candidates'][1]['init_checkpoint'] = s['candidates'][1]['checkpoint']
        with self.assertRaises(ValueError):
            confirm.locked_pair(s)

    def test_duplicate_or_identical_candidate_rejected(self):
        for mutation in ('same_id', 'same_mask'):
            s = selection(Path('/tmp/source'))
            if mutation == 'same_id':
                s['selected'] = s['candidates'][0]
            else:
                s['candidates'][1]['kept_source_blocks'] = OLD
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                confirm.locked_pair(s)

    def test_expand_all_views_excluding_entire_reserved_sources(self):
        samples = [{'sample_id': f'src{i}', 'distillation_index': i*6+j,
                    'variant': j//3} for i in range(8) for j in range(6)]
        split = {'source_groups': {'probe': ['src0'], 'rank': ['src1']},
                 'excluded_validation_sources': ['src2'], 'probe': [0], 'rank': [6], 'adapt': [24, 30]}
        result = confirm.training_indices(samples, [{'sample_id': 'src3'}], split)
        self.assertEqual(result['indices'], list(range(24,48)))
        self.assertEqual(result['views'], 24)
        self.assertEqual(result['sources'], 4)
        self.assertGreater(result['views'], len(split['adapt']))
        self.assertEqual(result, confirm.training_indices(samples[::-1], [{'sample_id': 'src3'}], split))
        split['adapt'] = [0]
        with self.assertRaises(ValueError):
            confirm.training_indices(samples, [], split)

    def test_data_schedule_extends_without_restart_or_skipping(self):
        short = training_order(17, 1000, 4, 42)
        extended = training_order(17, 2000, 4, 42)
        self.assertEqual(extended[:len(short)], short)
        for start in (0,17,34):
            self.assertEqual(set(extended[start:start+17]), set(range(17)))
        self.assertEqual(extended[4000:4004], training_order(17,1001,4,42)[4000:4004])

    def test_confirm_parser_preserves_legacy_defaults(self):
        p = ls.build_parser()
        a = p.parse_args(['confirm','--work-dir','old','--output-dir','new'])
        self.assertEqual((a.steps,a.accumulation,a.warmup_steps,a.eval_every),(2000,4,25,500))
        self.assertFalse(a.resume)
        old = p.parse_args(['recover','--work-dir','old'])
        self.assertEqual(old.steps,250)
        self.assertEqual(old.accumulation,4)

    def test_bad_packet_index_rejected_before_cache_access(self):
        with tempfile.TemporaryDirectory() as d:
            obj = confirm.EncodedTrainingViews(None,None,None,Path(d),[1,2],torch.device('cpu'),torch.float32)
            with self.assertRaises(ValueError):
                obj.get(3)
            torch.save({'identity':{'distillation_index':3}},Path(d)/'00000001.pt')
            with self.assertRaises(ValueError):
                obj.get(1)

    def test_training_resume_restores_fp32_optimizer_rng_and_order(self):
        torch.manual_seed(6)
        original = nn.Sequential(nn.Linear(2,8), nn.Dropout(.2), nn.Linear(8,1))
        uninterrupted = copy.deepcopy(original)
        resumed = copy.deepcopy(original)
        full_packets, resumed_packets = Packets(), Packets()
        order = training_order(5,10,2,42)
        device = torch.device('cpu')

        def advance(model, optimizer, packets, start, end):
            for step in range(start,end):
                confirm.train_step(model,optimizer,packets,order[2*step:2*step+2],
                                   lr=.003,device=device,dtype=torch.float32)

        with patch.object(ls,'_velocity',side_effect=toy_velocity):
            torch.manual_seed(123)
            opt1=torch.optim.AdamW(uninterrupted.parameters(),lr=.003,weight_decay=0)
            advance(uninterrupted,opt1,full_packets,0,10)
            torch.manual_seed(123)
            opt2=torch.optim.AdamW(resumed.parameters(),lr=.003,weight_decay=0)
            advance(resumed,opt2,resumed_packets,0,4)
            before={k:v.clone() for k,v in resumed.state_dict().items()}
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'resume.pt'
                confirm.save_resume(path,resumed,opt2,step=4,plan_hash='plan',device=device)
                for k,v in resumed.state_dict().items():
                    torch.testing.assert_close(v,before[k],atol=0,rtol=0)
                restored=copy.deepcopy(original)
                opt3=torch.optim.AdamW(restored.parameters(),lr=.003,weight_decay=0)
                with self.assertRaises(ValueError):
                    confirm.load_resume(path,restored,opt3,plan_hash='different',device=device)
                torch.rand(100)
                step=confirm.load_resume(path,restored,opt3,plan_hash='plan',device=device)
                self.assertEqual(step,4)
                self.assertTrue(all(p.dtype==torch.float32 for p in restored.parameters()))
                advance(restored,opt3,resumed_packets,step,10)
        self.assertEqual(full_packets.calls,resumed_packets.calls)
        for k,v in uninterrupted.state_dict().items():
            torch.testing.assert_close(v,restored.state_dict()[k],atol=0,rtol=0)
        for state in opt3.state.values():
            self.assertEqual(state['exp_avg'].dtype,torch.float32)

    def test_compare_common_steps_not_independent_best(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            pair=[{'role':'heuristic'},{'role':'selected'}]
            for role,steps in (('heuristic',[0,1000,2000]),('selected',[0,1000])):
                for step in steps:
                    m={'lpips':.2-(.01 if role=='selected' else 0),'rgb_mse':.001,
                       'student_teacher_psnr':30.0}
                    path=out/role/'evaluations'/f'step_{step:08d}'/'metrics.json'
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({'step':step,'rank16':m,'val13':m}))
            confirm._comparison(out,pair,{'training':{'views':500},'recipe':{'accumulation':4},
                                           'baseline_reference_report':'old.json'})
            r=json.loads((out/'comparison.json').read_text())
            self.assertEqual(r['matched_steps'],[0,1000])
            self.assertEqual(r['comparisons'][1]['sample_presentations_per_candidate'],4000)
            self.assertEqual(r['comparisons'][1]['data_passes_equivalent'],8)
            self.assertAlmostEqual(r['comparisons'][1]['val13_selected_minus_heuristic']['lpips'],-.01)

    def test_end_to_end_cpu_workflow_fresh_then_resume(self):
        """Run the orchestration with synthetic models/data; no GPU quality claim."""
        # Initialize lazy optimizer modules before the temporary sys.modules fixture.
        import torch._dynamo
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);work=base/'search';out=base/'confirm';work.mkdir()
            inp=base/'inputs';inp.mkdir()
            roots={k:str(inp/k) for k in ('base','decoder','source','baseline','teacher_cache','val_teacher_cache')}
            for p in roots.values():
                Path(p).mkdir()
            src=[{'sample_id':f's{i}','distillation_index':i} for i in range(20)]
            val=[{'sample_id':f'v{i}','distillation_index':i} for i in range(13)]
            split={'source_groups':{'probe':['s0'],'rank':['s1']},'excluded_validation_sources':[],
                   'probe':[0],'rank':[1],'adapt':[2,3]}
            cfg={**roots,'immutable_sha256':{},'splits':split,'dtype':'float32','seed':3,'path_root':str(base)}
            (work/'run_config.json').write_text(json.dumps(cfg))
            sel=selection(work/'recovery');(work/'selection.json').write_text(json.dumps(sel))
            for c in sel['candidates']:
                root=Path(c['init_checkpoint']);(root/'transformer').mkdir(parents=True)
                (root/'selection_lineage.json').write_text(json.dumps(c))
                (root/'transformer/config.json').write_text('{}')
                (root/'transformer/model.safetensors').write_bytes(b'fake')
            (work/'formal_val13').mkdir();(work/'formal_val13/report.json').write_text('{}')
            # Actual packet identity validation, plus fake metrics for model evaluation.
            for folder,indices in ((work/'packets/rank',[1]),(work/'formal_val13/packets',range(13))):
                folder.mkdir(parents=True)
                for n,i in enumerate(indices):
                    torch.save({'identity':{'distillation_index':i}},folder/f'{n:05d}.pt')
            class Cache:
                def __init__(self,path):
                    self.metadata={'split':'train' if str(path)==roots['teacher_cache'] else 'val',
                                   'samples':src if str(path)==roots['teacher_cache'] else val}
                    self.samples_by_index={r['distillation_index']:r for r in self.metadata['samples']}
            class SimplePackets:
                def __init__(self,*args):pass
                def get(self,i):return {'z':torch.tensor([[i/20.,1.]]),'v':torch.tensor([[i/10.+.5]])}
            module=types.ModuleType('swiftvr.training')
            module.TeacherVelocityCache=Cache
            module.build_fp32_adamw=lambda m,learning_rate,weight_decay,eps:torch.optim.AdamW(m.parameters(),lr=learning_rate,weight_decay=weight_decay,eps=eps)
            module.cast_trainable_parameters=lambda m,dtype:m.to(dtype=dtype)
            reference=types.ModuleType('swiftvr.training.reference')
            reference.sha256_file=lambda p:'hash'
            fake_fingerprint=lambda paths:{str(p):'hash' for p in paths}
            decoder=nn.Linear(1,1).requires_grad_(False)
            reae=nn.Module();reae.decoder=nn.Linear(1,1)
            def snapshot(model,path,dtype,metadata):
                path.mkdir(parents=True,exist_ok=True)
                (path/'metadata.json').write_text(json.dumps(metadata))
            def evaluation(folder,step,checkpoint,*args):
                m={'lpips':.1,'rgb_mse':.001,'student_teacher_psnr':30.0}
                root=folder/'evaluations'/f'step_{step:08d}';root.mkdir(parents=True,exist_ok=True)
                (root/'metrics.json').write_text(json.dumps({'step':step,'rank16':m,'val13':m}))
            patches={'swiftvr.training':module,'swiftvr.training.reference':reference}
            with patch.dict(sys.modules,patches), patch.object(ls,'_verify'), patch.object(ls,'_fingerprint',side_effect=fake_fingerprint), \
                 patch.object(ls,'_local_perceptual',return_value=(nn.Identity(),Path('fake'))), \
                 patch.object(ls,'_frozen',return_value=(reae,decoder)), patch.object(ls,'_dataset',return_value=[]), \
                 patch.object(ls,'_load_model',side_effect=lambda *args:nn.Linear(2,1)), \
                 patch.object(ls,'_velocity',side_effect=toy_velocity),patch.object(confirm,'EncodedTrainingViews',SimplePackets), \
                 patch.object(confirm,'_snapshot',side_effect=snapshot),patch.object(confirm,'_evaluate_step',side_effect=evaluation):
                argv=['confirm','--work-dir',str(work),'--output-dir',str(out),'--steps','2','--eval-every','1','--warmup-steps','0','--visual-samples','0']
                args=ls.build_parser().parse_args(argv)
                confirm.run(args,torch.device('cpu'))
                r=json.loads((out/'comparison.json').read_text())
                self.assertEqual(r['matched_steps'],[0,1,2])
                self.assertEqual(r['train_views'],18)
                args.steps=4;args.resume=True
                confirm.run(args,torch.device('cpu'))
                r=json.loads((out/'comparison.json').read_text())
                self.assertEqual(r['matched_steps'],[0,1,2,3,4])
                args.learning_rate=.003
                with self.assertRaises(ValueError):
                    confirm.run(args,torch.device('cpu'))
            self.assertTrue(all(p.grad is None for p in decoder.parameters()))


if __name__=='__main__':
    unittest.main()
