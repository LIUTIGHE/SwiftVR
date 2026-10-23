"""CPU unit contracts plus optional canonical repository crop integration.

FakeBackbone/Decoder test gradient routing, NOT actual checkpoint quality.
Real CUDA/ReAE/M8/A1 integration is tools/train_m8_task_posttrain.py --smoke-steps.
"""
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch import nn
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

TASK = load("m8_task_unit", "swiftvr/training/m8_task.py")
PAIRS = load("m8_pairs_unit", "swiftvr/data/task_pairs.py")
TRAIN = load("m8_train_unit", "tools/train_m8_task_posttrain.py")

class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.reae = nn.Conv2d(3, 3, 1)
        self.transformer = nn.Conv2d(3, 3, 1)
    def forward(self, batch):
        b,t,c,h,w = batch['lr'].shape
        z = self.reae(batch['lr'].flatten(0,1)).reshape(b,t,c,h,w).permute(0,2,1,3,4)
        v = self.transformer(z.permute(0,2,1,3,4).flatten(0,1)).reshape(b,t,c,h,w).permute(0,2,1,3,4)
        return dict(z_lq=z,velocity=v,target=batch.get('hr'),router_balance_loss=v.square().mean())

class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__(); self.conv=nn.Conv2d(3,3,1)
    def forward(self,z,*,output_frames,clamp):
        b,t,c,h,w=z.shape
        out=self.conv(z.flatten(0,1)).reshape(b,t,c,h,w)[:,:output_frames]
        return out.clamp(0,1) if clamp else out

class TaskObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.batch=dict(lr=torch.rand(1,5,3,4,4),hr=torch.rand(1,5,3,4,4))

    def test_gt_alone_reaches_transformer_not_encoder_or_a1(self):
        for recompute in (False,True):
            model=TASK.M8TaskForward(FakeBackbone(),FakeDecoder(),checkpoint_decoder=recompute)
            out=model(self.batch)
            terms=TASK.task_objective(out,lpips_weight=0,router_weight=0)
            terms['gt_loss'].backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.transformer.parameters()))
            self.assertTrue(all(not p.requires_grad and p.grad is None for m in (model.backbone.reae,model.decoder) for p in m.parameters()))
            model.train()
            self.assertFalse(model.decoder.training); self.assertFalse(model.backbone.reae.training)

    def test_checkpointing_same_prediction_and_gradient(self):
        a=TASK.M8TaskForward(FakeBackbone(),FakeDecoder(),checkpoint_decoder=False)
        b=TASK.M8TaskForward(FakeBackbone(),FakeDecoder(),checkpoint_decoder=True)
        b.load_state_dict(a.state_dict())
        pa,pb=a(self.batch),b(self.batch)
        torch.testing.assert_close(pa['prediction'],pb['prediction'])
        for out in (pa,pb): TASK.task_objective(out,lpips_weight=0)['loss'].backward()
        for x,y in zip(a.transformer.parameters(),b.transformer.parameters()): torch.testing.assert_close(x.grad,y.grad)

    def test_temporal_is_gt_relative_not_static_smoothing(self):
        gt=self.batch['hr']; pred=gt+0.1
        out=dict(prediction=pred,target=gt,router_balance_loss=gt.new_zeros(()))
        result=TASK.task_objective(out,lpips_weight=0)
        self.assertLess(float(result['temporal_mse']),1e-12)
        pred=pred.clone(); pred[:,2]+=0.2
        result=TASK.task_objective(dict(out,prediction=pred),lpips_weight=0)
        self.assertGreater(float(result['temporal_mse']),0.001)
        expected=((pred[:,1:]-pred[:,:-1])-(gt[:,1:]-gt[:,:-1])).square().mean()
        torch.testing.assert_close(expected,result['temporal_mse'])

    def test_missing_gt_fails(self):
        model=TASK.M8TaskForward(FakeBackbone(),FakeDecoder())
        with self.assertRaisesRegex(ValueError,'real HR'): model(dict(lr=self.batch['lr']))

    def test_missing_perceptual_fails(self):
        out=TASK.M8TaskForward(FakeBackbone(),FakeDecoder())(self.batch)
        with self.assertRaisesRegex(ValueError,'perceptual'): TASK.task_objective(out)

    def test_lpips_term_changes_gt_gradient(self):
        pred=torch.rand_like(self.batch['hr'],requires_grad=True)
        out=dict(prediction=pred,target=self.batch['hr'],router_balance_loss=pred.new_zeros(()))
        feature=lambda a,b: (a*2-b*2).square().mean()
        x=TASK.task_objective(out,lpips_weight=0)['gt_loss']
        y=TASK.task_objective(out,perceptual=feature,lpips_weight=0.3)['gt_loss']
        gx=torch.autograd.grad(x,pred,retain_graph=True)[0]
        gy=torch.autograd.grad(y,pred)[0]
        self.assertGreater(float((gx-gy).abs().sum()),0)

    def test_invalid_weights_rejected(self):
        out=dict(prediction=self.batch['lr'],target=self.batch['hr'],router_balance_loss=torch.tensor(0.))
        for weight in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): TASK.task_objective(out,lpips_weight=weight)

    def test_local_lpips_requires_complete_state(self):
        class Metric(nn.Module):
            def __init__(self): super().__init__(); self.weight=nn.Parameter(torch.tensor(1.0))
            def forward(self,a,b): return ((a-b)*self.weight).square().mean((1,2,3),keepdim=True)
        calls=[]
        def factory(**kwargs): calls.append(kwargs); return Metric()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(sys.modules,{'lpips':types.SimpleNamespace(LPIPS=factory)}):
            p=Path(d)/'complete.pt'; torch.save(Metric().state_dict(),p)
            loss=TASK.LocalLPIPS(p,frame_batch=2)
            self.assertTrue(calls[0]['pnet_rand']); self.assertFalse(calls[0]['pretrained'])
            pred=self.batch['lr'].clone().requires_grad_()
            loss(pred,self.batch['hr']).backward()
            self.assertIsNotNone(pred.grad)
            self.assertTrue(all(p.grad is None for p in loss.parameters()))
            torch.save({},p)
            with self.assertRaises(RuntimeError): TASK.LocalLPIPS(p)

    def test_phase_numbering_and_geometry(self):
        gt=torch.zeros(1,13,3,2,2); pred=gt.clone(); pred[:,1::4]=1
        phase=TASK.phase_errors(pred,gt)
        self.assertEqual(float(phase['phase_0_mse']),1.)
        self.assertEqual(float(phase['phase_3_mse']),0.)

class PairTests(unittest.TestCase):
    def test_lr_identity_binds_crop_frames_flip(self):
        row=dict(frame_indices=[1,3,5],horizontal_flip=False,crop_top=3,bundle_path='lr')
        digest=PAIRS.lr_row_hash(row)
        self.assertEqual(digest,PAIRS.lr_row_hash(dict(row,hr_bundle_path='new',hr_bundle_key='x',lr_row_sha256=digest)))
        self.assertNotEqual(digest,PAIRS.lr_row_hash(dict(row,horizontal_flip=True)))
        self.assertNotEqual(digest,PAIRS.lr_row_hash(dict(row,frame_indices=[2,4,6])))

    def test_paired_hr_is_not_lr_upsampling(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'hr.safetensors'; hr=torch.randint(0,256,(5,3,12,12),dtype=torch.uint8)
            save_file({'x':hr},str(path))
            canonical=[dict(lr=torch.rand(5,3,4,4),scale=3)]
            rows=[dict(hr_bundle_path=str(path),hr_bundle_key='x',lr_row_sha256='id')]
            result=PAIRS.TaskViewDataset(canonical,'ultra',hr_rows=rows)[0]
            torch.testing.assert_close(result['hr'],hr.float()/255)
            self.assertEqual(result['pair_id'],'id')

    @unittest.skipUnless((ROOT/'tools/materialize_ultravideo_lr_bundles.py').is_file(),'complete repository crop helper unavailable in isolated container')
    def test_hr_crop_downsamples_to_exact_canonical_hq(self):
        helper=load('tools.materialize_ultravideo_lr_bundles','tools/materialize_ultravideo_lr_bundles.py')
        for factor in (1,2):
            raw=np.random.default_rng(4).integers(0,256,(40*factor,48*factor,3),dtype=np.uint8)
            row=dict(canonical_hr_width=48,canonical_hr_height=40,scale=3,crop_box_hq=[2,3,5,6])
            hr=PAIRS.clean_hr_crop(raw,row)
            self.assertEqual(hr.size,(18,15))
            hq=helper._clean_hq_crop(raw,row)
            np.testing.assert_array_equal(np.asarray(hq),np.asarray(hr.resize((6,5),Image.Resampling.BOX)))

class EntryTests(unittest.TestCase):
    def test_schedule_independent_of_stop(self):
        self.assertEqual(TRAIN.lr_scale(50,100,10000),0.5)
        self.assertAlmostEqual(TRAIN.lr_scale(10000,100,10000),0.1)
        self.assertGreater(TRAIN.lr_scale(1500,100,10000),0.9)

    def test_help_does_not_need_model_dependencies(self):
        for name in ('train_m8_task_posttrain.py','materialize_ultravideo_hr_targets.py'):
            result=subprocess.run([sys.executable,str(ROOT/'tools'/name),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg unavailable')
    def test_native_custom_composer_pixel_geometry_and_frame_mapping(self):
        # Exercise the actual shell's composition block with synthetic frame sources.
        # This tests FFV1/ROI/time-index logic, not MP4/decord or model inference.
        script=(ROOT/'tools/visual_check_m8_task.sh').read_text()
        block=script.rsplit("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); bank=root/'bank'; out=root/'out'; bank.mkdir()
            folder=out/'Wetland1'; folder.mkdir(parents=True)
            current=folder/'current_png'; current.mkdir()
            sources={str(root/'lq'): [np.full((4,4,3),k+1,np.uint8) for k in range(5)],
                     str(root/'original'): [np.full((12,12,3),20+k,np.uint8) for k in range(5)],
                     str(current): [np.full((12,12,3),40+k,np.uint8) for k in range(5)],
                     str(root/'tiny'): [np.full((12,12,3),60+k,np.uint8) for k in range(10)]}
            class Source:
                def __init__(self,path):
                    self.frames=sources[str(path)]; self.height,self.width=self.frames[0].shape[:2]; self.fps=30
                def __len__(self): return len(self.frames)
                def frame(self,i): return self.frames[i]
            def resize(a,w,h): return np.asarray(Image.fromarray(a).resize((w,h),Image.Resampling.BICUBIC))
            def mapping(i,*,basic_count,reference_count):
                return round(i*(basic_count-1)/(reference_count-1))
            m=dict(clips=[dict(name='Wetland1',input=dict(path=str(root/'lq')),basic=dict(path=str(root/'tiny')),
                              original=str(root/'original'),frames=5,roi=[3,3,6,6])])
            (bank/'reference.json').write_text(json.dumps(m))
            mods={'tools.compare_720p3x_outputs':types.SimpleNamespace(FrameSource=Source,_resize_rgb=resize),
                  'tools.compose_4way_video':types.SimpleNamespace(_auto_basic_index=mapping)}
            env=dict(CUSTOM_REF_ROOT=str(bank),OUT=str(out),CURRENT='synthetic',KEEP_CUSTOM_PNG='1')
            with mock.patch.dict(sys.modules,mods),mock.patch.dict(os.environ,env),mock.patch.object(sys,'argv',['compose','Wetland1']):
                exec(compile(block,'custom_composition_block','exec'),{})
            aligned=json.loads((folder/'alignment.json').read_text())
            self.assertEqual(aligned['tiny_indices'],[0,2,4,7,9])
            decoded=subprocess.run(['ffmpeg','-v','error','-i',str(folder/'native_fourway.mkv'),
                                    '-f','rawvideo','-pix_fmt','rgb24','-'],check=True,capture_output=True).stdout
            frames=np.frombuffer(decoded,np.uint8).reshape(5,68,12,3)
            for k,f in enumerate(frames):
                for region,value in ((f[28:34,:6],k+1),(f[28:34,6:],20+k),
                                     (f[62:68,:6],40+k),(f[62:68,6:],60+aligned['tiny_indices'][k])):
                    self.assertTrue(np.all(region==value))
            self.assertEqual(len(list(folder.glob('native_frame_*.png'))),5)

    def test_shell_syntax(self):
        subprocess.run(['bash','-n',str(ROOT/'tools/visual_check_m8_task.sh')],check=True)

if __name__=='__main__': unittest.main()
