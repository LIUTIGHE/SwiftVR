"""P2 1K->50K continuation invariants, no model/GPU needed."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_continuation_test_target", ROOT/"tools/train_m8_p2_gan_continue.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ContinueScheduleTest(unittest.TestCase):
    def test_continuous_base_lr_after_original_1k(self):
        self.assertAlmostEqual(MODULE.continuation_lr_scale(1000,1000,200,50000),0.1)
        self.assertGreater(MODULE.continuation_lr_scale(1001,1000,200,50000),0.1)
        self.assertLess(MODULE.continuation_lr_scale(1001,1000,200,50000),0.11)
        self.assertAlmostEqual(MODULE.continuation_lr_scale(1200,1000,200,50000),1.0)
        self.assertAlmostEqual(MODULE.continuation_lr_scale(50000,1000,200,50000),0.1)

    def test_no_undefined_schedule(self):
        for inputs in ((900,1000,200,50000),(1001,1000,60000,50000),
                       (1001,50000,0,50000)):
            with self.subTest(inputs=inputs):
                with self.assertRaises(ValueError):
                    MODULE.continuation_lr_scale(*inputs)

    def test_source_config_rejects_changed_training_semantics(self):
        from pathlib import Path
        cfg=dict(kind="m8_p2_video_patchgan_v1",batch_size=32,gradient_accumulation_steps=1,
                 world_size=4,gan_weight=0.01, hashes={"/repo/data.json":"same","/repo/tools/train_m8_p2_video_gan.py":"old"})
        new=dict(cfg,hashes={"/repo/data.json":"same","/repo/tools/train_m8_p2_gan_continue.py":"new"})
        self.assertTrue(MODULE.verify_old_p2(cfg,new,original_trainer="/repo/tools/train_m8_p2_video_gan.py"))
        for field,value in (("batch_size",1),("gan_weight",0.1),("world_size",1)):
            changed={**new,field:value}
            with self.subTest(field=field),self.assertRaises(ValueError):
                MODULE.verify_old_p2(cfg,changed,original_trainer="/repo/tools/train_m8_p2_video_gan.py")
        altered={**new,"hashes":{"/repo/data.json":"tampered"}}
        with self.assertRaises(ValueError):
            MODULE.verify_old_p2(cfg,altered,original_trainer="/repo/tools/train_m8_p2_video_gan.py")

    def test_script_help_and_shell_syntax(self):
        import os
        subprocess.run(["bash","-n",str(ROOT/"tools/run_m8_p2_gan_50k.sh")],check=True)
        r=subprocess.run([sys.executable,str(ROOT/"tools/train_m8_p2_gan_continue.py"),"--help"],
                         capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn("--continue-from",r.stdout)

    def test_source_stays_untouched_and_new_run_required(self):
        args=MODULE.build_parser().parse_args([
          "--continue-from","/tmp/old/checkpoints/step_00001000",
          "--base-checkpoint","E","--student-init","P1",
          "--decoder-checkpoint","A1","--legacy-view-cache","legacy",
          "--val-view-cache","val","--ultra-pairs","paired",
          "--output-dir","/tmp/new"])
        self.assertEqual(args.max_steps,50000)
        self.assertEqual(args.schedule_steps,50000)
        self.assertEqual(args.warmup_steps,200)
        self.assertEqual(args.validate_every,2500)
        self.assertEqual(args.save_every,1000)
        self.assertIsNone(args.resume)

if __name__=="__main__":
    unittest.main()
