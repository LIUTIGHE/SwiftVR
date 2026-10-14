"""CPU/META contract tests for 530-GFLOP M11 Transformer architecture gate.

No GPU, image data, teacher weights or val13 are needed.
"""
import unittest

from tools.audit_b2b_moe_architecture import (
    BASELINE_M8_TRANSFORMER_GMAC,
    FIXED_REAE_ENCODER_GMAC,
    FIXED_A1_DECODER_GMAC,
    M8_MOE_SPEC, M11A_MOE_SPEC, M11B_MOE_SPEC,
    calibrated_budget_compute, canonical_compute, make_config,
)
from swiftvr.training.b2b_moe import expected_moe_shape


class TestM11ArchitectureBudget(unittest.TestCase):
    def test_m8_anchor_reproduced(self):
        row=calibrated_budget_compute(expected_moe_shape(M8_MOE_SPEC))
        self.assertAlmostEqual(row["calibrated_transformer_gmac_per_frame"],
                               BASELINE_M8_TRANSFORMER_GMAC,places=8)
        self.assertAlmostEqual(row["calibrated_total_gmac_per_frame"],
                               FIXED_REAE_ENCODER_GMAC+
                               BASELINE_M8_TRANSFORMER_GMAC+
                               FIXED_A1_DECODER_GMAC,places=8)

    def test_a_depth_heavy(self):
        spec=M11A_MOE_SPEC
        self.assertEqual((spec.hidden_dim,spec.num_layers,spec.active_ffn_dim),
                         (1024,22,1536))
        result=calibrated_budget_compute(expected_moe_shape(spec))
        self.assertAlmostEqual(result["calibrated_total_gflops_per_frame_2flop_per_mac"],
                               527.2066702933334,places=3)
        self.assertLess(result["calibrated_total_gflops_per_frame_2flop_per_mac"],530)

    def test_b_ffn_heavy(self):
        spec=M11B_MOE_SPEC
        self.assertEqual((spec.hidden_dim,spec.num_layers,spec.normal_expert_dim,
                          spec.active_ffn_dim,spec.total_ffn_dim),
                         (1024,20,576,2176,7936))
        self.assertLessEqual(spec.total_ffn_dim,8960)
        result=calibrated_budget_compute(expected_moe_shape(spec))
        self.assertAlmostEqual(result["calibrated_total_gflops_per_frame_2flop_per_mac"],
                               527.513488,places=3)
        self.assertLess(result["calibrated_total_gflops_per_frame_2flop_per_mac"],530)
        baseline=canonical_compute(expected_moe_shape(M8_MOE_SPEC))
        bigger=canonical_compute(expected_moe_shape(spec))
        # Only activated FFN linears change in B; QKV/attention/router stay fixed.
        self.assertAlmostEqual(baseline["attention_qkv_out_projection_gmac_per_frame"],
                               bigger["attention_qkv_out_projection_gmac_per_frame"],places=9)
        self.assertAlmostEqual(baseline["router_linear_gmac_per_frame"],
                               bigger["router_linear_gmac_per_frame"],places=9)
        self.assertAlmostEqual(bigger["active_ffn_linear_gmac_per_frame"]-
                               baseline["active_ffn_linear_gmac_per_frame"],
                               13.369344,places=6)

    def test_spec_export_only_changes_declared_fields(self):
        m8={
            "_class_name":"WanTransformer3DModelPromptFreeNoTimeMoE",
            "num_layers":20,"num_attention_heads":8,"attention_head_dim":128,
            "ffn_dim":1536,"shared_expert_dim":1024,"normal_expert_dim":256,
            "num_experts":12,"top_k":2,
            "patch_size":[1,2,2], "self_attn_window_hw":[16,16]
        }
        a=make_config(m8,expected_moe_shape(M11A_MOE_SPEC))
        b=make_config(m8,expected_moe_shape(M11B_MOE_SPEC))
        self.assertEqual({k:v for k,v in a.items() if v!=m8.get(k)},
                         {"num_layers":22})
        self.assertEqual({k:v for k,v in b.items() if v!=m8.get(k)},
                         {"normal_expert_dim":576,"ffn_dim":2176})
        m8["num_layers"]=21
        with self.assertRaises(ValueError):
            make_config(m8,expected_moe_shape(M11A_MOE_SPEC))

    def test_initializers_and_trainer_specs_agree(self):
        from tools.build_b2b_moe_depth_init import DEPTH_TARGETS, M11A_ARCHITECTURE
        from tools.build_b2b_moe_init import (
            INIT_ARCHITECTURES, M11B_ARCHITECTURE, M11B_SPEC as INITIALIZER_B,
        )
        from tools.train_b2b_moe_ta_distill_ddp_cache_safe import (
            _spec_for_architecture, SUPPORTED_ARCHITECTURES,
        )
        self.assertEqual(DEPTH_TARGETS[M11A_ARCHITECTURE],M11A_MOE_SPEC)
        self.assertEqual(INITIALIZER_B,M11B_MOE_SPEC)
        for name,expected in (
            (M11A_ARCHITECTURE,M11A_MOE_SPEC),
            (M11B_ARCHITECTURE,M11B_MOE_SPEC),
        ):
            self.assertIn(name,SUPPORTED_ARCHITECTURES)
            self.assertEqual(_spec_for_architecture(name),expected)
        self.assertIn(M11B_ARCHITECTURE,INIT_ARCHITECTURES)

    def test_initializer_source_contract(self):
        # The mature M5 source is D1024/L30, and B's TA source is D1536/L30,
        # dense FFN8960. Both initializers must be able to fill their targets.
        self.assertLess(M11A_MOE_SPEC.num_layers,30)
        self.assertEqual(M11A_MOE_SPEC.hidden_dim,1024)
        self.assertEqual(M11B_MOE_SPEC.num_layers,20)
        self.assertLessEqual(M11B_MOE_SPEC.total_ffn_dim,8960)


if __name__=="__main__":
    unittest.main()
