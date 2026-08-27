from __future__ import annotations

import unittest

from tools import train_b2b_extreme_decoder_distill_ddp as b2b


class B2BExtremeDecoderDistillTest(unittest.TestCase):
    def test_extreme_variant_registration(self) -> None:
        self.assertEqual(b2b.base.VARIANT_CHANNELS["extreme"], (96, 48, 24, 16))
        self.assertAlmostEqual(
            b2b.base.VARIANT_GMAC["extreme"],
            13.35785472,
            places=8,
        )
        self.assertEqual(
            b2b.base.TRAINER_ID,
            "swiftvr_b2b0c_extreme_decoder_teacher_distill_ddp_v1",
        )

    def test_base_parser_accepts_extreme_variant(self) -> None:
        parser = b2b.base.build_parser()
        variant_action = next(action for action in parser._actions if action.dest == "variant")
        self.assertIn("extreme", tuple(variant_action.choices))


if __name__ == "__main__":
    unittest.main()
