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

    def test_wrapper_parser_accepts_extreme_without_duplicate_options(self) -> None:
        parser = b2b.build_parser()
        variant_action = next(action for action in parser._actions if action.dest == "variant")
        self.assertIn("extreme", tuple(variant_action.choices))
        self.assertIn("--teacher-l2-weight", parser._option_string_actions)
        self.assertIn("--teacher-lpips-weight", parser._option_string_actions)
        self.assertIn("--teacher-temporal-weight", parser._option_string_actions)

        # argparse itself rejects duplicate option strings, so reaching this
        # point already verifies the original conflict is gone. Check the
        # intended B2B/B1-compatible defaults as an additional regression guard.
        self.assertEqual(
            parser._option_string_actions["--teacher-l2-weight"].default,
            10.0,
        )
        self.assertEqual(
            parser._option_string_actions["--teacher-lpips-weight"].default,
            0.1,
        )
        self.assertEqual(
            parser._option_string_actions["--teacher-temporal-weight"].default,
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
