from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_m8_decoder76_warmup_ddp as m8b


def test_m8_warmup_parser_has_no_duplicate_teacher_l2_option():
    parser = m8b._build_m8_parser()
    teacher_l2_actions = [
        action
        for action in parser._actions
        if "--teacher-l2-weight" in action.option_strings
    ]
    assert len(teacher_l2_actions) == 1
    assert teacher_l2_actions[0].default == 10.0


def test_m8_warmup_parser_exposes_m8_specific_options():
    parser = m8b._build_m8_parser()
    destinations = {action.dest for action in parser._actions}
    assert "teacher_lpips_weight" in destinations
    assert "teacher_temporal_weight" in destinations
    assert "visual_validation_samples" in destinations
    assert m8b.M8_VARIANT in m8b.trainer.VARIANT_CHANNELS
