from __future__ import annotations

import unittest

import torch

from swiftvr.training.perceptual_review import (
    _normalize_metric_output,
    flatten_video_frames,
    make_comparison_frame,
    make_difference_frame,
    parse_metric_names,
    parse_student_checkpoint,
)


class PerceptualReviewTests(unittest.TestCase):
    def test_parse_student_checkpoint(self) -> None:
        spec = parse_student_checkpoint(
            "step300=/tmp/checkpoints/step_00000300"
        )
        self.assertEqual(spec.label, "step300")
        self.assertTrue(str(spec.path).endswith("step_00000300"))
        with self.assertRaises(ValueError):
            parse_student_checkpoint("missing-separator")

    def test_flatten_video_frames(self) -> None:
        video = torch.zeros(2, 3, 3, 8, 10)
        frames = flatten_video_frames(video)
        self.assertEqual(tuple(frames.shape), (6, 3, 8, 10))

    def test_metric_names_are_unique_and_validated(self) -> None:
        self.assertEqual(
            parse_metric_names("lpips,dists,lpips,musiq"),
            ("lpips", "dists", "musiq"),
        )
        with self.assertRaises(ValueError):
            parse_metric_names("not-a-metric")

    def test_metric_output_normalization(self) -> None:
        self.assertEqual(
            _normalize_metric_output(torch.tensor([[1.0], [2.0]]), 2),
            [1.0, 2.0],
        )
        with self.assertRaises(ValueError):
            _normalize_metric_output(torch.tensor(1.0), 2)

    def test_visual_panels_have_expected_width(self) -> None:
        frame = torch.zeros(3, 16, 20)
        comparison = make_comparison_frame({"A": frame, "B": frame}, gap=4)
        self.assertEqual(comparison.size, (44, 44))
        difference = make_difference_frame({"A": frame}, frame)
        self.assertEqual(difference.size, (20, 44))


if __name__ == "__main__":
    unittest.main()
