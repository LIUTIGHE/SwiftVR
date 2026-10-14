from __future__ import annotations

import unittest

import torch

from swiftvr.training.reference import (
    VelocityMetricAccumulator,
    cache_sample_key,
    expand_prompt_embedding,
)


class ReferenceMetricTest(unittest.TestCase):
    def test_cache_key_is_stable_and_frame_sensitive(self):
        left = cache_sample_key("plain:vid0", [1, 2, 3])
        self.assertEqual(left, cache_sample_key("plain:vid0", [1, 2, 3]))
        self.assertNotEqual(left, cache_sample_key("plain:vid0", [2, 3, 4]))
        self.assertNotEqual(
            left,
            cache_sample_key("plain:vid0", [1, 2, 3], crop_left=4),
        )

    def test_prompt_embedding_expands_batch(self):
        prompt = torch.arange(12).reshape(3, 4)
        expanded = expand_prompt_embedding(prompt, 2)
        self.assertEqual(tuple(expanded.shape), (2, 3, 4))
        torch.testing.assert_close(expanded[0], prompt)
        torch.testing.assert_close(expanded[1], prompt)

    def test_identical_velocity_metrics(self):
        value = torch.randn(2, 3, 4)
        metric = VelocityMetricAccumulator()
        metric.update(value, value)
        result = metric.compute()
        self.assertEqual(result["velocity_mse"], 0.0)
        self.assertEqual(result["velocity_relative_l2"], 0.0)
        self.assertAlmostEqual(result["velocity_cosine"], 1.0, places=6)

    def test_known_relative_error(self):
        reference = torch.ones(4)
        student = torch.zeros(4)
        metric = VelocityMetricAccumulator()
        metric.update(student, reference)
        result = metric.compute()
        self.assertEqual(result["velocity_mse"], 1.0)
        self.assertEqual(result["velocity_relative_l2"], 1.0)
        self.assertEqual(result["velocity_cosine"], 0.0)


if __name__ == "__main__":
    unittest.main()
