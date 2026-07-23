from __future__ import annotations

import unittest

import torch

from symm_template_reg.engine.single_fragment import inverse_sqrt_frequency_weights


class RegionClassBalancingTest(unittest.TestCase):
    def test_present_weights_have_mean_one_and_cap(self):
        weights = inverse_sqrt_frequency_weights([1, 4, 16, 0], max_class_weight=2.0)
        present = weights[:3]
        self.assertAlmostEqual(float(present.mean()), 1.0, places=5)
        self.assertLessEqual(float(present.max()), 2.0)
        self.assertEqual(float(weights[3]), 0.0)
        self.assertGreater(float(weights[0]), float(weights[2]))


if __name__ == "__main__":
    unittest.main()
