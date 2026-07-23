import unittest
import torch


class PerPointFeatureVarianceTest(unittest.TestCase):
    def test_distinct_features_have_nonzero_variance(self):
        features = torch.arange(48, dtype=torch.float32).reshape(6, 8)
        self.assertGreater(float(features.var(0, unbiased=False).mean()), 0.)
        self.assertEqual(torch.unique(features, dim=0).shape[0], 6)

