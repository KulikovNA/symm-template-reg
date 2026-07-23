import unittest
import torch

from symm_template_reg.engine.frozen_feature_cache import cache_eligibility, FINE_ONLY_PREFIXES


class CacheUpstreamTrainableTest(unittest.TestCase):
    def test_upstream_trainable_disables_cache(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2)); model.eval()
        result = cache_eligibility(model, trainable_prefixes=FINE_ONLY_PREFIXES, augmentations_enabled=False, deterministic_point_sampling=True)
        self.assertFalse(result["cache_allowed_by_policy"])


if __name__ == "__main__": unittest.main()

