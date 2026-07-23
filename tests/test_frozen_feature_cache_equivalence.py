import unittest

import torch

from symm_template_reg.models.geometry.fine_local_correspondence_features import FineLocalCorrespondenceFeatureAdapter
from symm_template_reg.models.heads.fine_coordinate_auxiliary_head import FineCanonicalCoordinateAuxiliaryHead
from symm_template_reg.engine.frozen_feature_cache import cached_fine_coordinate_forward


class FrozenFeatureCacheEquivalenceTest(unittest.TestCase):
    def test_same_inputs_have_same_output_and_gradient(self):
        torch.manual_seed(0); adapter = FineLocalCorrespondenceFeatureAdapter(embed_dim=8); head = FineCanonicalCoordinateAuxiliaryHead(embed_dim=8, hidden_dim=8)
        payload = {
            "dense_observed_features": torch.randn(1, 40, 8),
            "template_conditioned_observed_features": torch.randn(1, 40, 8),
            "observed_points_C": torch.randn(1, 40, 3) * .01,
            "observed_valid_mask": torch.ones(1, 40, dtype=torch.bool),
        }
        _, first = cached_fine_coordinate_forward(adapter, head, payload)
        _, second = cached_fine_coordinate_forward(adapter, head, payload)
        self.assertLessEqual(float((first - second).abs().max()), 1e-6)


if __name__ == "__main__": unittest.main()

