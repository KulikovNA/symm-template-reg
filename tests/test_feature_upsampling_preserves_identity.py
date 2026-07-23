import unittest
import torch
from symm_template_reg.models.geometry.point_ops import nearest_interpolate


class UpsamplingIdentityTest(unittest.TestCase):
    def test_nearest_token_mapping_is_point_conditioned(self):
        dense = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
        sampled = dense.clone(); features = torch.tensor([[[1., 0.], [0., 1.]]])
        result = nearest_interpolate(dense, sampled, features, torch.ones((1, 2), dtype=torch.bool))
        self.assertFalse(torch.equal(result[0, 0], result[0, 1]))

