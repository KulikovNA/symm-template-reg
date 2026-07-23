import unittest
import torch
from symm_template_reg.models.pose import WeightedProcrustes


class RankInvalidPoseTest(unittest.TestCase):
    def test_planar_correspondence_is_invalid_under_v4_contract(self):
        q = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]])
        result = WeightedProcrustes().solve(q, q, torch.ones((1, 4)), torch.ones((1, 4), dtype=torch.bool))
        self.assertEqual(int(result["rank"][0]), 2)
        self.assertFalse(bool(result["rank_valid"][0]))

