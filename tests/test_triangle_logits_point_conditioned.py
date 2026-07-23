import unittest
import torch
from symm_template_reg.models.heads.fine_candidate_triangle_head import FineCandidateTriangleHead


class PointConditioningTest(unittest.TestCase):
    def test_swapping_points_changes_logits(self):
        torch.manual_seed(2); h = FineCandidateTriangleHead(embed_dim=4, hidden_dim=8, observed_geometry_dim=2, candidate_geometry_dim=2)
        p = torch.randn(2, 4); c = torch.randn(2, 3, 4); g = torch.randn(2, 2); cg = torch.randn(2, 3, 2)
        self.assertFalse(torch.allclose(h(p, c, g, cg), h(p.flip(0), c, g, cg)))


if __name__ == "__main__": unittest.main()

