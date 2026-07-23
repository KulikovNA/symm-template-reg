import unittest
import torch
from symm_template_reg.models.heads.fine_candidate_triangle_head import FineCandidateTriangleHead


class CandidateConditioningTest(unittest.TestCase):
    def test_zero_candidate_geometry_changes_logits(self):
        torch.manual_seed(3); h = FineCandidateTriangleHead(embed_dim=4, hidden_dim=8, observed_geometry_dim=2, candidate_geometry_dim=2)
        p, c, g, cg = torch.randn(2, 4), torch.randn(2, 3, 4), torch.randn(2, 2), torch.randn(2, 3, 2)
        self.assertFalse(torch.allclose(h(p, c, g, cg), h(p, c, g, torch.zeros_like(cg))))


if __name__ == "__main__": unittest.main()

