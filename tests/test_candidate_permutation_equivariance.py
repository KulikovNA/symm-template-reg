import unittest
import torch
from symm_template_reg.models.heads.fine_candidate_triangle_head import FineCandidateTriangleHead


class CandidatePermutationTest(unittest.TestCase):
    def test_permutation_permutes_logits(self):
        torch.manual_seed(4); h = FineCandidateTriangleHead(embed_dim=4, hidden_dim=8, observed_geometry_dim=2, candidate_geometry_dim=2)
        p, c, g, cg = torch.randn(2, 4), torch.randn(2, 5, 4), torch.randn(2, 2), torch.randn(2, 5, 2); order = torch.tensor([3, 0, 4, 1, 2])
        self.assertTrue(torch.allclose(h(p, c[:, order], g, cg[:, order]), h(p, c, g, cg)[:, order], atol=1e-6))


if __name__ == "__main__": unittest.main()

