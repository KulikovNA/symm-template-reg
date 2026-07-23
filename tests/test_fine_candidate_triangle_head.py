import unittest
import torch
from symm_template_reg.models.heads.fine_candidate_triangle_head import FineCandidateTriangleHead


class FineCandidateHeadTest(unittest.TestCase):
    def test_output_shape_and_gradient(self):
        head = FineCandidateTriangleHead(embed_dim=8, hidden_dim=16, observed_geometry_dim=3, candidate_geometry_dim=4)
        p = torch.randn(5, 8, requires_grad=True); c = torch.randn(5, 7, 8)
        logits = head(p, c, torch.randn(5, 3), torch.randn(5, 7, 4)); self.assertEqual(logits.shape, (5, 7))
        logits.sum().backward(); self.assertIsNotNone(p.grad)


if __name__ == "__main__": unittest.main()

