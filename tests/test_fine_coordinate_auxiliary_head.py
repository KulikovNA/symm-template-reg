import unittest
import torch
from symm_template_reg.models.heads.fine_coordinate_auxiliary_head import FineCanonicalCoordinateAuxiliaryHead


class FineCoordinateAuxTest(unittest.TestCase):
    def test_bounded_output_and_gradient(self):
        x = torch.randn(7, 8, requires_grad=True); q = FineCanonicalCoordinateAuxiliaryHead(8, 16)(x)
        self.assertLessEqual(float(q.detach().abs().max()), 1.0); q.sum().backward(); self.assertGreater(float(x.grad.abs().sum()), 0)


if __name__ == "__main__": unittest.main()

