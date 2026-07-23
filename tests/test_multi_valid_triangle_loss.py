import unittest
import torch

from symm_template_reg.models.geometry.patch_targets import multi_positive_softmax_loss


class MultiValidTriangleLossTest(unittest.TestCase):
    def test_any_valid_triangle_is_accepted(self):
        valid = torch.tensor([[False, True, True, False]])
        for winner in (1, 2):
            logits = torch.full((1, 4), -20.); logits[0, winner] = 20.
            self.assertLess(float(multi_positive_softmax_loss(logits, valid)), 1e-6)


if __name__ == "__main__": unittest.main()
