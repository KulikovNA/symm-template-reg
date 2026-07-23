import unittest
import torch

from symm_template_reg.models.geometry.patch_targets import multi_positive_softmax_loss


class OracleTriangleLogitsTest(unittest.TestCase):
    def test_oracle_small_and_wrong_large(self):
        valid = torch.tensor([[False, True, True, False]])
        oracle = torch.tensor([[-20.,20.,20.,-20.]])
        wrong = torch.tensor([[-20.,-20.,-20.,20.]])
        self.assertLess(float(multi_positive_softmax_loss(oracle, valid)), 1e-6)
        self.assertGreater(float(multi_positive_softmax_loss(wrong, valid)), 20.)


if __name__ == "__main__": unittest.main()
