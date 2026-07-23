import unittest
import torch

from symm_template_reg.models.geometry.patch_targets import multi_positive_softmax_loss


class TriangleCandidatePermutationTest(unittest.TestCase):
    def test_joint_permutation_keeps_loss(self):
        logits = torch.tensor([[1.,3.,-2.,4.]]); valid = torch.tensor([[False,True,True,False]])
        order = torch.tensor([2,0,3,1])
        self.assertTrue(torch.allclose(multi_positive_softmax_loss(logits, valid), multi_positive_softmax_loss(logits[:,order], valid[:,order])))


if __name__ == "__main__": unittest.main()
