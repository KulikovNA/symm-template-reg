import unittest

import torch

from symm_template_reg.models.losses.pairwise_pose_response_loss import PairwisePoseResponseLoss


class PairwiseResponseLossTest(unittest.TestCase):
    def test_perfect_zero_and_constant_response_penalized(self):
        gt = torch.eye(4).repeat(2, 1, 1)
        gt[1, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        module = PairwisePoseResponseLoss()
        self.assertLess(float(module(gt, gt)["pairwise_pose_response_loss"]), 1e-8)
        constant = torch.eye(4).repeat(2, 1, 1)
        self.assertGreater(float(module(constant, gt)["pairwise_rotation_response_loss"]), 0.1)


if __name__ == "__main__": unittest.main()
