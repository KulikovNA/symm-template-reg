import unittest

import torch

from symm_template_reg.models.losses.cross_view_world_pose_loss import CrossViewWorldPoseLoss
from tests.test_fragment_symmetry_targets import metadata


class CrossViewWorldPoseLossTest(unittest.TestCase):
    def test_correct_camera_poses_zero_and_constant_camera_pose_penalized(self):
        T_W_from_C = torch.eye(4).repeat(2, 1, 1)
        T_W_from_C[1, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        correct = torch.linalg.inv(T_W_from_C)
        module = CrossViewWorldPoseLoss(reference_mode="all_pairs")
        args = ([metadata(), metadata()], [{"type": "C", "order": 1}] * 2)
        zero = module(correct, T_W_from_C, *args)["cross_view_world_pose_loss"]
        bad = module(torch.eye(4).repeat(2, 1, 1), T_W_from_C, *args)["cross_view_world_pose_loss"]
        self.assertLess(float(zero), 1e-6)
        self.assertGreater(float(bad), 0.1)


if __name__ == "__main__": unittest.main()
