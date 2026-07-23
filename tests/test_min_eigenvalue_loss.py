import unittest
import torch
from symm_template_reg.models.losses.joint_surface_correspondence_pose_loss_v3 import covariance_collapse_penalties


class MinEigenvalueTest(unittest.TestCase):
    def test_line_cloud_has_positive_penalty(self):
        line = torch.stack((torch.linspace(-1, 1, 16), torch.zeros(16), torch.zeros(16)), -1)
        self.assertGreater(float(covariance_collapse_penalties(line, torch.randn(16, 3), 1e-4)["min_eigenvalue_penalty"]), .99)

