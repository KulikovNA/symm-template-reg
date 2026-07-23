import unittest
import torch
from symm_template_reg.models.losses.joint_surface_correspondence_pose_loss_v3 import covariance_collapse_penalties


class CovarianceCollapseTest(unittest.TestCase):
    def test_collapsed_cloud_is_penalized(self):
        target = torch.randn(32, 3)
        collapsed = torch.zeros_like(target, requires_grad=True)
        value = covariance_collapse_penalties(collapsed, target, 1e-4)["covariance_error_m2"]
        self.assertGreater(float(value), 0.); value.backward(); self.assertIsNotNone(collapsed.grad)

