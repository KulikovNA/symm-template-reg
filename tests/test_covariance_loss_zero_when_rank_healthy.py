import unittest
import torch

from symm_template_reg.models.losses.joint_surface_correspondence_pose_loss_v3 import conditional_covariance_penalty


class HealthyCovarianceLossTest(unittest.TestCase):
    def test_healthy_rank_masks_covariance_error(self):
        result=conditional_covariance_penalty(torch.tensor([4.]),torch.tensor([2e-6,3e-6,4e-6]),torch.tensor([[2e-6,3e-6,4e-6]]),min_eigenvalue_m2=1e-6)
        self.assertFalse(bool(result['active'][0])); self.assertEqual(float(result['penalty'][0]),0.)


if __name__ == "__main__": unittest.main()
