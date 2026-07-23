import unittest
import torch

from symm_template_reg.models.losses.joint_surface_correspondence_pose_loss_v3 import conditional_covariance_penalty


class ConditionalCovarianceLossTest(unittest.TestCase):
    def test_degenerate_rank_activates_penalty(self):
        result=conditional_covariance_penalty(torch.tensor([4.]),torch.tensor([1e-8,2e-6,3e-6]),torch.tensor([[1e-6,2e-6,3e-6]]),min_eigenvalue_m2=1e-6)
        self.assertTrue(bool(result['active'][0])); self.assertEqual(float(result['penalty'][0]),4.)


if __name__ == "__main__": unittest.main()
