import unittest

import torch

from symm_template_reg.models import build_model
from symm_template_reg.models.losses import CorrespondenceLoss, OverlapLoss, PoseSetLoss
from tests.test_model_forward import tiny_model_config, variable_batch


class LossBackwardTest(unittest.TestCase):
    def test_finite_backward(self):
        torch.manual_seed(7)
        model = build_model(tiny_model_config()).train()
        batch = variable_batch()
        output = model(batch)
        target_pose = torch.eye(4).repeat(2, 1, 1)
        target_pose[:, :3, 3] = torch.tensor([[0.01, -0.02, 0.4], [-0.03, 0.01, 0.5]])
        pose_losses = PoseSetLoss()(output.pose_hypotheses, output.pose_logits, target_pose)
        target_corr = torch.zeros_like(output.correspondence_points_O)
        correspondence = CorrespondenceLoss()(
            output.correspondence_points_O, target_corr, output.observed_valid_mask
        )
        overlap = OverlapLoss()(
            output.observed_overlap_logits,
            output.observed_valid_mask,
            output.observed_valid_mask,
        )
        loss = pose_losses["loss_pose_set"] + correspondence + overlap
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()

