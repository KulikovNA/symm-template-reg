import unittest

import torch

from symm_template_reg.models.attention import FocusAttention, RotationInvariantAttention
from symm_template_reg.models.geometry import PointPairFeatures
from symm_template_reg.models.losses import PoseSetLoss


class OptionalModulesTest(unittest.TestCase):
    def test_ppf_parallel_and_antiparallel_is_finite(self):
        source = torch.zeros(1, 2, 3)
        target = torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0]]])
        normals = torch.tensor([[[1.0, 0, 0], [1.0, 0, 0]]])
        result = PointPairFeatures()(source, target, normals, -normals)
        self.assertEqual(tuple(result.shape), (1, 2, 4))
        self.assertTrue(torch.isfinite(result).all())

    def test_rotation_invariant_attention_and_focus_masks(self):
        query = torch.randn(2, 4, 16)
        key = torch.randn(2, 5, 16)
        key_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        pair_features = torch.randn(2, 4, 5, 4)
        output = RotationInvariantAttention(16, 4)(query, key, key_mask, pair_features)
        self.assertEqual(tuple(output.shape), (2, 4, 16))
        self.assertTrue(torch.isfinite(output).all())
        focused, indices = FocusAttention(16, keep_ratio=0.8)(key, key_mask)
        self.assertEqual(focused.shape[1], 3)
        self.assertTrue(key_mask.gather(1, indices).all())

    def test_empty_pose_target_is_no_pose_only(self):
        predicted = torch.eye(4).repeat(1, 3, 1, 1).requires_grad_()
        logits = torch.randn(1, 3, requires_grad=True)
        target = [torch.empty(0, 4, 4)]
        result = PoseSetLoss()(predicted, logits, target)
        result["loss_pose_set"].backward()
        self.assertTrue(torch.isfinite(result["loss_pose_set"]))
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
