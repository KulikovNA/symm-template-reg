import unittest

import torch

from symm_template_reg.models.heads.conditioned_base_pose_head import ConditionedBasePoseHead


class CentroidInvariantRotationTest(unittest.TestCase):
    def test_centroid_shift_does_not_change_rotation(self):
        torch.manual_seed(0)
        head = ConditionedBasePoseHead(
            embed_dim=8, hidden_dim=16, split_rotation_translation=True,
            rotation_uses_centroid=False, translation_uses_centroid=True,
        )
        rotation_context = torch.randn(2, 8)
        translation_context = torch.randn(2, 8)
        kwargs = dict(rotation_context=rotation_context, translation_context=translation_context)
        first = head(rotation_context, torch.zeros(2, 3), torch.ones(2), **kwargs)
        second = head(rotation_context, torch.full((2, 3), 10.0), torch.ones(2), **kwargs)
        self.assertTrue(torch.allclose(
            first["base_T_C_from_O"][:, :3, :3],
            second["base_T_C_from_O"][:, :3, :3], atol=1e-7,
        ))


if __name__ == "__main__": unittest.main()
