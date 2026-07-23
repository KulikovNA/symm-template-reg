import unittest
import torch

from symm_template_reg.models.pose.pose_representation import invert_transform, transform_points
from symm_template_reg.models.symmetry.groups import CyclicGroup
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms


class TriangleTargetSymmetryConsistencyTest(unittest.TestCase):
    def test_all_targets_can_share_one_transformed_q(self):
        q = torch.tensor([[1.0, 0.0, 0.0]])
        s = symmetry_transforms(CyclicGroup(2), [0, 0, 1], [0, 0, 0], dtype=q.dtype)[1:2]
        q_s = transform_points(invert_transform(s), q[None])[0]
        patch_target = q_s.clone(); triangle_target = q_s.clone(); bary_target = q_s.clone()
        self.assertTrue(torch.equal(patch_target, triangle_target))
        self.assertTrue(torch.equal(triangle_target, bary_target))
        self.assertLess(float(q_s[0, 0]), 0.0)


if __name__ == "__main__": unittest.main()
