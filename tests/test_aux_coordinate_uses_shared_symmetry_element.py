import unittest, torch
from tools.audit_local_triangle_target_contract import shared_symmetry_target
from tests.test_fragment_symmetry_targets import metadata
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms
from symm_template_reg.models.symmetry.groups import CyclicGroup
from symm_template_reg.models.pose.pose_representation import invert_transform, transform_points


class AuxSharedSymmetryTest(unittest.TestCase):
    def test_one_element_transforms_all_rows(self):
        q = torch.tensor([[.02, 0., .01], [.03, .01, .02]])
        result = shared_symmetry_target(q, metadata(), {"type": "C", "order": 2}, 1)
        transforms = symmetry_transforms(
            CyclicGroup(2), metadata().axis.direction, metadata().axis.origin,
            dtype=q.dtype, device=q.device,
        )
        expected = transform_points(invert_transform(transforms[1:2]), q[None])[0]
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))


if __name__ == "__main__": unittest.main()
