import unittest

import torch

from symm_template_reg.models.losses.clean_coordinate_pose_loss_v3 import CleanCoordinatePoseLossV3
from symm_template_reg.models.pose.pose_representation import invert_transform, transform_points
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms
from tests.test_fragment_symmetry_targets import metadata


class SharedSymmetryCleanLossTest(unittest.TestCase):
    def test_one_element_owns_target_pose_and_every_component(self):
        vertices = torch.tensor([[-.02, -.03, -.01], [.03, -.01, .02], [-.01, .04, .03], [.02, .02, -.02]])
        target = vertices.unsqueeze(0)
        bbox_min, bbox_max = vertices.amin(0), vertices.amax(0)
        normalized = (2.0 * (target - bbox_min) / (bbox_max - bbox_min) - 1.0).requires_grad_()
        pose = torch.eye(4).unsqueeze(0)
        result = CleanCoordinatePoseLossV3(current_epoch=250)(
            normalized, target, target, torch.ones((1, 4), dtype=torch.bool), pose,
            [metadata()], [{"type": "C", "order": 2}], [vertices],
        )
        index = int(result["selected_shared_symmetry_element"][0])
        transforms = symmetry_transforms({"type": "C", "order": 2}, metadata().axis.direction, metadata().axis.origin, dtype=vertices.dtype, device=vertices.device)
        expected_target = transform_points(invert_transform(transforms), target)[index]
        self.assertTrue(torch.allclose(result["matched_target_points_O"][0], expected_target))
        self.assertTrue(torch.allclose(result["matched_gt_pose_T_C_from_O"][0], transforms[index]))
        self.assertEqual(index, int(result["loss_by_symmetry_element"][0].argmin()))
        result["loss_total"].backward()
        self.assertIsNotNone(normalized.grad)


if __name__ == "__main__": unittest.main()
