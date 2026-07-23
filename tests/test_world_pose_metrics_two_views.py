import unittest,torch
from symm_template_reg.evaluation.two_view_coordinate import two_view_world_pose_metrics
class WorldMetricsTest(unittest.TestCase):
    def test_identical_world_poses_are_zero(self):
        eye=torch.eye(4).repeat(2,1,1);m=two_view_world_pose_metrics(eye,eye);self.assertAlmostEqual(m['world_translation_difference_mm'],0);self.assertAlmostEqual(m['world_axis_difference_deg'],0)
