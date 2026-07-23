import unittest
from symm_template_reg.config import load_config


class BarycentricDeprecationTest(unittest.TestCase):
    def test_fallback_uses_analytic_projection(self):
        cfg=load_config('configs/debug/fine_correspondence_v1/02_coordinate_guided_triangle_frame04.py'); head=cfg['model']['correspondence_head']
        self.assertTrue(head['analytic_barycentric_projection']); self.assertEqual(head['learned_barycentric_status'],'failed_frozen_feature_barycentric_capacity_on_frame04')
        self.assertEqual(cfg['loss']['joint_surface_correspondence_pose_v3']['lambda_barycentric'],0)


if __name__ == "__main__": unittest.main()

