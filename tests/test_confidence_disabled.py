import unittest
from symm_template_reg.config import load_config
from symm_template_reg.models import register_all_modules, build_model

class ConfidenceDisabledTest(unittest.TestCase):
    def test_main_config_has_no_confidence_head(self):
        register_all_modules(); c=load_config("configs/debug/joint_correspondence_pose_v2/01_uniform_joint_2views.py"); m=build_model(c["model"])
        self.assertEqual(m.weighting_mode,"uniform"); self.assertIsNone(m.point_weight_head); self.assertFalse(hasattr(m,"pose_head"))
