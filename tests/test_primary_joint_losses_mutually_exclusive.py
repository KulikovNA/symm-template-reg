import unittest
from symm_template_reg.config import validate_primary_joint_losses


class PrimaryLossConfigTest(unittest.TestCase):
    def test_two_primary_losses_rejected(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_primary_joint_losses({"loss": {"joint_correspondence_pose": {"enabled": True}, "joint_surface_correspondence_pose_v3": {"enabled": True}}})

