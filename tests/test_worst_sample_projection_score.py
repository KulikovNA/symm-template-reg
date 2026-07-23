import unittest
from symm_template_reg.evaluation.two_view_coordinate import worst_sample_projection_score
class WorstScoreTest(unittest.TestCase):
    def test_max_not_mean(self):
        def r(x):return {'projected_correspondence_p95_mm':x,'projection_alignment_p95_mm':0,'projection_pose_rotation_error_deg':0,'projection_pose_translation_total_mm':0}
        self.assertEqual(worst_sample_projection_score([r(1),r(4)]),4)
