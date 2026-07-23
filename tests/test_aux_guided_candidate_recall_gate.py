import unittest
from tools.recheck_coordinate_guided_surface import _passes

class CandidateGateTest(unittest.TestCase):
    def test_recall_below_threshold_fails(self):
        m={'exact_global_selected_triangle_in_shortlist_fraction':.99,'projected_correspondence_p95_mm':.5,'projection_alignment_p95_mm':.5,'projection_pose_rotation_error_deg':.1,'projection_pose_translation_total_mm':.1,'projection_correspondence_rank':3,'surface_membership_p95_mm':0.,'nonfinite_detected':False}
        self.assertFalse(_passes(m)['passed'])
