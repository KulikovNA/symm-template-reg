import unittest
from symm_template_reg.evaluation.two_view_coordinate import two_view_gate
class GateTest(unittest.TestCase):
    def test_good_average_cannot_hide_failed_frame(self):
        def r(frame,p):return {'frame_id':frame,'projected_correspondence_p95_mm':p,'projection_alignment_p95_mm':p,'projection_pose_rotation_error_deg':0,'projection_pose_translation_total_mm':0,'projection_correspondence_rank':3,'surface_membership_p95_mm':0,'exact_global_selected_triangle_in_shortlist_fraction':1,'fallback_fraction':0}
        self.assertFalse(two_view_gate([r(4,.1),r(8,1.1)])['stage_passed'])
