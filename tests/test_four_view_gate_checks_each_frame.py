import unittest
from symm_template_reg.evaluation.active_coordinate import four_view_stage_gate


def row(frame, correspondence=.2):
    return {"frame_id": frame, "exact_global_projected_correspondence_p95_mm": correspondence, "exact_global_projection_alignment_p95_mm": .2, "exact_global_projection_rotation_error_deg": .2, "exact_global_projection_translation_error_mm": .2, "exact_global_projection_rank": 3, "exact_global_surface_membership_p95_mm": 0., "k16_exact_global_triangle_recall": 1., "k16_fallback_fraction": 0., "active_nonfinite_detected": False}


class FourViewGateTest(unittest.TestCase):
    def test_one_bad_frame_blocks_stage(self):
        rows = [row(4), row(5), row(2, 1.1), row(8)]
        self.assertFalse(four_view_stage_gate(rows)["stage_passed"])


if __name__ == "__main__": unittest.main()

