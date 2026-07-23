import unittest

from symm_template_reg.evaluation.active_coordinate import strict_and_practical_stage_gates


def row(frame, correspondence=1.25):
    return {
        "frame_id": frame, "sample_id": str(frame),
        "exact_global_projected_correspondence_p95_mm": correspondence,
        "exact_global_projection_alignment_p95_mm": 1.2,
        "exact_global_projection_rotation_error_deg": 0.2,
        "exact_global_projection_translation_error_mm": 0.08,
        "exact_global_projection_rank": 3,
        "exact_global_surface_membership_p95_mm": 0.0,
        "k16_exact_global_triangle_recall": 1.0, "k16_fallback_fraction": 0.0,
    }


class SeparateGatesTest(unittest.TestCase):
    def test_practical_pass_does_not_rewrite_strict_failure(self):
        frames = (4, 5, 2, 8, 0, 1, 6, 9)
        result = strict_and_practical_stage_gates([row(frame) for frame in frames], frames)
        self.assertFalse(result["strict_submillimetre_gate"]["stage_passed"])
        self.assertTrue(result["practical_pose_first_gate"]["stage_passed"])
        self.assertTrue(result["next_stage_allowed"])


if __name__ == "__main__":
    unittest.main()
