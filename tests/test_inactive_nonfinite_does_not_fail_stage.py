import math
import unittest

from symm_template_reg.evaluation.active_coordinate import active_sample_gate


class InactiveNonfiniteTest(unittest.TestCase):
    def test_legacy_infinity_is_ignored(self):
        row = {
            "exact_global_projected_correspondence_p95_mm": .2,
            "exact_global_projection_alignment_p95_mm": .2,
            "exact_global_projection_rotation_error_deg": .2,
            "exact_global_projection_translation_error_mm": .2,
            "exact_global_projection_rank": 3,
            "exact_global_surface_membership_p95_mm": 0.0,
            "k16_exact_global_triangle_recall": 1.0,
            "k16_fallback_fraction": 0.0,
            "active_nonfinite_detected": False,
            "legacy_local_triangle_set_ce": math.inf,
        }
        self.assertTrue(active_sample_gate(row)["passed"])


if __name__ == "__main__": unittest.main()

