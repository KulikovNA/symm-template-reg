import unittest
from symm_template_reg.evaluation.active_coordinate import worst_sample_projection_score


class FourViewWorstScoreTest(unittest.TestCase):
    def test_max_not_average(self):
        def value(x): return {"exact_global_projected_correspondence_p95_mm": x, "exact_global_projection_alignment_p95_mm": 0., "exact_global_projection_rotation_error_deg": 0., "exact_global_projection_translation_error_mm": 0.}
        self.assertEqual(worst_sample_projection_score([value(1), value(4)]), 4)


if __name__ == "__main__": unittest.main()

