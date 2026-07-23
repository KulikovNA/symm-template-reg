import unittest

from symm_template_reg.evaluation.active_coordinate import worst_sample_practical_score


class WorstPracticalScoreTest(unittest.TestCase):
    def test_worst_not_mean_selects_checkpoint(self):
        def item(value):
            return {
                "exact_global_projected_correspondence_p95_mm": value,
                "exact_global_projection_alignment_p95_mm": 0.0,
                "exact_global_projection_rotation_error_deg": 0.0,
                "exact_global_projection_translation_error_mm": 0.0,
            }
        self.assertAlmostEqual(worst_sample_practical_score([item(0.3), item(1.5)]), 1.0)


if __name__ == "__main__":
    unittest.main()
