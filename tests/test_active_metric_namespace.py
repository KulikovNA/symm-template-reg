import unittest

from tools.recheck_coordinate_guided_surface import _aggregate_active_namespaces


class ActiveMetricNamespaceTest(unittest.TestCase):
    def test_only_explicit_active_and_inactive_namespaces(self):
        row = {
            "exact_global_projection_score": 1.0,
            "exact_global_sample_gate_passed": True,
            "exact_global_projected_correspondence_p95_mm": .2,
            "exact_global_projection_alignment_p95_mm": .3,
            "exact_global_projection_rotation_error_deg": .1,
            "exact_global_projection_translation_error_mm": .1,
            "k16_exact_global_triangle_recall": 1.0,
            "k16_fallback_fraction": 0.0,
            "k16_projected_correspondence_p95_mm": .2,
        }
        values = _aggregate_active_namespaces([row], {})
        self.assertIn("eval/active/exact_global", values)
        self.assertFalse(values["eval/inactive/legacy_triangle"]["active"])


if __name__ == "__main__": unittest.main()

