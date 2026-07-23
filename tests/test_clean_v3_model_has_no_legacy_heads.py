import unittest

from symm_template_reg.models import build_model
from symm_template_reg.models.detectors.coordinate_guided_surface_registration_v3 import LEGACY_MODULE_TOKENS
from tests.clean_v3_test_utils import tiny_clean_v3_config
from symm_template_reg.engine.overfit_trainer import clean_active_metric_row


class CleanV3ModelTest(unittest.TestCase):
    def test_clean_model_has_no_legacy_heads(self):
        model = build_model(tiny_clean_v3_config())
        names = [name.lower() for name, _ in model.named_modules()]
        self.assertFalse([name for name in names if any(token in name for token in LEGACY_MODULE_TOKENS)])
        self.assertFalse(hasattr(model, "correspondence_head"))
        self.assertFalse(hasattr(model, "pose_head"))

    def test_inactive_metric_fields_are_absent(self):
        filtered = clean_active_metric_row({
            "sample_id": "x", "frame_id": 0,
            "exact_global_projection_rank": 3,
            "k16_exact_global_triangle_recall": 1.0,
            "query_pose_costs": [1.0], "ranking_regret": 0.0,
            "patch_confusion_matrix": [[1]], "barycentric_error": 0.0,
            "active_region_accuracy": 1.0,
        })
        self.assertEqual(
            set(filtered),
            {"sample_id", "frame_id", "exact_global_projection_rank", "k16_exact_global_triangle_recall"},
        )


if __name__ == "__main__": unittest.main()
