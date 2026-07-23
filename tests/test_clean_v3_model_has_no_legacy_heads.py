import unittest

from symm_template_reg.models import build_model
from symm_template_reg.models.detectors.coordinate_guided_surface_registration_v3 import LEGACY_MODULE_TOKENS
from tests.clean_v3_test_utils import tiny_clean_v3_config


class CleanV3ModelTest(unittest.TestCase):
    def test_clean_model_has_no_legacy_heads(self):
        model = build_model(tiny_clean_v3_config())
        names = [name.lower() for name, _ in model.named_modules()]
        self.assertFalse([name for name in names if any(token in name for token in LEGACY_MODULE_TOKENS)])
        self.assertFalse(hasattr(model, "correspondence_head"))
        self.assertFalse(hasattr(model, "pose_head"))

if __name__ == "__main__": unittest.main()
