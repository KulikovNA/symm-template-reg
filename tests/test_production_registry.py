import unittest

from symm_template_reg.models import register_all_modules
from symm_template_reg.registry import MODELS


class ProductionRegistryTest(unittest.TestCase):
    def test_only_one_production_model_is_registered(self):
        register_all_modules()
        self.assertEqual(
            sorted(MODELS.module_dict),
            ["CoordinateGuidedSurfaceRegistrationV3"],
        )


if __name__ == "__main__":
    unittest.main()
