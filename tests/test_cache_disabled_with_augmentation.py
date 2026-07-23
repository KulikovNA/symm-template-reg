import unittest
from symm_template_reg.engine.static_geometry_cache import validate_static_cache_configuration

class CacheAugmentationTest(unittest.TestCase):
    def test_enabled_augmentation_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"augmentations"):
            validate_static_cache_configuration(enabled=True,augmentations={"enabled":True})

