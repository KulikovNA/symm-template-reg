import unittest

from symm_template_reg.registry import Registry, build_from_cfg
from symm_template_reg.models import register_all_modules
from symm_template_reg.registry import POSE_MODULES, SYMMETRY_MODULES


class RegistryTest(unittest.TestCase):
    def test_decorator_and_build(self):
        registry = Registry("test")

        @registry.register_module()
        class Add:
            def __init__(self, value, offset=0):
                self.value = value + offset

        config = {"type": "Add", "value": 2}
        instance = build_from_cfg(config, registry, {"offset": 3})
        self.assertEqual(instance.value, 5)
        self.assertEqual(config, {"type": "Add", "value": 2})

    def test_duplicate_is_rejected(self):
        registry = Registry("test")
        registry.register_module(lambda: None, name="same")
        with self.assertRaises(KeyError):
            registry.register_module(lambda: None, name="same")

    def test_pose_and_symmetry_modules_are_config_buildable(self):
        register_all_modules()
        self.assertIn("PoseRepresentation", POSE_MODULES)
        self.assertIn("SymmetryHypothesisExpander", SYMMETRY_MODULES)
        self.assertEqual(
            build_from_cfg(
                {"type": "SymmetryHypothesisExpander", "so2_num_samples": 12},
                SYMMETRY_MODULES,
            ).so2_num_samples,
            12,
        )


if __name__ == "__main__":
    unittest.main()
