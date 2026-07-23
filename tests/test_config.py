import unittest
from pathlib import Path

from symm_template_reg.config import _merge, load_config


class ConfigTest(unittest.TestCase):
    def test_baseline_inherits_runtime_and_keeps_nested_dicts(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "symm_template_reg_baseline.py")
        self.assertEqual(config["dataloader"]["batch_size"], 2)
        self.assertEqual(config["collate"]["type"], "FragmentTemplateCollator")
        self.assertEqual(config["model"]["pose_head"]["num_queries"], 8)
        self.assertEqual(config["dataset"]["max_observed_points"], 4096)

    def test_delete_replaces_inherited_mapping(self):
        result = _merge(
            {"loss": {"type": "Old", "weight": 1.0}},
            {"loss": {"_delete_": True, "weight": 0.2}},
        )
        self.assertEqual(result, {"loss": {"weight": 0.2}})


if __name__ == "__main__":
    unittest.main()
