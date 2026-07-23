import unittest
from pathlib import Path

from symm_template_reg.config import load_config


class TenViewsPerStepTest(unittest.TestCase):
    def test_config_accounts_for_exactly_ten_views(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/debug/coordinate_guided_surface_v3/views10_scratch_full.py")
        data, train = config["data"], config["train"]
        self.assertEqual(data["train_batch_size"] * train["gradient_accumulation_steps"], 10)
        self.assertEqual(data["effective_views_per_optimizer_step"], 10)
        self.assertFalse(config["frozen_feature_cache"]["enabled"])


if __name__ == "__main__": unittest.main()
