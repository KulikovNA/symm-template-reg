import unittest
from pathlib import Path

from symm_template_reg.config import load_config


class EightViewsPerOptimizerStepTest(unittest.TestCase):
    def test_preferred_mode_covers_all_views_once(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs/debug/coordinate_guided_surface_v2/views08.py")
        batch = cfg["data"]["train_batch_size"]
        accumulation = cfg["train"]["gradient_accumulation_steps"]
        self.assertEqual((batch, accumulation), (8, 1))
        self.assertEqual(batch * accumulation, 8)
        self.assertEqual(cfg["data"]["effective_views_per_optimizer_step"], 8)


if __name__ == "__main__":
    unittest.main()
