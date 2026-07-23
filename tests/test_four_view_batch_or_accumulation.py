import unittest
from symm_template_reg.config import load_config


class FourViewBatchTest(unittest.TestCase):
    def test_primary_config_uses_all_views_per_update(self):
        config = load_config("configs/debug/coordinate_guided_surface_v2/views04.py")
        self.assertEqual(config["data"]["train_batch_size"] * config["train"]["gradient_accumulation_steps"], 4)


if __name__ == "__main__": unittest.main()

