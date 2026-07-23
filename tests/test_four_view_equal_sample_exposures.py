import unittest
from symm_template_reg.config import load_config
from symm_template_reg.engine.training_budget import resolve_training_budget


class FourViewExposureTest(unittest.TestCase):
    def test_every_frame_gets_2500_exposures(self):
        config = load_config("configs/debug/coordinate_guided_surface_v2/views04.py")
        budget = resolve_training_budget(config["train_budget"], selected_samples=4, batch_size=4, gradient_accumulation_steps=1, drop_last=False, configured_max_optimizer_steps=None, configured_max_epochs=2500)
        self.assertEqual(budget.target_sample_exposures, 2500)
        self.assertEqual(budget.computed_max_optimizer_steps, 2500)


if __name__ == "__main__": unittest.main()

