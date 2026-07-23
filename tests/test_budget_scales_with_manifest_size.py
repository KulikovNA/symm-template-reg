import unittest

from symm_template_reg.engine.training_budget import resolve_training_budget


class BudgetManifestScalingTest(unittest.TestCase):
    def test_incomplete_batch_keeps_exact_epoch_exposure(self):
        budget = resolve_training_budget(
            {"mode": "sample_exposures", "target_exposures_per_sample": 1500},
            selected_samples=10, batch_size=4, gradient_accumulation_steps=1,
            drop_last=False, configured_max_optimizer_steps=None,
            configured_max_epochs=1,
        )
        self.assertEqual(budget.batches_per_epoch, 3)
        self.assertEqual(budget.computed_max_optimizer_steps, 4500)


if __name__ == "__main__": unittest.main()
