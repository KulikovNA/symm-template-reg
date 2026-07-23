import unittest

from symm_template_reg.engine.training_budget import resolve_training_budget


class SampleExposureBudgetTest(unittest.TestCase):
    def test_batch_one_expected_steps(self):
        for samples in (1, 2, 4, 10):
            budget = resolve_training_budget(
                {"mode": "sample_exposures", "target_exposures_per_sample": 1500},
                selected_samples=samples, batch_size=1,
                gradient_accumulation_steps=1, drop_last=False,
                configured_max_optimizer_steps=1500, configured_max_epochs=1500,
            )
            self.assertEqual(budget.computed_max_optimizer_steps, 1500 * samples)


if __name__ == "__main__": unittest.main()
