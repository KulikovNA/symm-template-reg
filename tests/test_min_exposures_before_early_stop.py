import unittest

from symm_template_reg.engine.training_budget import early_stopping_is_eligible


class EarlyStopExposureFloorTest(unittest.TestCase):
    def test_minimum_controls_eligibility(self):
        self.assertFalse(early_stopping_is_eligible({"a": 750, "b": 749}, 750))
        self.assertTrue(early_stopping_is_eligible({"a": 750, "b": 800}, 750))


if __name__ == "__main__": unittest.main()
