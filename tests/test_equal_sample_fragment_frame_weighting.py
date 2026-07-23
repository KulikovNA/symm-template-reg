import unittest
from symm_template_reg.engine.multifragment_overfit import equal_sample_weights
from tests.multifragment_test_utils import samples
class TestWeights(unittest.TestCase):
    def test_equal(self):
        value=equal_sample_weights(samples()); self.assertEqual(set(value["per_sample"].values()), {1/16}); self.assertEqual(set(value["per_fragment"].values()), {1/4}); self.assertEqual(set(value["per_frame"].values()), {1/4})

