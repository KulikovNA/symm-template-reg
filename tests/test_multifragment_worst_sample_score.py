import unittest
from symm_template_reg.engine.multifragment_overfit import worst_multifragment_sample_score
from tests.multifragment_test_utils import metric_rows
class TestWorst(unittest.TestCase):
    def test_bad_sample_controls_score(self): self.assertGreater(worst_multifragment_sample_score(metric_rows(metric_rows()[0]["sample_id"])),4.0)

