import unittest
from symm_template_reg.engine.multifragment_overfit import aggregate_metrics
from tests.multifragment_test_utils import metric_rows
class TestFrameMetrics(unittest.TestCase):
    def test_four_groups(self): self.assertEqual([x["sample_count"] for x in aggregate_metrics(metric_rows(),"frame_id")],[4,4,4,4])

