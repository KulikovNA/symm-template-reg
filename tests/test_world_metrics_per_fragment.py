import unittest
from collections import Counter
from tests.multifragment_test_utils import metric_rows
class TestWorldGrouping(unittest.TestCase):
    def test_each_physical_fragment_has_four_views(self): self.assertEqual(Counter(r["fragment_id"] for r in metric_rows()),Counter({0:4,1:4,2:4,3:4}))

