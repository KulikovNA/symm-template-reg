import unittest
from collections import Counter
from tests.multifragment_test_utils import samples
class TestCounts(unittest.TestCase):
    def test_four_each(self):
        rows=samples(); self.assertEqual(Counter(r["fragment_id"] for r in rows), Counter({0:4,1:4,2:4,3:4})); self.assertEqual(Counter(r["frame_id"] for r in rows), Counter({2:4,4:4,5:4,8:4}))

