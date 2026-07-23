import unittest
from tests.multifragment_test_utils import samples
class TestShell(unittest.TestCase):
    def test_no_fracture_input(self): self.assertTrue(all(r["registration_point_selection"]=="shell_only" and r["fracture_point_count"]==0 for r in samples()))

