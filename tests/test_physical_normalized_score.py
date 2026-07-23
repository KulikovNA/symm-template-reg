import unittest
from symm_template_reg.engine.metrics import physical_normalized_score

class PhysicalScoreTest(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(physical_normalized_score(2,2,2,2,1),5.0)
