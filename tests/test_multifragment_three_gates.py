import unittest
from symm_template_reg.engine.multifragment_overfit import multifragment_stage_gates
from tests.multifragment_test_utils import metric_rows
class TestGates(unittest.TestCase):
    def test_one_bad_sample_blocks_practical(self):
        rows=metric_rows(metric_rows()[0]["sample_id"]); gate=multifragment_stage_gates(rows); self.assertFalse(gate["stage_passed"]); self.assertEqual(gate["practical_surface_gate"]["passed_sample_count"],15)

