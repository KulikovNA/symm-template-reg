import unittest
from tools.audit_coordinate_checkpoint_transfer import select_initialization
class SelectionTest(unittest.TestCase):
    def test_uses_worst_frame_not_self_frame(self):
        rows=[{'checkpoint':'a','manifest_frame_id':4,'physical_normalized_score':.1,'exact_global_gate_passed':True},{'checkpoint':'a','manifest_frame_id':8,'physical_normalized_score':9.,'exact_global_gate_passed':False},{'checkpoint':'b','manifest_frame_id':4,'physical_normalized_score':2.,'exact_global_gate_passed':True},{'checkpoint':'b','manifest_frame_id':8,'physical_normalized_score':2.,'exact_global_gate_passed':True}]
        self.assertEqual(select_initialization(rows)['selected_checkpoint'],'b')
