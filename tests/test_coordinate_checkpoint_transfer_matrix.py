import unittest
from pathlib import Path
class TransferMatrixTest(unittest.TestCase):
    def test_tool_declares_four_combinations_and_outputs(self):
        t=Path('tools/audit_coordinate_checkpoint_transfer.py').read_text();self.assertIn('for ci,checkpoint',t);self.assertIn('for mi,manifest',t);self.assertIn('coordinate_checkpoint_transfer_matrix.csv',t);self.assertIn('selected_two_view_initialization.json',t)
