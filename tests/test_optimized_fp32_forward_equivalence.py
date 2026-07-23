import unittest, torch
from tests.optimization_test_utils import loss_pair

class OptimizedForwardTest(unittest.TestCase):
    def test_loss_and_selection(self):
        _,_,left,right=loss_pair(); self.assertLessEqual(abs(float(left["loss_total"]-right["loss_total"])),1e-6); self.assertTrue(torch.equal(left["selected_shared_symmetry_element"],right["selected_shared_symmetry_element"]))

