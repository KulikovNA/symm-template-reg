import unittest, torch
from tests.optimization_test_utils import loss_pair

class MixedC2C4Test(unittest.TestCase):
    def test_padding_mask_and_selection(self):
        _,_,_,result=loss_pair(); values=result["loss_by_symmetry_element"]
        self.assertEqual(tuple(values.shape),(2,4)); self.assertTrue(torch.isnan(values[0,2:]).all()); self.assertTrue((result["selected_shared_symmetry_element"]<torch.tensor([2,4])).all())

