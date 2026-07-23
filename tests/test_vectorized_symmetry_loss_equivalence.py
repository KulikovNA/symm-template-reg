import unittest, torch
from tests.optimization_test_utils import loss_pair

class VectorizedSymmetryLossTest(unittest.TestCase):
    def test_all_scalar_components_match(self):
        _,_,left,right=loss_pair()
        for name,value in left.items():
            if isinstance(value,torch.Tensor) and value.ndim==0: self.assertTrue(torch.allclose(value,right[name],atol=1e-6,rtol=1e-6),name)

