import unittest, torch
from tests.optimization_test_utils import loss_pair

class OptimizedGradientTest(unittest.TestCase):
    def test_q_gradient(self):
        left_inputs,right_inputs,left,right=loss_pair(True); left["loss_total"].backward(); right["loss_total"].backward()
        self.assertTrue(torch.allclose(left_inputs["predicted_normalized_O"].grad,right_inputs["predicted_normalized_O"].grad,atol=1e-5,rtol=1e-5))

