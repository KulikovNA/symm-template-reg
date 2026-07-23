import unittest
import torch


class FreeCapacityTest(unittest.TestCase):
    def test_free_q_converges_nearly_to_zero(self):
        target=torch.tensor([[.01,.02,.03],[-.02,.01,.04]])
        value=torch.nn.Parameter(target+0.01)
        optimizer=torch.optim.Adam([value],lr=.05)
        for _ in range(300):
            optimizer.zero_grad();loss=(value-target).square().mean();loss.backward();optimizer.step()
        self.assertLess(float(torch.linalg.vector_norm(value.detach()-target,dim=-1).max()),1e-5)
