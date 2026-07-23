import unittest
import torch


class FrozenFeatureCoordinateRegressionTest(unittest.TestCase):
    def test_linear_control_can_fit_linear_coordinates(self):
        torch.manual_seed(0); x=torch.randn(32,6); w=torch.randn(6,3); y=x@w
        model=torch.nn.Linear(6,3,bias=False); opt=torch.optim.Adam(model.parameters(),lr=.1)
        for _ in range(200):
            loss=(model(x)-y).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
        self.assertLess(float((model(x)-y).abs().max()), 1e-3)


if __name__ == "__main__": unittest.main()
