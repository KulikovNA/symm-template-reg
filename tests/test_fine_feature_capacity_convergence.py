import unittest
import torch
from tools.audit_fine_feature_capacity_v2 import CoordinateDiagnostic


class CapacityConvergenceTest(unittest.TestCase):
    def test_coordinate_probe_can_reduce_synthetic_loss(self):
        torch.manual_seed(0); x = torch.randn(32, 8); target = torch.tanh(x[:, :3]); model = CoordinateDiagnostic(8); opt = torch.optim.Adam(model.parameters(), 1e-2)
        initial = torch.nn.functional.mse_loss(model(x), target).item()
        for _ in range(30):
            loss = torch.nn.functional.mse_loss(model(x), target); opt.zero_grad(); loss.backward(); opt.step()
        self.assertLess(torch.nn.functional.mse_loss(model(x), target).item(), initial)


if __name__ == "__main__": unittest.main()

