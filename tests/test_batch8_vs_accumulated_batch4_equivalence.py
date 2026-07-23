import copy
import unittest

import torch


class BatchEquivalenceTest(unittest.TestCase):
    def test_gradients_and_one_step_match(self):
        torch.manual_seed(0)
        full = torch.nn.Linear(3, 2)
        accumulated = copy.deepcopy(full)
        x, y = torch.randn(8, 3), torch.randn(8, 2)
        opt_full = torch.optim.AdamW(full.parameters(), lr=1e-4, weight_decay=0.0)
        opt_acc = torch.optim.AdamW(accumulated.parameters(), lr=1e-4, weight_decay=0.0)
        torch.nn.functional.mse_loss(full(x), y, reduction="none").mean(-1).mean().backward()
        for start in (0, 4):
            loss = torch.nn.functional.mse_loss(
                accumulated(x[start:start + 4]), y[start:start + 4], reduction="none"
            ).mean(-1).mean() / 2.0
            loss.backward()
        for left, right in zip(full.parameters(), accumulated.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-6, atol=1e-7)
        opt_full.step(); opt_acc.step()
        for left, right in zip(full.parameters(), accumulated.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
