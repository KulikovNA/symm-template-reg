import unittest, torch

class PerformanceLoggingTest(unittest.TestCase):
    def test_detached_batched_logging_is_observational(self):
        x=torch.tensor(2.,requires_grad=True); loss=x.square(); snapshot=torch.stack([loss.detach()]).cpu().tolist(); loss.backward()
        self.assertEqual(snapshot,[4.0]); self.assertEqual(float(x.grad),4.0)

