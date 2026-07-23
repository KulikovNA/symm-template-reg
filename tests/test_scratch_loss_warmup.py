import unittest

from symm_template_reg.models.losses.clean_coordinate_pose_loss_v3 import scratch_warmup_progress


class ScratchWarmupTest(unittest.TestCase):
    def test_linear_loss_weight_warmup(self):
        self.assertEqual(scratch_warmup_progress(0, 250), 0.0)
        self.assertAlmostEqual(scratch_warmup_progress(249, 250), 0.996)
        self.assertEqual(scratch_warmup_progress(250, 250), 1.0)
        self.assertEqual(scratch_warmup_progress(900, 250), 1.0)
        self.assertEqual(scratch_warmup_progress(0, 0), 1.0)


if __name__ == "__main__": unittest.main()
