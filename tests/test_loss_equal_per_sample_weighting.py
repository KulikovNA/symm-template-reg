import unittest

import torch

from symm_template_reg.models.losses.joint_surface_correspondence_pose_loss_v3 import (
    per_sample_mean_then_batch_mean,
)


class EqualPerSampleWeightingTest(unittest.TestCase):
    def test_point_count_does_not_change_sample_weight(self):
        short = torch.ones(1)
        long = torch.full((100,), 3.0)
        self.assertEqual(float(per_sample_mean_then_batch_mean([short, long])), 2.0)
        pooled = torch.cat((short, long)).mean()
        self.assertNotAlmostEqual(float(pooled), 2.0)


if __name__ == "__main__":
    unittest.main()
