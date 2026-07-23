import unittest

import torch

from symm_template_reg.models.losses.joint_surface_correspondence_pose_loss_v3 import per_sample_mean_then_batch_mean


class EqualSampleWeightingTest(unittest.TestCase):
    def test_point_count_does_not_change_sample_weight(self):
        short = torch.tensor([0.0, 2.0]).mean()
        long = torch.full((200,), 3.0).mean()
        reduced = per_sample_mean_then_batch_mean([short, long])
        self.assertEqual(float(reduced), 2.0)


if __name__ == "__main__": unittest.main()
