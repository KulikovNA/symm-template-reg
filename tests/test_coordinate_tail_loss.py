import unittest, torch
from symm_template_reg.models.losses.joint_surface_correspondence_pose_loss_v3 import coordinate_mean_and_tail_loss


class CoordinateTailLossTest(unittest.TestCase):
    def test_worst_ten_percent_dominates_tail(self):
        prediction=torch.zeros(10,3); target=torch.zeros(1,10,3); target[0,-1]=1
        mean,tail=coordinate_mean_and_tail_loss(prediction,target,.1)
        self.assertGreater(float(tail),float(mean)); self.assertAlmostEqual(float(tail),.5,places=6)


if __name__ == "__main__": unittest.main()

