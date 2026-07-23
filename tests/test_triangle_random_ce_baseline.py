import math
import unittest
import torch

from symm_template_reg.models.geometry.patch_targets import multi_positive_softmax_loss


class TriangleRandomCEBaselineTest(unittest.TestCase):
    def test_uniform_single_target_is_log_candidate_count(self):
        logits = torch.zeros((1, 32)); valid = torch.zeros((1, 32), dtype=torch.bool); valid[0, 7] = True
        self.assertAlmostEqual(float(multi_positive_softmax_loss(logits, valid)), math.log(32), places=5)


if __name__ == "__main__": unittest.main()
