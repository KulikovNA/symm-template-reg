import unittest
import torch
from symm_template_reg.models.heads.surface_constrained_correspondence_head_v2 import SurfaceConstrainedCorrespondenceHeadV2


class TeacherForcingInclusionTest(unittest.TestCase):
    def test_probability_one_guarantees_inclusion(self):
        topk = torch.tensor([[0, 1, 2, 3]] * 16)
        gt = torch.arange(16) % 8
        result, _ = SurfaceConstrainedCorrespondenceHeadV2.inject_gt_patch(topk, gt, 1.0)
        self.assertTrue(bool(result.eq(gt[:, None]).any(-1).all()))

