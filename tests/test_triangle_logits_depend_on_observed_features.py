import unittest
import torch

from symm_template_reg.models.heads.surface_constrained_correspondence_head_v2 import point_conditioned_candidate_scores


class TriangleObservedConditioningTest(unittest.TestCase):
    def test_swapping_observed_features_swaps_logit_rows(self):
        q=torch.tensor([[1.,0.],[0.,1.]]); c=torch.tensor([[[1.,0.],[0.,1.]],[[1.,0.],[0.,1.]]])
        original=point_conditioned_candidate_scores(q,c)
        swapped=point_conditioned_candidate_scores(q.flip(0),c)
        self.assertTrue(torch.equal(swapped,original.flip(0)))
        zero_observed = point_conditioned_candidate_scores(torch.zeros_like(q),c)
        self.assertFalse(torch.equal(zero_observed, original))
        target = torch.tensor([0, 1])
        original_accuracy = original.argmax(-1).eq(target).float().mean()
        zero_accuracy = zero_observed.argmax(-1).eq(target).float().mean()
        self.assertGreater(float(original_accuracy), float(zero_accuracy))


if __name__ == "__main__": unittest.main()
