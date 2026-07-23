import unittest
import torch

from symm_template_reg.models.heads.surface_constrained_correspondence_head_v2 import point_conditioned_candidate_scores


class TriangleCandidateConditioningTest(unittest.TestCase):
    def test_candidate_feature_permutation_permutes_logits(self):
        q=torch.tensor([[1.,0.],[0.,1.]])
        c=torch.tensor([[[1.,0.],[0.,1.]],[[1.,0.],[0.,1.]]])
        order=torch.tensor([1,0])
        self.assertTrue(torch.allclose(point_conditioned_candidate_scores(q,c[:,order]),point_conditioned_candidate_scores(q,c)[:,order]))
        original = point_conditioned_candidate_scores(q,c)
        zero_geometry = point_conditioned_candidate_scores(q,torch.zeros_like(c))
        self.assertFalse(torch.equal(zero_geometry, original))
        target = torch.tensor([0, 1])
        original_accuracy = original.argmax(-1).eq(target).float().mean()
        zero_accuracy = zero_geometry.argmax(-1).eq(target).float().mean()
        self.assertGreater(float(original_accuracy), float(zero_accuracy))


if __name__ == "__main__": unittest.main()
