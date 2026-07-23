import unittest

import torch

from symm_template_reg.models.geometry.patch_targets import multi_positive_softmax_loss


class MultiValidPatchLossTest(unittest.TestCase):
    def test_one_valid_patch_matches_cross_entropy_behavior(self):
        logits = torch.tensor([[2.0, -1.0, 0.0]])
        valid = torch.tensor([[True, False, False]])
        expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
        self.assertTrue(torch.allclose(multi_positive_softmax_loss(logits, valid), expected))

    def test_either_valid_patch_can_receive_probability_mass(self):
        valid = torch.tensor([[True, True, False]])
        first = multi_positive_softmax_loss(torch.tensor([[20.0, 0.0, -20.0]]), valid)
        second = multi_positive_softmax_loss(torch.tensor([[0.0, 20.0, -20.0]]), valid)
        invalid = multi_positive_softmax_loss(torch.tensor([[0.0, 0.0, 20.0]]), valid)
        self.assertLess(float(first), 1e-6)
        self.assertLess(float(second), 1e-6)
        self.assertGreater(float(invalid), 10.0)


if __name__ == "__main__":
    unittest.main()
