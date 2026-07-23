import unittest

import torch

from symm_template_reg.models.geometry.patch_targets import valid_patch_mask


class PatchTargetAmbiguityTest(unittest.TestCase):
    def test_alternative_patch_containing_same_triangle_is_correct(self):
        candidates = torch.tensor([[0, 1], [1, 2], [3, 4]])
        valid = valid_patch_mask(torch.tensor([1]), candidates)
        self.assertEqual(valid.tolist(), [[True, True, False]])
        predicted = torch.tensor([1])
        self.assertTrue(bool(valid.gather(-1, predicted[:, None]).item()))


if __name__ == "__main__":
    unittest.main()
