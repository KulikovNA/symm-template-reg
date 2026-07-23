import unittest
import torch

from symm_template_reg.models.geometry.triangle_targets import local_valid_triangle_mask


class LocalTriangleTargetIndexTest(unittest.TestCase):
    def test_local_index_maps_back_to_global_gt(self):
        candidates = torch.tensor([[8, 3, 5]])
        valid_global = torch.zeros((1, 10), dtype=torch.bool); valid_global[0, 3] = True
        valid = local_valid_triangle_mask(candidates, valid_global)
        local = valid.long().argmax(-1)
        self.assertEqual(int(candidates[0, local]), 3)


if __name__ == "__main__": unittest.main()
