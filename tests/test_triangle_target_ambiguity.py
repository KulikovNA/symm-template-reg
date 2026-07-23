import unittest
import torch

from symm_template_reg.models.geometry.triangle_targets import triangle_target_sets


class TriangleTargetAmbiguityTest(unittest.TestCase):
    def test_shared_diagonal_has_two_valid_triangles(self):
        vertices = torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
        faces = torch.tensor([[0,1,2],[0,2,3]])
        result = triangle_target_sets(torch.tensor([[.5,.5,0.]]), vertices, faces, tolerance_m=1e-7)
        self.assertEqual(int(result["valid_triangle_mask"].sum()), 2)


if __name__ == "__main__": unittest.main()
