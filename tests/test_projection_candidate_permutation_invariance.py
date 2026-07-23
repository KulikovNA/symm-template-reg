import unittest, torch
from tests.test_coordinate_guided_surface_projection import fixture
from symm_template_reg.models.heads.coordinate_guided_surface_projection import CoordinateGuidedSurfaceProjectionHead


class ProjectionPermutationTest(unittest.TestCase):
    def test_permutation_does_not_change_points(self):
        v,f,q,ids,mask=fixture(); h=CoordinateGuidedSurfaceProjectionHead(); a=h(q,ids,[v],[f],mask); b=h(q,ids.flip(-1),[v],[f],mask)
        self.assertTrue(torch.allclose(a["surface_correspondence_points_O"],b["surface_correspondence_points_O"],atol=1e-6))


if __name__ == "__main__": unittest.main()

