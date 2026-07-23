import unittest, torch
from tests.test_coordinate_guided_surface_projection import fixture
from symm_template_reg.geometry import closest_points_on_triangle_mesh
from symm_template_reg.models.heads.coordinate_guided_surface_projection import CoordinateGuidedSurfaceProjectionHead


class ProjectionMembershipTest(unittest.TestCase):
    def test_surface_distance_is_zero(self):
        v,f,q,ids,mask=fixture(); p=CoordinateGuidedSurfaceProjectionHead()(q,ids,[v],[f],mask)["surface_correspondence_points_O"][0]
        self.assertLess(float(closest_points_on_triangle_mesh(p,v,f)["distances"].max()),1e-6)


if __name__ == "__main__": unittest.main()

