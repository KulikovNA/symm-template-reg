import unittest
import torch
from symm_template_reg.models.heads.soft_coarse_local_surface_correspondence_head import SoftCoarseLocalSurfaceCorrespondenceHead
from symm_template_reg.geometry import closest_points_on_triangle_mesh, nearest_triangles_on_mesh


class SoftCoarseHeadTest(unittest.TestCase):
    def test_candidates_use_triangle_distance_not_centroid_distance(self):
        # The large triangle contains the query but has a distant centroid.
        # A centroid heuristic would incorrectly select the small triangle.
        vertices = torch.tensor([
            [-10., -10., 0.], [10., -10., 0.], [0., 20., 0.],
            [.1, .1, 1.], [.2, .1, 1.], [.1, .2, 1.],
        ])
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        nearest = nearest_triangles_on_mesh(
            torch.tensor([[0., 0., .01]]), vertices, faces, 1
        )
        self.assertEqual(int(nearest["face_ids"][0, 0]), 0)
        self.assertAlmostEqual(float(nearest["distances"][0, 0]), .01, places=5)

    def test_output_is_on_triangle_surface(self):
        head = SoftCoarseLocalSurfaceCorrespondenceHead(embed_dim=8, nearest_triangle_candidates=2)
        observed = torch.randn(1, 5, 8); template_features = torch.randn(1, 4, 8)
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        output = head(observed, template_features, vertices[None], torch.ones((1, 5), dtype=torch.bool), torch.ones((1, 4), dtype=torch.bool), template_mesh_vertices_O=[vertices], template_mesh_faces=[faces])
        distance = closest_points_on_triangle_mesh(output["points_O"][0], vertices, faces)["distances"]
        self.assertLess(float(distance.max()), 1e-5)
