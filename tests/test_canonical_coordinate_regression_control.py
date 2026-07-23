import unittest
import torch
from symm_template_reg.models.heads.canonical_coordinate_regression_control import CanonicalCoordinateRegressionControl


class CoordinateControlTest(unittest.TestCase):
    def test_output_is_bounded_to_template_bbox(self):
        head = CanonicalCoordinateRegressionControl(embed_dim=8, hidden_dim=16)
        vertices = torch.tensor([[-1., -2., -3.], [2., 3., 4.]])
        output = head(torch.randn(1, 7, 8), torch.randn(1, 2, 8), vertices[None], torch.ones((1, 7), dtype=torch.bool), torch.ones((1, 2), dtype=torch.bool), template_mesh_vertices_O=[vertices], template_mesh_faces=[torch.empty((0, 3), dtype=torch.long)])
        self.assertTrue(bool((output["points_O"][0] >= vertices.amin(0)).all()))
        self.assertTrue(bool((output["points_O"][0] <= vertices.amax(0)).all()))
