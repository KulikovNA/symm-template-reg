import unittest

import torch

from symm_template_reg.models.heads.surface_constrained_correspondence_head_v2 import (
    SurfaceConstrainedCorrespondenceHeadV2,
)


class TeacherForcingGTExactlyTest(unittest.TestCase):
    def test_probability_one_includes_exact_gt_triangle(self):
        torch.manual_seed(2)
        head = SurfaceConstrainedCorrespondenceHeadV2(
            embed_dim=4,
            num_patches=2,
            top_k_patches=1,
            local_candidates=1,
            teacher_forcing_initial_probability=1.0,
            teacher_forcing_final_probability=1.0,
        )
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]]
        )
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        template_points = torch.tensor([[[0.2, 0.2, 0.0], [2.2, 0.2, 0.0]]])
        result = head(
            torch.randn(1, 1, 4),
            torch.randn(1, 2, 4),
            template_points,
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
            template_mesh_vertices_O=[vertices],
            template_mesh_faces=[faces],
            teacher_forcing_target_points_O=torch.tensor([[[2.2, 0.2, 0.0]]]),
        )
        aux = result["auxiliary"]
        gt = aux["teacher_forcing_gt_triangle_ids"][0, 0]
        self.assertTrue(bool(aux["candidate_triangle_ids"][0, 0].eq(gt).any()))


if __name__ == "__main__":
    unittest.main()
