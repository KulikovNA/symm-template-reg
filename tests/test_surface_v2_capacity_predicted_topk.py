import unittest
import torch
from symm_template_reg.models.heads.surface_constrained_correspondence_head_v2 import SurfaceConstrainedCorrespondenceHeadV2


class PredictedTopKCapacityTest(unittest.TestCase):
    def test_gt_injection_preserves_topk_width(self):
        predicted = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]])
        result, forced = SurfaceConstrainedCorrespondenceHeadV2.inject_gt_patch(predicted, torch.tensor([7, 3]), 1.0)
        self.assertEqual(result.shape, predicted.shape)
        self.assertTrue(result[0].eq(7).any())
        self.assertTrue(bool(forced[0]))

    def test_patch_candidates_cover_every_template_face(self):
        torch.manual_seed(0)
        head = SurfaceConstrainedCorrespondenceHeadV2(
            embed_dim=8, num_patches=4, top_k_patches=2, local_candidates=2
        )
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        output = head(
            torch.randn(1, 5, 8), torch.randn(1, 4, 8), vertices[None],
            torch.ones((1, 5), dtype=torch.bool), torch.ones((1, 4), dtype=torch.bool),
            template_mesh_vertices_O=[vertices], template_mesh_faces=[faces],
        )
        covered = torch.unique(output["auxiliary"]["all_candidate_triangle_ids"])
        self.assertEqual(set(covered.tolist()), set(range(len(faces))))
