import unittest

import torch

from symm_template_reg.models.geometry.point_ops import nearest_grouped_point_ids
from symm_template_reg.models.heads.surface_constrained_correspondence_head_v2 import (
    selected_patch_candidate_scores,
)


class SurfaceV2MemoryBoundedScoringTest(unittest.TestCase):
    def test_sparse_scores_and_gradients_equal_dense_reference(self):
        torch.manual_seed(7)
        topk = torch.tensor([[0, 2], [1, 3], [3, 0]])
        query_sparse = torch.randn(3, 5, requires_grad=True)
        feature_sparse = torch.randn(4, 6, 5, requires_grad=True)
        query_dense = query_sparse.detach().clone().requires_grad_(True)
        feature_dense = feature_sparse.detach().clone().requires_grad_(True)

        actual = selected_patch_candidate_scores(
            query_sparse, feature_sparse, topk
        )
        dense = torch.einsum("nd,pcd->npc", query_dense, feature_dense)
        expected = dense.gather(
            1, topk[..., None].expand(-1, -1, dense.shape[-1])
        ).reshape(3, -1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

        weights = torch.randn_like(actual)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        self.assertTrue(
            torch.allclose(query_sparse.grad, query_dense.grad, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(feature_sparse.grad, feature_dense.grad, atol=1e-6)
        )

    def test_grouped_nearest_ids_match_full_distance_matrix(self):
        torch.manual_seed(11)
        query = torch.randn(9, 3)
        support = torch.randn(4, 7, 3)
        expected = torch.cdist(query, support.reshape(-1, 3)).reshape(
            len(query), len(support), -1
        ).amin(-1).argmin(-1)
        actual = nearest_grouped_point_ids(query, support, chunk_size=2)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
