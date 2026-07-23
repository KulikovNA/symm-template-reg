import unittest

import torch

from symm_template_reg.models import build_model


def tiny_model_config():
    return dict(
        type="SymmTemplateReg",
        embed_dim=32,
        max_observed_tokens=8,
        max_template_tokens=8,
        observed_encoder=dict(type="SimplePointEncoder", embed_dim=32, hidden_dim=16, num_neighbors=3),
        template_encoder=dict(type="SimplePointEncoder", embed_dim=32, hidden_dim=16, num_neighbors=3),
        geometric_embedding=dict(type="GeometricStructureEmbedding", embed_dim=32, num_neighbors=2),
        coarse_matcher=dict(type="CoarseMatching", temperature=0.2),
        interaction_transformer=dict(
            type="RegTRInteractionTransformer", embed_dim=32, num_heads=4, num_layers=1, feedforward_dim=64
        ),
        overlap_head=dict(type="OverlapHead", embed_dim=32, hidden_dim=16),
        correspondence_head=dict(type="CorrespondenceHead", embed_dim=32),
        point_weight_head=dict(type="PointWeightHead", embed_dim=32),
        symmetry_head=dict(type="SymmetryRegionHead", embed_dim=32, max_regions=5),
        pose_head=dict(
            type="PoseQueryHead", embed_dim=32, num_heads=4, num_queries=4, num_decoder_layers=2, feedforward_dim=64
        ),
    )


def variable_batch():
    observed = torch.randn(2, 11, 3)
    observed_mask = torch.tensor([[1] * 11, [1] * 7 + [0] * 4], dtype=torch.bool)
    template = torch.randn(2, 13, 3)
    template_mask = torch.tensor([[1] * 9 + [0] * 4, [1] * 13], dtype=torch.bool)
    return {
        "observed": {"points_C": observed, "valid_mask": observed_mask},
        "template": {"points_O": template, "valid_mask": template_mask},
        "meta": [{"symmetry_available": False}, {"symmetry_available": True}],
    }


class ModelForwardTest(unittest.TestCase):
    def test_variable_size_forward_shapes(self):
        torch.manual_seed(3)
        model = build_model(tiny_model_config()).eval()
        with torch.no_grad():
            prediction = model(variable_batch())
        self.assertEqual(tuple(prediction.pose_hypotheses.shape), (2, 4, 4, 4))
        self.assertEqual(tuple(prediction.pose_logits.shape), (2, 4))
        self.assertEqual(tuple(prediction.observed_overlap_logits.shape), (2, 11))
        self.assertEqual(tuple(prediction.template_visibility_logits.shape), (2, 13))
        self.assertEqual(tuple(prediction.correspondence_points_O.shape), (2, 11, 3))
        self.assertEqual(tuple(prediction.active_region_logits.shape), (2, 5))
        self.assertEqual(prediction.symmetry_available.tolist(), [False, True])
        determinants = torch.linalg.det(prediction.pose_hypotheses[..., :3, :3])
        self.assertTrue(torch.allclose(determinants, torch.ones_like(determinants), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
