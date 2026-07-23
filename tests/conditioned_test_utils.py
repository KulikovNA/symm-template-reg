from __future__ import annotations

import torch


def tiny_conditioned_config(num_hypotheses: int = 3, *, svd: bool = False) -> dict:
    dim = 16
    return dict(
        type="ConditionedSymmTemplateReg",
        embed_dim=dim,
        max_observed_tokens=6,
        max_template_tokens=6,
        observed_encoder=dict(
            type="SimplePointEncoder", embed_dim=dim, hidden_dim=8, num_neighbors=2
        ),
        template_encoder=dict(
            type="SimplePointEncoder", embed_dim=dim, hidden_dim=8, num_neighbors=2
        ),
        interaction_transformer=dict(
            type="RegTRInteractionTransformer",
            embed_dim=dim,
            num_heads=4,
            num_layers=1,
            feedforward_dim=32,
            dropout=0.0,
        ),
        dual_stream_geometry_encoder=dict(
            type="DualStreamGeometryEncoder",
            embed_dim=dim,
            matching_geometric_embedding=dict(
                type="GeometricStructureEmbedding",
                embed_dim=dim,
                num_neighbors=2,
            ),
        ),
        sample_context_aggregator=dict(
            type="SampleConditionedContextAggregator", embed_dim=dim
        ),
        base_pose_head=dict(
            type="ConditionedBasePoseHead",
            embed_dim=dim,
            hidden_dim=32,
            pose_codec=dict(type="PoseCodec"),
        ),
        residual_pose_head=dict(
            type="ResidualPoseHypothesisHead",
            embed_dim=dim,
            num_heads=4,
            num_hypotheses=num_hypotheses,
            num_decoder_layers=2,
            feedforward_dim=32,
            query_conditioning=dict(
                type="film",
                apply_each_decoder_layer=True,
                allow_unconditioned_bypass=False,
            ),
        ),
        correspondence_head=dict(type="CorrespondenceHead", embed_dim=dim),
        overlap_head=dict(type="OverlapHead", embed_dim=dim, hidden_dim=8),
        point_weight_head=dict(type="PointWeightHead", embed_dim=dim),
        weighted_procrustes=(dict(type="WeightedProcrustes") if svd else None),
    )


def conditioned_batch() -> dict:
    observed = torch.tensor(
        [
            [
                [0.00, 0.00, 0.40],
                [0.03, 0.01, 0.41],
                [-0.01, 0.04, 0.39],
                [0.02, -0.03, 0.42],
                [-0.04, 0.01, 0.38],
                [0.01, 0.02, 0.44],
            ],
            [
                [0.20, 0.00, 0.60],
                [0.21, -0.04, 0.62],
                [0.25, 0.01, 0.59],
                [0.17, 0.03, 0.61],
                [0.23, 0.04, 0.57],
                [0.18, -0.02, 0.64],
            ],
        ],
        dtype=torch.float32,
    )
    template_one = torch.tensor(
        [
            [-0.04, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.00, 0.04, 0.01],
            [0.00, -0.04, -0.01],
            [0.02, 0.01, 0.04],
            [-0.02, -0.01, -0.04],
        ],
        dtype=torch.float32,
    )
    template = template_one.unsqueeze(0).expand(2, -1, -1).clone()
    mask = torch.ones((2, 6), dtype=torch.bool)
    return {
        "observed": {"points_C": observed, "valid_mask": mask},
        "template": {"points_O": template, "valid_mask": mask},
        "meta": [
            {"symmetry_available": False},
            {"symmetry_available": False},
        ],
    }
