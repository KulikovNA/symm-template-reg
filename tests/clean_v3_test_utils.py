"""Small deterministic fixtures shared by clean-V3 contract tests."""

from __future__ import annotations

from functools import lru_cache

import torch

from symm_template_reg.models import build_model


def tiny_clean_v3_config() -> dict:
    dimension = 16
    return {
        "type": "CoordinateGuidedSurfaceRegistrationV3",
        "embed_dim": dimension,
        "max_observed_tokens": 12,
        "max_template_tokens": 14,
        "final_coordinate_initialization_std": 1e-3,
        "observed_encoder": {
            "type": "SimplePointEncoder", "embed_dim": dimension,
            "hidden_dim": 12, "num_neighbors": 4, "dropout": 0.0,
        },
        "template_encoder": {
            "type": "SimplePointEncoder", "embed_dim": dimension,
            "hidden_dim": 12, "num_neighbors": 4, "dropout": 0.0,
        },
        "interaction_transformer": {
            "type": "RegTRInteractionTransformer", "embed_dim": dimension,
            "num_heads": 4, "num_layers": 1, "feedforward_dim": 32,
            "dropout": 0.0,
        },
        "dual_stream_geometry_encoder": {
            "type": "DualStreamGeometryEncoder", "embed_dim": dimension,
            "matching_only": True,
            "matching_geometric_embedding": {
                "type": "GeometricStructureEmbedding", "embed_dim": dimension,
                "num_neighbors": 4, "distance_scale_m": 0.01,
            },
            "matching_ppf_embedding": {
                "type": "LocalPointPairFeatureEmbedding", "embed_dim": dimension,
                "num_neighbors": 4, "distance_scale_m": 0.01,
            },
        },
        "fine_feature_adapter": {
            "type": "FineLocalCorrespondenceFeatureAdapter",
            "embed_dim": dimension, "knn_scales": (8, 16, 32),
            "observed_only": True,
        },
        "canonical_coordinate_head": {
            "type": "FineCanonicalCoordinateAuxiliaryHead",
            "embed_dim": dimension, "hidden_dim": 16,
        },
        "weighted_procrustes": {
            "type": "WeightedProcrustes", "minimum_effective_points": 3.0,
            "rank_tolerance": 1e-7, "fail_on_degenerate": False,
        },
    }


def synthetic_batch(batch_size: int = 2, *, observed_offset: float = 0.0) -> dict:
    generator = torch.Generator().manual_seed(19)
    observed = torch.randn(batch_size, 18, 3, generator=generator) * 0.01
    observed[:, :, 0] += float(observed_offset)
    template = torch.randn(batch_size, 20, 3, generator=generator) * 0.015
    observed_mask = torch.ones((batch_size, 18), dtype=torch.bool)
    template_mask = torch.ones((batch_size, 20), dtype=torch.bool)
    return {
        "observed": {"points_C": observed, "valid_mask": observed_mask},
        "template": {"points_O": template, "valid_mask": template_mask},
        "template_mesh_vertices_O": [row.clone() for row in template],
        "template_mesh_faces": [
            torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
            for _ in range(batch_size)
        ],
    }


@lru_cache(maxsize=1)
def tiny_gradient_snapshot() -> tuple[tuple[str, ...], tuple[str, ...]]:
    torch.manual_seed(7)
    model = build_model(tiny_clean_v3_config()).train()
    prediction = model(synthetic_batch())
    loss = prediction.correspondence_auxiliary[
        "fine_aux_coordinate_normalized"
    ].square().mean()
    loss.backward()
    all_names = tuple(name for name, _ in model.named_parameters())
    missing = tuple(
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    )
    return all_names, missing

