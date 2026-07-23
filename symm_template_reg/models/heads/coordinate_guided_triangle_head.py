"""Residual candidate classifier using q_aux-to-triangle object-frame geometry."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.models.geometry.triangle_targets import closest_barycentric_on_triangles
from symm_template_reg.registry import HEADS


@HEADS.register_module()
class CoordinateGuidedTriangleHead(nn.Module):
    """Fallback learned triangle scorer; analytic projection remains the output."""

    required_pair_inputs = (
        "fine_observed_feature", "candidate_triangle_feature", "q_aux_O",
        "closest_point_O", "q_aux_to_triangle_distance", "triangle_normal_O",
        "triangle_edge_lengths", "coarse_patch_feature",
    )

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        pair_dim = 5 * int(embed_dim) + 13
        self.input = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(3)
        ])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        fine_observed_feature: Tensor,
        candidate_triangle_feature: Tensor,
        q_aux_O: Tensor,
        candidate_triangle_vertices_O: Tensor,
        triangle_normal_O: Tensor,
        triangle_edge_lengths: Tensor,
        coarse_patch_feature: Tensor,
        candidate_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        n, candidate_count, dimension = candidate_triangle_feature.shape
        repeated_q = q_aux_O[:, None].expand(n, candidate_count, 3).reshape(-1, 3)
        projected = closest_barycentric_on_triangles(
            repeated_q, candidate_triangle_vertices_O.reshape(-1, 3, 3)
        )
        closest = projected["points"].reshape(n, candidate_count, 3)
        barycentric = projected["barycentric"].reshape(n, candidate_count, 3)
        distance = projected["distances"].reshape(n, candidate_count, 1)
        point = fine_observed_feature[:, None].expand(-1, candidate_count, -1)
        q = q_aux_O[:, None].expand(-1, candidate_count, -1)
        pair = torch.cat((
            point, candidate_triangle_feature, point * candidate_triangle_feature,
            point - candidate_triangle_feature, coarse_patch_feature,
            q, closest, distance, triangle_normal_O, triangle_edge_lengths,
        ), -1)
        value = self.input(pair)
        for block in self.blocks:
            value = torch.nn.functional.gelu(value + block(value))
        logits = self.output(value).squeeze(-1)
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, float("-inf"))
        return {
            "logits": logits,
            "candidate_closest_points_O": closest,
            "candidate_analytic_barycentric": barycentric,
            "candidate_q_aux_distance": distance.squeeze(-1),
        }


__all__ = ["CoordinateGuidedTriangleHead"]
