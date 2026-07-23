"""Point- and candidate-conditioned scoring over a fixed local triangle set."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import HEADS


@HEADS.register_module()
class FineCandidateTriangleHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        observed_geometry_dim: int = 30,
        candidate_geometry_dim: int = 22,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.observed_geometry_dim = int(observed_geometry_dim)
        self.candidate_geometry_dim = int(candidate_geometry_dim)
        pair_dim = 4 * embed_dim + observed_geometry_dim + candidate_geometry_dim
        self.scorer = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        fine_point_features: Tensor,
        candidate_triangle_features: Tensor,
        observed_local_geometry: Tensor,
        candidate_local_geometry: Tensor,
        candidate_mask: Tensor | None = None,
    ) -> Tensor:
        if fine_point_features.ndim != 2 or candidate_triangle_features.ndim != 3:
            raise ValueError("fine point/candidate tensors must be [N,D] and [N,L,D]")
        if candidate_triangle_features.shape[0] != len(fine_point_features):
            raise ValueError("point and candidate rows disagree")
        point = fine_point_features[:, None].expand_as(candidate_triangle_features)
        point_geometry = observed_local_geometry[:, None].expand(
            -1, candidate_triangle_features.shape[1], -1
        )
        pair = torch.cat(
            (
                point,
                candidate_triangle_features,
                point * candidate_triangle_features,
                point - candidate_triangle_features,
                point_geometry,
                candidate_local_geometry,
            ), -1,
        )
        logits = self.scorer(pair).squeeze(-1)
        return logits if candidate_mask is None else logits.masked_fill(~candidate_mask, float("-inf"))


__all__ = ["FineCandidateTriangleHead"]
