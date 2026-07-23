"""Separate invariant matching features from orientation-sensitive pose features."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from symm_template_reg.registry import GEOMETRY_MODULES, build_from_cfg


@GEOMETRY_MODULES.register_module()
class DualStreamGeometryEncoder(nn.Module):
    """Preserve coordinate-sensitive features while enriching matching only."""

    def __init__(
        self,
        embed_dim: int = 256,
        matching_geometric_embedding: Mapping[str, Any] | None = None,
        matching_ppf_embedding: Mapping[str, Any] | None = None,
        matching_geometric_only: bool = False,
        matching_only: bool = False,
    ) -> None:
        super().__init__()
        self.matching_geometric_embedding = (
            build_from_cfg(matching_geometric_embedding, GEOMETRY_MODULES)
            if matching_geometric_embedding is not None
            else None
        )
        self.matching_ppf_embedding = (
            build_from_cfg(matching_ppf_embedding, GEOMETRY_MODULES)
            if matching_ppf_embedding is not None
            else None
        )
        self.matching_geometric_only = bool(matching_geometric_only)
        self.matching_only = bool(matching_only)
        self.matching_norm = nn.LayerNorm(embed_dim)
        if not self.matching_only:
            self.pose_norm = nn.LayerNorm(embed_dim)

    def matching_geometry(
        self,
        points: Tensor,
        valid_mask: Tensor,
        normals: Tensor | None = None,
        precomputed_indices: Tensor | None = None,
    ) -> Tensor:
        result: Tensor | float = 0.0
        if self.matching_geometric_embedding is not None:
            result = result + self.matching_geometric_embedding(
                points, valid_mask, precomputed_indices
            )
        if self.matching_ppf_embedding is not None:
            result = result + self.matching_ppf_embedding(
                points, valid_mask, normals, precomputed_indices
            )
        if isinstance(result, float):
            return points.new_zeros((*points.shape[:2], self.matching_norm.normalized_shape[0]))
        return result * valid_mask.unsqueeze(-1)

    def finalize_matching(
        self, cross_features: Tensor, geometry: Tensor, valid_mask: Tensor
    ) -> Tensor:
        return self.matching_norm(cross_features + geometry) * valid_mask.unsqueeze(-1)

    def pose_features(self, features: Tensor, valid_mask: Tensor) -> Tensor:
        if self.matching_only:
            raise RuntimeError("matching-only geometry encoder has no pose stream")
        return self.pose_norm(features) * valid_mask.unsqueeze(-1)

    def forward(
        self,
        features: Tensor,
        points: Tensor,
        valid_mask: Tensor,
        normals: Tensor | None = None,
        precomputed_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if features.shape[:2] != points.shape[:2] or valid_mask.shape != points.shape[:2]:
            raise ValueError("dual-stream features, points and mask shapes disagree")
        geometric = self.matching_geometry(
            points, valid_mask, normals, precomputed_indices
        )
        matching = self.finalize_matching(features, geometric, valid_mask)
        pose = None if self.matching_only else self.pose_features(features, valid_mask)
        mask = valid_mask.unsqueeze(-1)
        return {
            "matching_features": matching * mask,
            **({} if pose is None else {"pose_features": pose}),
        }


__all__ = ["DualStreamGeometryEncoder"]
