"""Guaranteed baseline encoder implemented only with public PyTorch operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from symm_template_reg.registry import BACKBONES
from symm_template_reg.models.geometry.point_ops import (
    batched_gather,
    knn_indices,
    masked_max,
    masked_mean,
)


@dataclass
class EncodedPointCloud:
    points: Tensor
    point_features: Tensor
    global_feature: Tensor
    valid_mask: Tensor


def as_padded_points(
    points: Any,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Normalize tensors, dictionaries, or ``PackedPointBatch``-like inputs."""

    if isinstance(points, dict):
        valid_mask = points.get("valid_mask", valid_mask)
        points = points.get("points_C", points.get("points_O", points.get("points")))
    elif not isinstance(points, Tensor) and hasattr(points, "to_padded"):
        padded = points.to_padded()
        if isinstance(padded, tuple):
            points, valid_mask = padded[:2]
        elif isinstance(padded, dict):
            valid_mask = padded.get("valid_mask")
            points = padded.get("points", padded.get("points_C", padded.get("points_O")))
        else:
            valid_mask = getattr(padded, "valid_mask", None)
            points = getattr(padded, "points", padded)
    if not isinstance(points, Tensor):
        raise TypeError("could not extract a point tensor from the input")
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B,N,3] or [N,3]")
    if valid_mask is None:
        valid_mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
    elif valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)
    return points, valid_mask.bool()


@BACKBONES.register_module()
class SimplePointEncoder(nn.Module):
    """Point MLP + local kNN aggregation + masked global token."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_neighbors: int = 12,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.input_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.local_mlp = nn.Sequential(
            nn.Linear(embed_dim + 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        points: Any,
        valid_mask: Tensor | None = None,
        normals: Tensor | None = None,
        precomputed_indices: Tensor | None = None,
        samplewise_learned_ops: bool = False,
    ) -> EncodedPointCloud:
        del normals  # Reserved by the interchangeable encoder contract.
        points, valid_mask = as_padded_points(points, valid_mask)
        indices = (
            knn_indices(points, points, valid_mask, self.num_neighbors + 1)
            if precomputed_indices is None else precomputed_indices
        )
        if indices.shape[:2] != points.shape[:2]:
            raise ValueError("precomputed encoder kNN indices have incompatible shape")
        if samplewise_learned_ops and points.shape[0] > 1:
            rows = [
                self._encode_tensors(
                    points[index : index + 1],
                    valid_mask[index : index + 1],
                    indices[index : index + 1],
                )
                for index in range(points.shape[0])
            ]
            return EncodedPointCloud(
                points=torch.cat([row.points for row in rows]),
                point_features=torch.cat([row.point_features for row in rows]),
                global_feature=torch.cat([row.global_feature for row in rows]),
                valid_mask=torch.cat([row.valid_mask for row in rows]),
            )
        return self._encode_tensors(points, valid_mask, indices)

    def _encode_tensors(
        self, points: Tensor, valid_mask: Tensor, indices: Tensor
    ) -> EncodedPointCloud:
        features = self.input_mlp(points)
        neighbor_indices = indices[..., 1:]
        if neighbor_indices.shape[-1] == 0:
            neighbor_indices = indices[..., :1]
        neighbor_features = batched_gather(features, neighbor_indices)
        neighbor_points = batched_gather(points, neighbor_indices)
        relative = neighbor_points - points.unsqueeze(-2)
        local = self.local_mlp(torch.cat((neighbor_features, relative), dim=-1)).max(-2).values
        features = self.output_norm(features + local) * valid_mask.unsqueeze(-1)
        global_feature = self.global_mlp(
            torch.cat(
                (masked_mean(features, valid_mask, 1), masked_max(features, valid_mask, 1)),
                dim=-1,
            )
        )
        return EncodedPointCloud(points, features, global_feature, valid_mask)
