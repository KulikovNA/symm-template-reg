"""Clean-room geometric descriptors.

Architectural reference: GeoTransformer (https://github.com/qinzheng93/GeoTransformer), commit
e7a135af4c318ff3b8d7f6c963df094d7e4ea540, paths
``geotransformer/modules/geotransformer/geotransformer.py``,
``geotransformer/modules/ops/pairwise_distance.py`` and
``geotransformer/modules/transformer/positional_embedding.py`` (MIT).
No source text was copied. Changes: local invariant distance statistics replace
the O(N^2 k) triplet tensor; project-local kNN and masks; device-agnostic PyTorch.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from symm_template_reg.registry import GEOMETRY_MODULES

from .point_ops import batched_gather, knn_indices


class SinusoidalScalarEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        half = max(dim // 2, 1)
        frequencies = torch.exp(torch.linspace(0.0, -math.log(10_000.0), half))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.dim = dim

    def forward(self, value: Tensor) -> Tensor:
        phase = value.unsqueeze(-1) * self.frequencies
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return embedding[..., : self.dim]


@GEOMETRY_MODULES.register_module()
class GeometricStructureEmbedding(nn.Module):
    """Per-point rotation/translation-invariant embedding from local distances."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_neighbors: int = 8,
        distance_scale_m: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.distance_scale_m = distance_scale_m
        self.scalar_embedding = SinusoidalScalarEmbedding(embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self, points: Tensor, valid_mask: Tensor,
        precomputed_indices: Tensor | None = None,
    ) -> Tensor:
        indices = (
            knn_indices(points, points, valid_mask, self.num_neighbors + 1)
            if precomputed_indices is None else precomputed_indices
        )
        neighbor_indices = indices[..., 1:]
        if neighbor_indices.shape[-1] == 0:
            neighbor_indices = indices[..., :1]
        neighbors = batched_gather(points, neighbor_indices)
        distances = torch.linalg.vector_norm(neighbors - points.unsqueeze(-2), dim=-1)
        scaled = distances / max(self.distance_scale_m, 1e-8)
        mean_emb = self.scalar_embedding(scaled.mean(dim=-1))
        max_emb = self.scalar_embedding(scaled.max(dim=-1).values)
        result = self.projection(torch.cat((mean_emb, max_emb), dim=-1))
        return result * valid_mask.unsqueeze(-1)
