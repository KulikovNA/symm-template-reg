"""Pure-PyTorch point-pair features for optional invariant modules.

Architectural reference: RoITr (https://github.com/haoyu94/RoITr), commit
393539d6709c55b2465231cccb7b951f736a5c72, paths ``lib/utils.py``,
``dataset/common.py`` and ``model/transformer/positional_encoding.py`` (MIT).
No source text was copied. Changes: caller-provided tensors, stable clamped
angles, and no Open3D, NumPy, pointops, or CUDA dependency.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import GEOMETRY_MODULES

from .point_ops import batched_gather, knn_indices


def chunked_eigvalsh_3x3(
    matrices: Tensor, *, chunk_size: int = 512
) -> Tensor:
    """Eigenvalues of independent 3x3 matrices with bounded CUDA workspace."""

    if matrices.shape[-2:] != (3, 3):
        raise ValueError("chunked_eigvalsh_3x3 expects [...,3,3]")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    flat = matrices.reshape(-1, 3, 3)
    chunks = [
        torch.linalg.eigvalsh(flat[start : start + int(chunk_size)])
        for start in range(0, len(flat), int(chunk_size))
    ]
    return torch.cat(chunks, dim=0).reshape(*matrices.shape[:-2], 3)


def _angle(left: Tensor, right: Tensor, eps: float = 1e-8) -> Tensor:
    left = left / torch.linalg.vector_norm(left, dim=-1, keepdim=True).clamp_min(eps)
    right = right / torch.linalg.vector_norm(right, dim=-1, keepdim=True).clamp_min(eps)
    sine = torch.linalg.vector_norm(
        torch.linalg.cross(left, right, dim=-1), dim=-1
    ).clamp_min(eps)
    cosine = (left * right).sum(-1).clamp(-1.0, 1.0)
    return torch.atan2(sine, cosine)


@GEOMETRY_MODULES.register_module()
class PointPairFeatures(nn.Module):
    """Return ``[distance, angle(n_i,d), angle(n_j,d), angle(n_i,n_j)]``."""

    def forward(
        self,
        source_points: Tensor,
        target_points: Tensor,
        source_normals: Tensor,
        target_normals: Tensor,
    ) -> Tensor:
        delta = target_points - source_points
        distance = torch.linalg.vector_norm(delta, dim=-1)
        return torch.stack(
            (
                distance,
                _angle(source_normals, delta),
                _angle(target_normals, delta),
                _angle(source_normals, target_normals),
            ),
            dim=-1,
        )


@torch.no_grad()
def _estimate_unoriented_normals(
    points: Tensor,
    valid_mask: Tensor,
    num_neighbors: int,
    *,
    eigh_chunk_size: int = 512,
) -> Tensor:
    """Estimate local normals with a bounded batched-eigh workspace.

    CUDA's solver may reserve several GiB when all ``B*N`` covariance matrices
    are passed to one batched ``eigh`` call.  Chunking only the independent
    3x3 decompositions is mathematically identical and keeps scratch batch-10
    training below a predictable memory ceiling.
    """

    if int(eigh_chunk_size) < 1:
        raise ValueError("eigh_chunk_size must be positive")

    indices = knn_indices(points, points, valid_mask, num_neighbors + 1)[..., 1:]
    if indices.shape[-1] == 0:
        return torch.zeros_like(points)
    neighbors = batched_gather(points, indices)
    centered = neighbors - neighbors.mean(dim=-2, keepdim=True)
    covariance = centered.transpose(-1, -2) @ centered
    covariance = covariance / max(indices.shape[-1], 1)
    # The first eigenvector is the least-variance surface direction.  Its sign
    # is deliberately left unoriented; angular PPF values are folded below.
    flat_covariance = covariance.float().reshape(-1, 3, 3)
    normal_chunks = [
        torch.linalg.eigh(
            flat_covariance[start : start + int(eigh_chunk_size)]
        ).eigenvectors[..., 0]
        for start in range(0, len(flat_covariance), int(eigh_chunk_size))
    ]
    normals = torch.cat(normal_chunks, dim=0).reshape(*points.shape[:-1], 3)
    return normals.to(points) * valid_mask.unsqueeze(-1)


@GEOMETRY_MODULES.register_module()
class LocalPointPairFeatureEmbedding(nn.Module):
    """Invariant local PPF descriptor built through :class:`PointPairFeatures`.

    Dataset normals are used when available.  Otherwise local PCA supplies an
    unoriented normal and all angular channels are folded around ``pi / 2`` so
    an arbitrary normal sign cannot alter the descriptor.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_neighbors: int = 8,
        distance_scale_m: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_neighbors = int(num_neighbors)
        self.distance_scale_m = float(distance_scale_m)
        self.ppf = PointPairFeatures()
        self.projection = nn.Sequential(
            nn.Linear(8, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        points: Tensor,
        valid_mask: Tensor,
        normals: Tensor | None = None,
        precomputed_indices: Tensor | None = None,
    ) -> Tensor:
        if normals is None:
            normals = _estimate_unoriented_normals(
                points, valid_mask, self.num_neighbors
            )
        else:
            normals = normals * valid_mask.unsqueeze(-1)
        indices = (
            knn_indices(points, points, valid_mask, self.num_neighbors + 1)
            if precomputed_indices is None else precomputed_indices
        )[..., 1:]
        if indices.shape[-1] == 0:
            return points.new_zeros((*points.shape[:2], self.projection[-1].out_features))
        neighbors = batched_gather(points, indices)
        neighbor_normals = batched_gather(normals, indices)
        center_points = points.unsqueeze(-2).expand_as(neighbors)
        center_normals = normals.unsqueeze(-2).expand_as(neighbor_normals)
        pair_features = self.ppf(
            center_points, neighbors, center_normals, neighbor_normals
        )
        pair_features = pair_features.clone()
        pair_features[..., 0] /= max(self.distance_scale_m, 1e-8)
        # Normal direction is intrinsically ambiguous for an unoriented mesh.
        pair_features[..., 1:] = torch.minimum(
            pair_features[..., 1:], torch.pi - pair_features[..., 1:]
        )
        statistics = torch.cat(
            (pair_features.mean(dim=-2), pair_features.amax(dim=-2)), dim=-1
        )
        return self.projection(statistics) * valid_mask.unsqueeze(-1)


__all__ = ["PointPairFeatures", "LocalPointPairFeatureEmbedding"]
