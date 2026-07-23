"""Reusable pairwise diagnostics for the rotation feature path."""

from __future__ import annotations

import torch
from torch import Tensor

from symm_template_reg.models.pose.rotation import rotation_geodesic_distance


def masked_token_summary(tokens: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def vector_pairwise_matrix(vectors: Tensor) -> Tensor:
    return torch.cdist(vectors.double(), vectors.double())


def rotation_pairwise_matrix(rotation: Tensor) -> Tensor:
    return torch.rad2deg(
        rotation_geodesic_distance(
            rotation[:, None].double(), rotation[None].double()
        )
    )


def centered_cloud_chamfer_matrix(points: Tensor, mask: Tensor) -> Tensor:
    count = len(points)
    matrix = points.new_zeros((count, count), dtype=torch.float64)
    clouds = []
    for index in range(count):
        cloud = points[index, mask[index]].double()
        clouds.append(cloud - cloud.mean(dim=0, keepdim=True))
    for left in range(count):
        for right in range(left + 1, count):
            distance = torch.cdist(clouds[left], clouds[right])
            chamfer = 0.5 * (distance.min(1).values.mean() + distance.min(0).values.mean())
            matrix[left, right] = matrix[right, left] = chamfer
    return matrix


__all__ = [
    "centered_cloud_chamfer_matrix",
    "masked_token_summary",
    "rotation_pairwise_matrix",
    "vector_pairwise_matrix",
]
