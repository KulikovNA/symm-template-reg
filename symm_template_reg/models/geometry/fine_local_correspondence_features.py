"""Dense, frame-safe features for fine local correspondence.

Observed geometry is built only from relations inside camera frame C.  Template
triangle geometry is built only inside object frame O; the module never forms a
raw C-minus-O coordinate difference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from symm_template_reg.models.geometry.point_ops import batched_gather, knn_indices
from symm_template_reg.models.geometry.ppf import (
    _estimate_unoriented_normals,
    chunked_eigvalsh_3x3,
)
from symm_template_reg.registry import GEOMETRY_MODULES


def _masked_feature_diagnostics(features: Tensor, valid_mask: Tensor) -> dict[str, Tensor]:
    rows = features[valid_mask]
    if len(rows) < 2:
        zero = features.new_zeros(())
        return {
            "variance": zero, "effective_rank": zero,
            "pairwise_distance": zero, "collision_fraction": zero,
        }
    centered = rows - rows.mean(0)
    variance = centered.var(0, unbiased=False).mean()
    singular = torch.linalg.svdvals(centered.float())
    effective_rank = singular.square().sum().square() / singular.pow(4).sum().clamp_min(1e-12)
    sample = rows[torch.linspace(0, len(rows) - 1, min(256, len(rows)), device=rows.device).long()]
    distance = torch.pdist(sample.float())
    pairwise = torch.cdist(sample.float(), sample.float())
    pairwise.fill_diagonal_(float("inf"))
    collisions = pairwise.amin(-1).le(1e-6).float().mean()
    return {
        "variance": variance,
        "effective_rank": effective_rank.to(features),
        "pairwise_distance": distance.mean().to(features) if distance.numel() else variance.new_zeros(()),
        "collision_fraction": collisions.to(features),
    }


@GEOMETRY_MODULES.register_module()
class FineLocalCorrespondenceFeatureAdapter(nn.Module):
    """Fuse identity-preserving dense features with invariant local geometry."""

    observed_geometry_dim = 30
    triangle_geometry_dim = 22

    def __init__(
        self,
        embed_dim: int = 256,
        knn_scales: Sequence[int] = (8, 16, 32),
        distance_scale_m: float = 0.01,
        triangle_neighbors: int = 8,
        observed_only: bool = False,
    ) -> None:
        super().__init__()
        if tuple(knn_scales) != (8, 16, 32):
            raise ValueError("fine adapter currently requires kNN scales (8,16,32)")
        self.embed_dim = int(embed_dim)
        self.knn_scales = tuple(map(int, knn_scales))
        self.distance_scale_m = float(distance_scale_m)
        self.triangle_neighbors = int(triangle_neighbors)
        self.observed_only = bool(observed_only)
        self.observed_projection = nn.Sequential(
            nn.Linear(2 * embed_dim + self.observed_geometry_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        self.observed_residual = nn.Linear(embed_dim, embed_dim)
        if not self.observed_only:
            self.triangle_projection = nn.Sequential(
                nn.Linear(2 * embed_dim + self.triangle_geometry_dim, 2 * embed_dim),
                nn.LayerNorm(2 * embed_dim),
                nn.GELU(),
                nn.Linear(2 * embed_dim, embed_dim),
            )

    def _observed_geometry(
        self, points_C: Tensor, valid_mask: Tensor, normals_C: Tensor | None,
        precomputed_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        normals = (
            _estimate_unoriented_normals(points_C, valid_mask, max(self.knn_scales))
            if normals_C is None
            else normals_C * valid_mask.unsqueeze(-1)
        )
        descriptors = []
        all_indices = (
            knn_indices(points_C, points_C, valid_mask, max(self.knn_scales) + 1)
            if precomputed_indices is None else precomputed_indices
        )[..., 1:]
        for k in self.knn_scales:
            indices = all_indices[..., : min(k, all_indices.shape[-1])]
            neighbors = batched_gather(points_C, indices)
            delta = neighbors - points_C.unsqueeze(-2)
            distance = torch.linalg.vector_norm(delta, dim=-1)
            scaled = distance / max(self.distance_scale_m, 1e-8)
            centered = delta - delta.mean(-2, keepdim=True)
            covariance = centered.transpose(-1, -2) @ centered / max(indices.shape[-1], 1)
            eigenvalues = chunked_eigvalsh_3x3(
                covariance.float(), chunk_size=512
            ).clamp_min(0)
            eigenvalues = eigenvalues / eigenvalues.sum(-1, keepdim=True).clamp_min(1e-12)
            unit_delta = delta / distance[..., None].clamp_min(1e-8)
            center_normal = normals.unsqueeze(-2)
            neighbor_normals = batched_gather(normals, indices)
            normal_direction = (center_normal * unit_delta).sum(-1).abs()
            normal_agreement = (center_normal * neighbor_normals).sum(-1).abs()
            descriptors.append(
                torch.cat(
                    (
                        scaled.mean(-1, keepdim=True),
                        scaled.std(-1, unbiased=False, keepdim=True),
                        scaled.amax(-1, keepdim=True),
                        scaled.amin(-1, keepdim=True),
                        eigenvalues.to(points_C),
                        normal_direction.mean(-1, keepdim=True),
                        normal_direction.amax(-1, keepdim=True),
                        normal_agreement.mean(-1, keepdim=True),
                    ),
                    dim=-1,
                )
            )
        geometry = torch.cat(descriptors, -1) * valid_mask.unsqueeze(-1)
        return geometry, normals

    def forward(
        self,
        original_dense_features: Tensor,
        interpolated_interaction_features: Tensor,
        points_C: Tensor,
        valid_mask: Tensor,
        normals_C: Tensor | None = None,
        precomputed_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        expected = original_dense_features.shape
        if interpolated_interaction_features.shape != expected:
            raise ValueError("dense encoder and interpolated interaction shapes disagree")
        if points_C.shape[:2] != expected[:2] or valid_mask.shape != expected[:2]:
            raise ValueError("fine adapter must preserve dense point identity")
        geometry, normals = self._observed_geometry(
            points_C, valid_mask, normals_C, precomputed_indices
        )
        fused = self.observed_projection(
            torch.cat((original_dense_features, interpolated_interaction_features, geometry), -1)
        ) + self.observed_residual(original_dense_features)
        fine = fused * valid_mask.unsqueeze(-1)
        diagnostics = _masked_feature_diagnostics(fine, valid_mask)
        return {
            "fine_point_features": fine,
            "observed_local_geometry": geometry,
            "estimated_or_input_normals_C": normals,
            "fine_feature_variance": diagnostics["variance"],
            "fine_feature_effective_rank": diagnostics["effective_rank"],
            "fine_feature_pairwise_distance": diagnostics["pairwise_distance"],
            "fine_feature_collision_fraction": diagnostics["collision_fraction"],
        }

    def template_triangle_features(
        self,
        vertices_O: Tensor,
        faces: Tensor,
        coarse_patch_features: Tensor,
        fine_template_features: Tensor,
        face_owner_patch_ids: Tensor,
        nearest_template_anchor_ids: Tensor,
        *,
        axis_direction_O: Tensor | Sequence[float],
        axis_origin_O: Tensor | Sequence[float],
    ) -> dict[str, Tensor]:
        if self.observed_only:
            raise RuntimeError("observed-only fine adapter has no triangle branch")
        triangles = vertices_O[faces.long()]
        centroids = triangles.mean(1)
        edges = torch.stack(
            (
                torch.linalg.vector_norm(triangles[:, 1] - triangles[:, 0], dim=-1),
                torch.linalg.vector_norm(triangles[:, 2] - triangles[:, 1], dim=-1),
                torch.linalg.vector_norm(triangles[:, 0] - triangles[:, 2], dim=-1),
            ), -1,
        )
        cross = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1
        )
        area = 0.5 * torch.linalg.vector_norm(cross, dim=-1, keepdim=True)
        normals = cross / torch.linalg.vector_norm(cross, dim=-1, keepdim=True).clamp_min(1e-12)
        centered_vertices = (triangles - centroids[:, None]).reshape(len(faces), 9)
        axis = torch.as_tensor(axis_direction_O, dtype=vertices_O.dtype, device=vertices_O.device)
        axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
        origin = torch.as_tensor(axis_origin_O, dtype=vertices_O.dtype, device=vertices_O.device)
        relative = centroids - origin
        axial = (relative * axis).sum(-1, keepdim=True)
        radial = torch.linalg.vector_norm(relative - axial * axis, dim=-1, keepdim=True)
        count = min(self.triangle_neighbors + 1, len(centroids))
        nearest = torch.cdist(centroids.float(), centroids.float()).topk(
            count, largest=False, sorted=True
        ).indices[:, 1:]
        if nearest.shape[-1]:
            neighbor_centroids = centroids[nearest]
            neighbor_distance = torch.linalg.vector_norm(
                neighbor_centroids - centroids[:, None], dim=-1
            ) / max(self.distance_scale_m, 1e-8)
            neighbor_normals = normals[nearest]
            normal_similarity = (neighbor_normals * normals[:, None]).sum(-1).abs()
            neighbor_area = area.squeeze(-1)[nearest] / area.clamp_min(1e-12)
            neighbor_descriptor = torch.stack(
                (
                    neighbor_distance.mean(-1),
                    neighbor_distance.amax(-1),
                    normal_similarity.mean(-1),
                    neighbor_area.mean(-1),
                ), -1,
            )
        else:
            neighbor_descriptor = vertices_O.new_zeros((len(faces), 4))
        geometry = torch.cat(
            (centered_vertices, normals, area, edges, axial, radial, neighbor_descriptor), -1
        )
        if geometry.shape[-1] != self.triangle_geometry_dim:
            raise AssertionError("triangle geometry descriptor width changed")
        coarse = coarse_patch_features[face_owner_patch_ids.long()]
        fine_template = fine_template_features[nearest_template_anchor_ids.long()]
        features = self.triangle_projection(torch.cat((coarse, fine_template, geometry), -1))
        return {
            "fine_triangle_features": features,
            "triangle_local_geometry": geometry,
            "triangle_centroids_O": centroids,
            "triangle_normals_O": normals,
            "triangle_areas_m2": area.squeeze(-1),
        }


__all__ = ["FineLocalCorrespondenceFeatureAdapter"]
