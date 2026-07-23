"""Diagnostic direct canonical-coordinate regression without matching."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from symm_template_reg.registry import HEADS


@HEADS.register_module()
class CanonicalCoordinateRegressionControl(nn.Module):
    requires_template_mesh = True
    is_canonical_coordinate_control = True

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)
        )

    def forward(
        self,
        observed_features: Tensor,
        template_features: Tensor,
        template_points: Tensor,
        observed_mask: Tensor,
        template_mask: Tensor,
        *,
        template_mesh_vertices_O: Sequence[Tensor],
        template_mesh_faces: Sequence[Tensor],
        teacher_forcing_target_points_O: Tensor | None = None,
    ):
        del template_features, template_points, template_mask, template_mesh_faces, teacher_forcing_target_points_O
        unit = torch.sigmoid(self.regressor(observed_features))
        bounds_min = torch.stack([value.to(unit).amin(0) for value in template_mesh_vertices_O])
        bounds_max = torch.stack([value.to(unit).amax(0) for value in template_mesh_vertices_O])
        points = bounds_min[:, None] + unit * (bounds_max - bounds_min)[:, None]
        points = points * observed_mask[..., None]
        confidence = observed_mask.to(points.dtype)
        return {
            "points_O": points,
            "confidence": confidence,
            "logits": points.new_zeros((*points.shape[:2], 1)),
            "auxiliary": {"bounded_unit_coordinates": unit},
        }


__all__ = ["CanonicalCoordinateRegressionControl"]
