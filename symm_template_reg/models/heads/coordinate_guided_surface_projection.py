"""Exact coordinate-guided projection onto a discrete triangle candidate set."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from symm_template_reg.models.geometry.triangle_targets import (
    closest_barycentric_on_triangles,
)
from symm_template_reg.registry import HEADS


def _batch_item(value: Tensor | Sequence[Tensor], index: int) -> Tensor:
    return value[index]


@HEADS.register_module()
class CoordinateGuidedSurfaceProjectionHead(nn.Module):
    """Hard nearest-triangle projection with analytic barycentric coordinates.

    The default hard argmin is intended for evaluation/inference.  A future
    differentiable candidate-selection mode is reserved explicitly but is not
    silently approximated by averaging points from different triangles.
    """

    def __init__(self, selection_mode: str = "hard_argmin") -> None:
        super().__init__()
        if selection_mode not in {"hard_argmin", "future_differentiable"}:
            raise ValueError(f"unsupported selection_mode: {selection_mode}")
        if selection_mode != "hard_argmin":
            raise NotImplementedError(
                "future_differentiable is reserved; hard_argmin remains the default"
            )
        self.selection_mode = selection_mode

    def forward(
        self,
        q_aux_O: Tensor,
        candidate_triangle_ids: Tensor,
        template_vertices_O: Tensor | Sequence[Tensor],
        template_faces: Tensor | Sequence[Tensor],
        valid_mask: Tensor,
        candidate_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if q_aux_O.ndim != 3 or q_aux_O.shape[-1] != 3:
            raise ValueError("q_aux_O must have shape [B,N,3]")
        if candidate_triangle_ids.ndim != 3:
            raise ValueError("candidate_triangle_ids must have shape [B,N,L]")
        if candidate_triangle_ids.shape[:2] != q_aux_O.shape[:2]:
            raise ValueError("q_aux and candidate point dimensions disagree")
        if valid_mask.shape != q_aux_O.shape[:2]:
            raise ValueError("valid_mask must have shape [B,N]")
        candidate_valid = (
            candidate_triangle_ids.ge(0)
            if candidate_mask is None
            else candidate_mask.bool() & candidate_triangle_ids.ge(0)
        )
        outputs, face_rows, bary_rows, distance_rows = [], [], [], []
        selected_local_rows, candidate_distance_rows = [], []
        rank_rows, rank_valid_rows, eigenvalue_rows = [], [], []
        for batch_index in range(len(q_aux_O)):
            vertices = _batch_item(template_vertices_O, batch_index).to(q_aux_O)
            faces = _batch_item(template_faces, batch_index).to(
                device=q_aux_O.device, dtype=torch.long
            )
            ids = candidate_triangle_ids[batch_index].clamp_min(0)
            triangles = vertices[faces[ids]]
            n, candidate_count = ids.shape
            repeated_q = q_aux_O[batch_index, :, None].expand(
                n, candidate_count, 3
            ).reshape(-1, 3)
            projected = closest_barycentric_on_triangles(
                repeated_q, triangles.reshape(-1, 3, 3)
            )
            candidate_points = projected["points"].reshape(n, candidate_count, 3)
            candidate_bary = projected["barycentric"].reshape(n, candidate_count, 3)
            candidate_distance = projected["distances"].reshape(n, candidate_count)
            candidate_distance = candidate_distance.masked_fill(
                ~candidate_valid[batch_index], float("inf")
            )
            if bool((valid_mask[batch_index] & ~candidate_valid[batch_index].any(-1)).any()):
                raise ValueError("a valid observed point has no valid triangle candidate")
            selected_local = candidate_distance.argmin(-1)
            row = torch.arange(n, device=q_aux_O.device)
            point = candidate_points[row, selected_local]
            barycentric = candidate_bary[row, selected_local]
            distance = candidate_distance[row, selected_local]
            face_id = ids[row, selected_local]
            point = point * valid_mask[batch_index, :, None]
            barycentric = barycentric * valid_mask[batch_index, :, None]
            distance = distance.masked_fill(~valid_mask[batch_index], 0.0)
            face_id = face_id.masked_fill(~valid_mask[batch_index], -1)
            valid_points = point[valid_mask[batch_index]].float()
            centered = valid_points - valid_points.mean(0, keepdim=True)
            covariance = centered.T @ centered / max(len(centered) - 1, 1)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
            rank = torch.linalg.matrix_rank(centered, tol=1e-7)
            outputs.append(point)
            face_rows.append(face_id)
            bary_rows.append(barycentric)
            distance_rows.append(distance)
            selected_local_rows.append(selected_local)
            candidate_distance_rows.append(candidate_distance)
            rank_rows.append(rank)
            rank_valid_rows.append(rank.ge(3))
            eigenvalue_rows.append(eigenvalues)
        return {
            "surface_correspondence_points_O": torch.stack(outputs),
            "selected_triangle_ids": torch.stack(face_rows),
            "analytic_barycentric_coordinates": torch.stack(bary_rows),
            "distance_to_selected_triangle": torch.stack(distance_rows),
            "selected_local_candidate_ids": torch.stack(selected_local_rows),
            "candidate_distances": torch.stack(candidate_distance_rows),
            "correspondence_rank": torch.stack(rank_rows),
            "rank_valid": torch.stack(rank_valid_rows),
            "covariance_eigenvalues": torch.stack(eigenvalue_rows),
        }


__all__ = ["CoordinateGuidedSurfaceProjectionHead"]
