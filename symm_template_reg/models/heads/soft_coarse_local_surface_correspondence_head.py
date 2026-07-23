"""Legacy soft coarse coordinates refined onto a local template triangle."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from symm_template_reg.geometry import nearest_triangles_on_mesh
from symm_template_reg.registry import HEADS


@HEADS.register_module()
class SoftCoarseLocalSurfaceCorrespondenceHead(nn.Module):
    requires_template_mesh = True
    is_surface_constrained_v2 = True

    def __init__(
        self,
        embed_dim: int = 256,
        nearest_triangle_candidates: int = 32,
        coarse_temperature: float = 1.0,
        local_temperature: float = 1.0,
        max_coarse_to_surface_distance_m: float = 0.05,
    ) -> None:
        super().__init__()
        self.nearest_triangle_candidates = int(nearest_triangle_candidates)
        self.coarse_temperature = float(coarse_temperature)
        self.local_temperature = float(local_temperature)
        self.max_coarse_to_surface_distance_m = float(max_coarse_to_surface_distance_m)
        self.observed_query = nn.Linear(embed_dim, embed_dim)
        self.template_key = nn.Linear(embed_dim, embed_dim)
        self.local_query = nn.Linear(embed_dim, embed_dim)
        self.barycentric_head = nn.Linear(embed_dim, 3)

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
        del teacher_forcing_target_points_O
        logits = self.observed_query(observed_features) @ self.template_key(
            template_features
        ).transpose(-2, -1) / (
            math.sqrt(observed_features.shape[-1]) * self.coarse_temperature
        )
        logits = logits.masked_fill(~template_mask[:, None], float("-inf"))
        probability = torch.softmax(logits, -1)
        q_coarse = probability @ template_points
        outputs, fine_rows, triangle_rows, bary_rows, candidate_rows = [], [], [], [], []
        coarse_surface_rows = []
        for batch_index in range(len(observed_features)):
            vertices = template_mesh_vertices_O[batch_index].to(template_points)
            faces = template_mesh_faces[batch_index].to(
                device=template_points.device, dtype=torch.long
            )
            triangles = vertices[faces]
            nearest = nearest_triangles_on_mesh(
                q_coarse[batch_index].detach(),
                vertices,
                faces,
                self.nearest_triangle_candidates,
                point_chunk_size=256,
            )
            candidate_ids = nearest["face_ids"]
            candidates = triangles[candidate_ids]
            candidate_centroids = candidates.mean(2)
            nearest_anchor = torch.cdist(
                candidate_centroids.reshape(1, -1, 3).float(),
                template_points[batch_index : batch_index + 1].float(),
            ).argmin(-1).reshape(candidate_centroids.shape[:2])
            candidate_features = template_features[batch_index][nearest_anchor]
            fine_logits = (
                self.local_query(observed_features[batch_index])[:, None]
                * candidate_features
            ).sum(-1) / (
                math.sqrt(observed_features.shape[-1]) * self.local_temperature
            )
            selected = fine_logits.argmax(-1)
            rows = torch.arange(len(selected), device=selected.device)
            triangle = candidates[rows, selected]
            bary = torch.softmax(self.barycentric_head(observed_features[batch_index]), -1)
            q = (bary[..., None] * triangle).sum(1)
            outputs.append(q * observed_mask[batch_index, :, None])
            fine_rows.append(fine_logits)
            triangle_rows.append(candidate_ids[rows, selected])
            bary_rows.append(bary)
            candidate_rows.append(candidate_ids)
            coarse_surface_rows.append(
                nearest["distances"][:, 0]
            )
        points = torch.stack(outputs)
        fine_logits = torch.stack(fine_rows)
        return {
            "points_O": points,
            "confidence": torch.softmax(fine_logits, -1).amax(-1) * observed_mask,
            "logits": logits,
            "auxiliary": {
                "coarse_points_O": q_coarse,
                "coarse_patch_logits": logits,
                "fine_local_logits": fine_logits,
                "selected_patch_ids": logits.argmax(-1),
                "selected_topk_patch_ids": logits.topk(min(4, logits.shape[-1]), -1).indices,
                "selected_triangle_ids": torch.stack(triangle_rows),
                "predicted_barycentric": torch.stack(bary_rows),
                "candidate_triangle_ids": torch.stack(candidate_rows),
                "patch_points_O": template_points,
                "coarse_to_local_surface_distance_m": torch.stack(coarse_surface_rows),
                "max_coarse_to_surface_distance_m": points.new_tensor(
                    self.max_coarse_to_surface_distance_m
                ),
                "coarse_distance_limit_exceeded": torch.stack(coarse_surface_rows).gt(
                    self.max_coarse_to_surface_distance_m
                ),
            },
        }


__all__ = ["SoftCoarseLocalSurfaceCorrespondenceHead"]
