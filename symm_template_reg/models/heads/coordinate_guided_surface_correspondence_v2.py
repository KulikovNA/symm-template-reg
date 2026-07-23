"""Coordinate-guided surface correspondence without learned triangle/bary heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from symm_template_reg.models.geometry.aux_guided_triangle_candidates import (
    AuxGuidedTriangleCandidateBuilder,
)
from symm_template_reg.models.heads.coordinate_guided_surface_projection import (
    CoordinateGuidedSurfaceProjectionHead,
)
from symm_template_reg.models.pose.weighted_procrustes import WeightedProcrustes
from symm_template_reg.registry import HEADS


@HEADS.register_module()
class CoordinateGuidedSurfaceCorrespondenceV2(nn.Module):
    """Exact analytic surface path with explicit global fallback reporting."""

    def __init__(
        self,
        projection_mode: str = "exact_global",
        candidate_mode: str = "aux_guided_global_topk",
        candidate_k: int = 32,
        projection_chunk_size: int = 256,
        fallback_to_global_exact: bool = True,
    ) -> None:
        super().__init__()
        self.projection_mode = projection_mode
        self.candidate_mode = candidate_mode
        self.candidate_k = int(candidate_k)
        self.projection_chunk_size = int(projection_chunk_size)
        self.fallback_to_global_exact = bool(fallback_to_global_exact)
        # Exact global needs only the exactly selected nearest face, not a
        # materialized [N, all_faces] candidate tensor.
        mode = "aux_guided_global_topk" if projection_mode == "exact_global" else candidate_mode
        self.builder = AuxGuidedTriangleCandidateBuilder(
            mode=mode, candidate_k=1 if projection_mode == "exact_global" else candidate_k,
            projection_chunk_size=projection_chunk_size,
        )
        self.projector = CoordinateGuidedSurfaceProjectionHead()
        self.procrustes = WeightedProcrustes()

    def forward(
        self,
        q_aux_O: Tensor,
        observed_points_C: Tensor,
        template_vertices_O: Tensor | Sequence[Tensor],
        template_faces: Tensor | Sequence[Tensor],
        valid_mask: Tensor,
        predicted_topk_patches: Tensor | None = None,
        face_owner_patch_ids: Tensor | Sequence[Tensor] | None = None,
        shortlist_pass_mask: Tensor | None = None,
    ) -> dict[str, Tensor | str | bool]:
        built = self.builder(
            q_aux_O, template_vertices_O, template_faces, valid_mask,
            predicted_topk_patches, face_owner_patch_ids,
        )
        fallback = torch.zeros_like(valid_mask)
        ids = built["candidate_triangle_ids"]
        candidate_mask = built["candidate_triangle_mask"]
        projected = self.projector(
            q_aux_O, ids, template_vertices_O, template_faces, valid_mask,
            candidate_mask,
        )
        if shortlist_pass_mask is not None:
            fallback = valid_mask & ~shortlist_pass_mask.bool()
            if bool(fallback.any()) and not self.fallback_to_global_exact:
                raise ValueError("shortlist failed and global fallback is disabled")
            if bool(fallback.any()):
                global_builder = AuxGuidedTriangleCandidateBuilder(
                    mode="aux_guided_global_topk", candidate_k=1,
                    projection_chunk_size=self.projection_chunk_size,
                )
                global_candidates = global_builder(
                    q_aux_O, template_vertices_O, template_faces, valid_mask
                )
                global_projected = self.projector(
                    q_aux_O, global_candidates["candidate_triangle_ids"],
                    template_vertices_O, template_faces, valid_mask,
                    global_candidates["candidate_triangle_mask"],
                )
                for key in (
                    "surface_correspondence_points_O", "selected_triangle_ids",
                    "analytic_barycentric_coordinates",
                    "distance_to_selected_triangle",
                ):
                    selector = fallback
                    while selector.ndim < projected[key].ndim:
                        selector = selector.unsqueeze(-1)
                    projected[key] = torch.where(
                        selector, global_projected[key], projected[key]
                    )
        solution = self.procrustes.solve(
            projected["surface_correspondence_points_O"], observed_points_C,
            valid_mask.to(q_aux_O.dtype), valid_mask,
        )
        counts = candidate_mask.sum(-1)
        return {
            **projected,
            "T_C_from_O": solution["transform"],
            "procrustes_rank": solution["rank"],
            "procrustes_rank_valid": solution["rank_valid"],
            "candidate_count": counts,
            "shortlist_fallback_mask": fallback,
            "shortlist_fallback_fraction": (
                fallback.float().sum() / valid_mask.float().sum().clamp_min(1)
            ),
            "global_projection_fraction": (
                fallback.float().sum() / valid_mask.float().sum().clamp_min(1)
                if self.projection_mode != "exact_global" else q_aux_O.new_tensor(1.0)
            ),
            "projection_mode": self.projection_mode,
            "candidate_mode": self.candidate_mode,
            "learned_barycentric_head_used": False,
            "learned_triangle_head_used": False,
        }


__all__ = ["CoordinateGuidedSurfaceCorrespondenceV2"]
