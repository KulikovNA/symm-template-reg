"""Triangle shortlists selected by exact distance from auxiliary coordinates."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from symm_template_reg.geometry.triangle_surface import nearest_triangles_on_mesh
from symm_template_reg.models.geometry.triangle_targets import (
    point_triangle_distance_matrix,
)
from symm_template_reg.registry import HEADS


def _item(value: Tensor | Sequence[Tensor], index: int) -> Tensor:
    return value[index]


@HEADS.register_module()
class AuxGuidedTriangleCandidateBuilder(nn.Module):
    """Build deterministic candidates using exact point-to-triangle distance.

    ``aux_guided_*`` modes deliberately do not use centroid distance.  Patch
    modes require the face-owner map produced by the template patch partition.
    """

    MODES = {
        "global_exact", "predicted_patch_union", "aux_guided_global_topk",
        "aux_guided_patch_union_topk",
    }

    def __init__(
        self,
        mode: str = "aux_guided_global_topk",
        candidate_k: int = 32,
        projection_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported candidate mode: {mode}")
        if int(candidate_k) < 1:
            raise ValueError("candidate_k must be positive")
        self.mode = mode
        self.candidate_k = int(candidate_k)
        self.projection_chunk_size = int(projection_chunk_size)

    def forward(
        self,
        q_aux_O: Tensor,
        template_vertices_O: Tensor | Sequence[Tensor],
        template_faces: Tensor | Sequence[Tensor],
        valid_mask: Tensor,
        predicted_topk_patches: Tensor | None = None,
        face_owner_patch_ids: Tensor | Sequence[Tensor] | None = None,
    ) -> dict[str, Tensor | str]:
        if q_aux_O.ndim != 3 or q_aux_O.shape[-1] != 3:
            raise ValueError("q_aux_O must have shape [B,N,3]")
        if valid_mask.shape != q_aux_O.shape[:2]:
            raise ValueError("valid_mask must have shape [B,N]")
        if "patch" in self.mode and (
            predicted_topk_patches is None or face_owner_patch_ids is None
        ):
            raise ValueError("patch modes require predicted patches and face owners")
        id_batches: list[Tensor] = []
        mask_batches: list[Tensor] = []
        distance_batches: list[Tensor] = []
        max_width = 0
        for batch_index, query in enumerate(q_aux_O):
            vertices = _item(template_vertices_O, batch_index).to(query)
            faces = _item(template_faces, batch_index).to(
                device=query.device, dtype=torch.long
            )
            valid_ids = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
            if self.mode in {"global_exact", "aux_guided_global_topk"}:
                width = len(faces) if self.mode == "global_exact" else min(
                    self.candidate_k, len(faces)
                )
                nearest = nearest_triangles_on_mesh(
                    query[valid_ids].detach(), vertices, faces, width,
                    point_chunk_size=self.projection_chunk_size,
                )
                ids = query.new_full((len(query), width), -1, dtype=torch.long)
                distances = query.new_full((len(query), width), float("inf"))
                ids[valid_ids] = nearest["face_ids"]
                distances[valid_ids] = nearest["distances"].to(query)
                masks = ids.ge(0)
            else:
                owners = _item(face_owner_patch_ids, batch_index).to(
                    device=query.device, dtype=torch.long
                )
                patches = predicted_topk_patches[batch_index]
                union_mask = owners[None, None, :].eq(patches[:, :, None]).any(1)
                union_mask &= valid_mask[batch_index, :, None]
                face_ids = torch.arange(len(faces), device=query.device)[None].expand(len(query), -1)
                if self.mode == "predicted_patch_union":
                    width = int(union_mask.sum(-1).max())
                    ids = face_ids.masked_fill(~union_mask, len(faces)).sort(-1).values[:, :width]
                    masks = ids.lt(len(faces)); ids = ids.masked_fill(~masks, -1)
                    distances = query.new_zeros(ids.shape).masked_fill(~masks, float("inf"))
                else:
                    width = min(self.candidate_k, len(faces))
                    exact_distance = point_triangle_distance_matrix(
                        query.detach(), vertices, faces,
                        point_chunk_size=self.projection_chunk_size,
                    ).masked_fill(~union_mask, float("inf"))
                    distances, ids = exact_distance.topk(width, dim=-1, largest=False, sorted=True)
                    masks = torch.isfinite(distances)
                    ids = ids.masked_fill(~masks, -1)
            id_batches.append(ids); mask_batches.append(masks)
            distance_batches.append(distances); max_width = max(max_width, ids.shape[-1])
        for index in range(len(id_batches)):
            pad = max_width - id_batches[index].shape[-1]
            if pad:
                id_batches[index] = torch.nn.functional.pad(id_batches[index], (0, pad), value=-1)
                mask_batches[index] = torch.nn.functional.pad(mask_batches[index], (0, pad), value=False)
                distance_batches[index] = torch.nn.functional.pad(distance_batches[index], (0, pad), value=float("inf"))
        return {
            "candidate_triangle_ids": torch.stack(id_batches),
            "candidate_triangle_mask": torch.stack(mask_batches),
            "candidate_distances": torch.stack(distance_batches),
            "candidate_count": torch.stack(mask_batches).sum(-1),
            "candidate_mode": self.mode,
            "selection_uses_centroid_distance": False,
        }


__all__ = ["AuxGuidedTriangleCandidateBuilder"]
