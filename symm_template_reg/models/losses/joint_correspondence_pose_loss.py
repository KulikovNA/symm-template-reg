"""Joint physical loss with one shared symmetry element for every component."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from symm_template_reg.models.pose.pose_representation import (
    invert_transform,
    transform_points,
)
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance
from symm_template_reg.models.symmetry.hypothesis_expander import (
    symmetry_transforms,
)
from symm_template_reg.models.symmetry.groups import parse_rotation_group
from symm_template_reg.registry import LOSSES


def template_surface_distances(
    predicted_points_O: Tensor,
    template_surface_points_O: Tensor,
    template_valid_mask: Tensor,
    *,
    chunk_size: int = 512,
) -> Tensor:
    """Nearest distance to actual template surface samples, with bounded memory."""

    chunks = []
    for start in range(0, predicted_points_O.shape[1], chunk_size):
        distance = torch.cdist(
            predicted_points_O[:, start : start + chunk_size].float(),
            template_surface_points_O.float(),
        )
        distance = distance.masked_fill(~template_valid_mask[:, None, :], float("inf"))
        chunks.append(distance.amin(dim=-1).to(predicted_points_O))
    return torch.cat(chunks, dim=1)


def _smooth_normalized(distance: Tensor, scale: float) -> Tensor:
    normalized = distance / max(float(scale), 1e-12)
    return F.smooth_l1_loss(normalized, torch.zeros_like(normalized), reduction="none")


def _pad_rows(rows: list[Tensor], fill: float = float("nan")) -> Tensor:
    width = max(len(row) for row in rows)
    result = rows[0].new_full((len(rows), width), fill)
    for index, row in enumerate(rows):
        result[index, : len(row)] = row
    return result


@LOSSES.register_module()
class JointCorrespondencePoseLoss(nn.Module):
    """Choose one symmetry element from a joint correspondence/pose objective."""

    def __init__(
        self,
        correspondence_scale_m: float = 0.002,
        rotation_scale_deg: float = 2.0,
        translation_scale_m: float = 0.002,
        alignment_scale_m: float = 0.002,
        template_surface_scale_m: float = 0.001,
        lambda_corr: float = 1.0,
        lambda_rot: float = 1.0,
        lambda_trans: float = 1.0,
        lambda_align: float = 0.5,
        lambda_surface: float = 0.25,
        so2_samples: int = 36,
    ) -> None:
        super().__init__()
        self.correspondence_scale_m = float(correspondence_scale_m)
        self.rotation_scale_rad = math.radians(float(rotation_scale_deg))
        self.translation_scale_m = float(translation_scale_m)
        self.alignment_scale_m = float(alignment_scale_m)
        self.template_surface_scale_m = float(template_surface_scale_m)
        self.weights = {
            "correspondence": float(lambda_corr),
            "rotation": float(lambda_rot),
            "translation": float(lambda_trans),
            "alignment": float(lambda_align),
            "surface": float(lambda_surface),
        }
        if self.weights["rotation"] <= 0 or self.weights["alignment"] <= 0:
            raise ValueError("joint baseline requires non-zero pose and alignment weights")
        self.so2_samples = int(so2_samples)

    def forward(
        self,
        predicted_points_O: Tensor,
        predicted_pose_T_C_from_O: Tensor,
        gt_pose_T_C_from_O: Tensor,
        observed_points_C: Tensor,
        target_points_O: Tensor,
        valid_mask: Tensor,
        template_surface_points_O: Tensor,
        template_valid_mask: Tensor,
        symmetry_metadata: Sequence[Any],
        effective_symmetry_groups: Sequence[Any],
    ) -> dict[str, Any]:
        surface_distance = template_surface_distances(
            predicted_points_O, template_surface_points_O, template_valid_mask
        )
        reconstructed = torch.einsum(
            "bij,bnj->bni",
            predicted_pose_T_C_from_O[:, :3, :3],
            predicted_points_O,
        ) + predicted_pose_T_C_from_O[:, None, :3, 3]
        alignment_distance = torch.linalg.vector_norm(
            reconstructed - observed_points_C, dim=-1
        )
        selected_totals: list[Tensor] = []
        selected_corr: list[Tensor] = []
        selected_rot: list[Tensor] = []
        selected_trans: list[Tensor] = []
        selected_align: list[Tensor] = []
        selected_surface: list[Tensor] = []
        selected_corr_raw: list[Tensor] = []
        selected_rot_raw: list[Tensor] = []
        selected_trans_raw: list[Tensor] = []
        selected_align_raw: list[Tensor] = []
        selected_surface_raw: list[Tensor] = []
        selected_indices: list[int] = []
        matched_targets: list[Tensor] = []
        matched_poses: list[Tensor] = []
        totals_by_element: list[Tensor] = []
        corr_by_element: list[Tensor] = []
        rotation_by_element: list[Tensor] = []
        translation_by_element: list[Tensor] = []
        for index in range(len(predicted_points_O)):
            mask = valid_mask[index]
            if not bool(mask.any()):
                raise ValueError("joint loss requires at least one valid observed point")
            metadata = symmetry_metadata[index]
            group = parse_rotation_group(effective_symmetry_groups[index])
            symmetries = symmetry_transforms(
                group,
                metadata.axis.direction,
                metadata.axis.origin,
                so2_num_samples=self.so2_samples if group.type == "SO2" else None,
                dtype=predicted_points_O.dtype,
                device=predicted_points_O.device,
            )
            # q^(S)=S^-1 q_GT and T_GT(S)=T_GT@S use the exact same ordered S.
            inverse = invert_transform(symmetries)
            equivalent_targets = transform_points(
                inverse, target_points_O[index].unsqueeze(0)
            )
            gt_pose = gt_pose_T_C_from_O[index]
            equivalent_poses = gt_pose.unsqueeze(0) @ symmetries
            corr_distance = torch.linalg.vector_norm(
                predicted_points_O[index].unsqueeze(0) - equivalent_targets,
                dim=-1,
            )
            corr_normalized = _smooth_normalized(
                corr_distance[:, mask], self.correspondence_scale_m
            ).mean(dim=-1)
            rotation_error = rotation_geodesic_distance(
                predicted_pose_T_C_from_O[index, :3, :3].unsqueeze(0),
                equivalent_poses[:, :3, :3],
            )
            rotation_normalized = _smooth_normalized(
                rotation_error, self.rotation_scale_rad
            )
            translation_error = torch.linalg.vector_norm(
                predicted_pose_T_C_from_O[index, :3, 3].unsqueeze(0)
                - equivalent_poses[:, :3, 3],
                dim=-1,
            )
            translation_normalized = _smooth_normalized(
                translation_error, self.translation_scale_m
            )
            align_raw = alignment_distance[index, mask].mean()
            surface_raw = surface_distance[index, mask].mean()
            align_normalized = _smooth_normalized(
                alignment_distance[index, mask], self.alignment_scale_m
            ).mean()
            surface_normalized = _smooth_normalized(
                surface_distance[index, mask], self.template_surface_scale_m
            ).mean()
            total = (
                self.weights["correspondence"] * corr_normalized
                + self.weights["rotation"] * rotation_normalized
                + self.weights["translation"] * translation_normalized
                + self.weights["alignment"] * align_normalized
                + self.weights["surface"] * surface_normalized
            )
            selected = int(total.argmin().detach())
            selected_indices.append(selected)
            selected_totals.append(total[selected])
            selected_corr.append(corr_normalized[selected])
            selected_rot.append(rotation_normalized[selected])
            selected_trans.append(translation_normalized[selected])
            selected_align.append(align_normalized)
            selected_surface.append(surface_normalized)
            selected_corr_raw.append(corr_distance[selected, mask].mean())
            selected_rot_raw.append(rotation_error[selected])
            selected_trans_raw.append(translation_error[selected])
            selected_align_raw.append(align_raw)
            selected_surface_raw.append(surface_raw)
            matched_targets.append(equivalent_targets[selected])
            matched_poses.append(equivalent_poses[selected])
            totals_by_element.append(total)
            corr_by_element.append(corr_distance[:, mask].mean(dim=-1))
            rotation_by_element.append(torch.rad2deg(rotation_error))
            translation_by_element.append(translation_error * 1000.0)
        total_loss = torch.stack(selected_totals).mean()
        corr = torch.stack(selected_corr).mean()
        rot = torch.stack(selected_rot).mean()
        trans = torch.stack(selected_trans).mean()
        align = torch.stack(selected_align).mean()
        surface = torch.stack(selected_surface).mean()
        selected_tensor = torch.tensor(
            selected_indices, dtype=torch.long, device=predicted_points_O.device
        )
        return {
            "loss_total": total_loss,
            "loss_correspondence_normalized": corr,
            "loss_rotation_normalized": rot,
            "loss_translation_normalized": trans,
            "loss_alignment_normalized": align,
            "loss_template_surface_normalized": surface,
            "weighted_loss_correspondence": self.weights["correspondence"] * corr,
            "weighted_loss_rotation": self.weights["rotation"] * rot,
            "weighted_loss_translation": self.weights["translation"] * trans,
            "weighted_loss_alignment": self.weights["alignment"] * align,
            "weighted_loss_template_surface": self.weights["surface"] * surface,
            "raw_correspondence_error_m": torch.stack(selected_corr_raw).mean(),
            "raw_rotation_error_rad": torch.stack(selected_rot_raw).mean(),
            "raw_translation_error_m": torch.stack(selected_trans_raw).mean(),
            "raw_alignment_error_m": torch.stack(selected_align_raw).mean(),
            "raw_template_surface_error_m": torch.stack(selected_surface_raw).mean(),
            "rotation_error_deg": torch.rad2deg(torch.stack(selected_rot_raw)).mean(),
            "translation_total_mm": torch.stack(selected_trans_raw).mean() * 1000.0,
            "selected_shared_symmetry_element": selected_tensor,
            "selected_shared_symmetry_element_mean": selected_tensor.float().mean(),
            "loss_by_symmetry_element": _pad_rows(totals_by_element),
            "correspondence_error_by_symmetry_element_mm": _pad_rows(corr_by_element) * 1000.0,
            "rotation_error_by_symmetry_element_deg": _pad_rows(rotation_by_element),
            "translation_error_by_symmetry_element_mm": _pad_rows(translation_by_element),
            "matched_target_points_O": torch.stack(matched_targets),
            "matched_gt_pose_T_C_from_O": torch.stack(matched_poses),
            "alignment_distance_m": alignment_distance,
            "template_surface_distance_m": surface_distance,
            "reconstructed_visible_points_C": reconstructed,
        }

__all__ = ["JointCorrespondencePoseLoss", "template_surface_distances"]
