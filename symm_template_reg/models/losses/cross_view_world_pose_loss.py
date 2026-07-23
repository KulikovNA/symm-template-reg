"""Symmetry-aware consistency of one fragment observed by multiple cameras."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from symm_template_reg.models.pose.metrics import symmetry_aware_pose_errors
from symm_template_reg.registry import LOSSES


@LOSSES.register_module()
class CrossViewWorldPoseLoss(nn.Module):
    def __init__(
        self,
        rotation_weight: float = 1.0,
        translation_weight: float = 10.0,
        reference_mode: str = "pairwise_medoid",
    ) -> None:
        super().__init__()
        if reference_mode not in {"pairwise_medoid", "all_pairs"}:
            raise ValueError("reference_mode must be pairwise_medoid or all_pairs")
        self.rotation_weight = float(rotation_weight)
        self.translation_weight = float(translation_weight)
        self.reference_mode = reference_mode

    def forward(
        self,
        predicted_T_C_from_O: Tensor,
        T_W_from_C: Tensor,
        symmetry_metadata: Sequence[object],
        effective_symmetry_groups: Sequence[object],
    ) -> dict[str, Tensor]:
        if predicted_T_C_from_O.shape != T_W_from_C.shape:
            raise ValueError("camera poses and extrinsics must both be [B,4,4]")
        world = T_W_from_C @ predicted_T_C_from_O
        count = len(world)
        if count < 2:
            zero = world.sum() * 0.0
            return {
                "cross_view_world_pose_loss": zero,
                "cross_view_world_rotation_loss": zero,
                "cross_view_world_translation_loss": zero,
            }
        pair_values: list[tuple[int, int, Tensor, Tensor]] = []
        costs = world.new_zeros((count, count))
        for left in range(count):
            for right in range(left + 1, count):
                errors = symmetry_aware_pose_errors(
                    world[left].unsqueeze(0),
                    world[right],
                    symmetry_metadata[left],
                    effective_group=effective_symmetry_groups[left],
                )
                rotation = errors["rotation_rad"][0]
                translation = errors["translation_m"][0]
                pair_values.append((left, right, rotation, translation))
                value = self.rotation_weight * rotation + self.translation_weight * translation
                costs[left, right] = value.detach()
                costs[right, left] = value.detach()
        if self.reference_mode == "pairwise_medoid":
            reference = int(torch.argmin(costs.sum(dim=1)))
            selected = [item for item in pair_values if reference in item[:2]]
        else:
            selected = pair_values
        rotation_loss = torch.stack([item[2] for item in selected]).mean()
        translation_loss = torch.stack([item[3] for item in selected]).mean()
        total = self.rotation_weight * rotation_loss + self.translation_weight * translation_loss
        return {
            "cross_view_world_pose_loss": total,
            "cross_view_world_rotation_loss": rotation_loss,
            "cross_view_world_translation_loss": translation_loss,
        }


__all__ = ["CrossViewWorldPoseLoss"]
