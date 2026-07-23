"""Losses for sample-conditioned base pose and camera-frame residual modes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from symm_template_reg.models.pose.metrics import symmetry_aware_pose_errors
from symm_template_reg.registry import LOSSES


def _pose_costs(
    poses: Tensor,
    gt: Tensor,
    metadata: Sequence[Any],
    groups: Sequence[Any],
    translation_weight: float,
    rotation_weight: float,
) -> Tensor:
    rows = []
    for index in range(len(poses)):
        errors = symmetry_aware_pose_errors(
            poses[index], gt[index], metadata[index], effective_group=groups[index]
        )
        rows.append(
            translation_weight * errors["translation_m"]
            + rotation_weight * errors["rotation_rad"]
        )
    return torch.stack(rows)


@LOSSES.register_module()
class ConditionedMultiHypothesisPoseLoss(nn.Module):
    def __init__(
        self,
        base_pose_weight: float = 1.0,
        best_residual_pose_weight: float = 1.0,
        residual_regularization_weight: float = 0.01,
        translation_weight: float = 10.0,
        rotation_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_pose_weight = float(base_pose_weight)
        self.best_residual_pose_weight = float(best_residual_pose_weight)
        self.residual_regularization_weight = float(
            residual_regularization_weight
        )
        self.translation_weight = float(translation_weight)
        self.rotation_weight = float(rotation_weight)

    def forward(
        self,
        base_pose: Tensor,
        pose_hypotheses: Tensor,
        gt_pose: Tensor,
        symmetry_metadata: Sequence[Any],
        effective_symmetry_groups: Sequence[Any],
        residual_pose_parameters: Tensor | None = None,
    ) -> dict[str, Tensor]:
        base_costs = _pose_costs(
            base_pose.unsqueeze(1),
            gt_pose,
            symmetry_metadata,
            effective_symmetry_groups,
            self.translation_weight,
            self.rotation_weight,
        ).squeeze(1)
        hypothesis_costs = _pose_costs(
            pose_hypotheses,
            gt_pose,
            symmetry_metadata,
            effective_symmetry_groups,
            self.translation_weight,
            self.rotation_weight,
        )
        best = hypothesis_costs.min(dim=1).values
        if residual_pose_parameters is None:
            regularization = base_costs.new_zeros(())
        else:
            rotation_6d = residual_pose_parameters[..., :6]
            identity_6d = rotation_6d.new_tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            )
            regularization = (
                (rotation_6d - identity_6d).square().mean()
                + residual_pose_parameters[..., 6:9].square().mean()
            )
        base_loss = base_costs.mean()
        best_loss = best.mean()
        total = (
            self.base_pose_weight * base_loss
            + self.best_residual_pose_weight * best_loss
            + self.residual_regularization_weight * regularization
        )
        return {
            "loss_conditioned_pose": total,
            "loss_base_pose": base_loss,
            "loss_best_residual_pose": best_loss,
            "loss_residual_regularization": regularization,
            "conditioned_best_query_indices": hypothesis_costs.argmin(dim=1),
        }


@LOSSES.register_module()
class DirectCorrespondencePoseConsistencyLoss(nn.Module):
    def __init__(self, translation_weight: float = 10.0) -> None:
        super().__init__()
        self.translation_weight = float(translation_weight)

    def forward(self, direct_pose: Tensor, correspondence_pose: Tensor) -> Tensor:
        relative = direct_pose[..., :3, :3].transpose(-2, -1) @ correspondence_pose[..., :3, :3]
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
        rotation = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        translation = torch.linalg.vector_norm(
            direct_pose[..., :3, 3] - correspondence_pose[..., :3, 3], dim=-1
        )
        return (rotation + self.translation_weight * translation).mean()


__all__ = [
    "ConditionedMultiHypothesisPoseLoss",
    "DirectCorrespondencePoseConsistencyLoss",
]
