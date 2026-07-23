"""Symmetry-aware minimum pose distance, including an SO(2) axis metric."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import LOSSES

from .pose_set_loss import rotation_geodesic_distance


@LOSSES.register_module()
class SymmetryPoseLoss(nn.Module):
    def __init__(self, translation_weight: float = 10.0, rotation_weight: float = 1.0) -> None:
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight

    def forward(
        self,
        predicted_pose: Tensor,
        gt_pose: Tensor,
        equivalent_gt_poses: Tensor | None = None,
        *,
        continuous_axis_O: Tensor | None = None,
        continuous_origin_O: Tensor | None = None,
        symmetry_targets: object | None = None,
        fragment_points_O: Tensor | None = None,
        symmetry_metadata: object | None = None,
        fragment_faces: Tensor | None = None,
        symmetry_target_kwargs: dict[str, object] | None = None,
    ) -> Tensor:
        builder_arguments_present = any(
            value is not None
            for value in (fragment_points_O, symmetry_metadata, fragment_faces)
        )
        if symmetry_targets is not None and builder_arguments_present:
            raise ValueError(
                "symmetry_targets cannot be combined with fragment target-builder arguments"
            )
        if symmetry_targets is None and builder_arguments_present:
            if fragment_points_O is None or symmetry_metadata is None:
                raise ValueError(
                    "fragment_points_O and symmetry_metadata are both required to build targets"
                )
            from symm_template_reg.models.symmetry.targets import (
                build_fragment_symmetry_targets,
            )

            symmetry_targets = build_fragment_symmetry_targets(
                fragment_points_O,
                symmetry_metadata,  # type: ignore[arg-type]
                fragment_faces=fragment_faces,
                base_pose=gt_pose,
                **(symmetry_target_kwargs or {}),
            )
        if symmetry_targets is not None:
            if equivalent_gt_poses is not None or continuous_axis_O is not None:
                raise ValueError(
                    "symmetry_targets cannot be combined with explicit symmetry arguments"
                )
            pose_set = getattr(symmetry_targets, "equivalent_pose_set", None)
            if pose_set is None:
                raise TypeError("symmetry_targets must expose equivalent_pose_set")
            if bool(pose_set.is_continuous):
                continuous_axis_O = pose_set.axis
                continuous_origin_O = pose_set.origin
            else:
                equivalent_gt_poses = pose_set.poses
        has_query_axis = predicted_pose.ndim == gt_pose.ndim + 1
        if not has_query_axis and predicted_pose.ndim != gt_pose.ndim:
            raise ValueError(
                "predicted_pose and gt_pose must have matching ranks, or "
                "predicted_pose may contain one additional query axis"
            )
        if continuous_axis_O is not None:
            target_pose = gt_pose.unsqueeze(-3) if has_query_axis else gt_pose
            axis = continuous_axis_O.to(gt_pose)
            if axis.shape != (3,):
                raise ValueError("continuous_axis_O must have shape [3]")
            axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
            predicted_axis = torch.matmul(predicted_pose[..., :3, :3], axis)
            target_axis = torch.matmul(target_pose[..., :3, :3], axis)
            dot = torch.sum(predicted_axis * target_axis, dim=-1).clamp(-1.0, 1.0)
            cross_norm = torch.linalg.vector_norm(
                torch.linalg.cross(predicted_axis, target_axis, dim=-1), dim=-1
            ).clamp_min(1e-7)
            rotation = torch.atan2(cross_norm, dot)
            origin = (
                torch.zeros(3, dtype=gt_pose.dtype, device=gt_pose.device)
                if continuous_origin_O is None
                else continuous_origin_O.to(gt_pose)
            )
            if origin.shape != (3,):
                raise ValueError("continuous_origin_O must have shape [3]")
            predicted_origin = torch.matmul(
                predicted_pose[..., :3, :3],
                origin,
            ) + predicted_pose[..., :3, 3]
            target_origin = torch.matmul(
                target_pose[..., :3, :3],
                origin,
            ) + target_pose[..., :3, 3]
            translation = torch.linalg.vector_norm(predicted_origin - target_origin, dim=-1)
            return (self.translation_weight * translation + self.rotation_weight * rotation).mean()
        targets = gt_pose.unsqueeze(-3) if equivalent_gt_poses is None else equivalent_gt_poses
        if has_query_axis:
            targets = targets.unsqueeze(-4)
        predicted = predicted_pose.unsqueeze(-3)
        translation = torch.linalg.vector_norm(
            predicted[..., :3, 3] - targets[..., :3, 3], dim=-1
        )
        rotation = rotation_geodesic_distance(predicted[..., :3, :3], targets[..., :3, :3])
        return (self.translation_weight * translation + self.rotation_weight * rotation).min(-1).values.mean()
