"""Direct translation/rotation and clean-room set-prediction losses.

Architectural reference: DETR (https://github.com/facebookresearch/detr), commit
29901c51d7fe8712168b8d0d64351170bc0f83e0, paths ``models/detr.py`` and
``models/matcher.py`` (Apache-2.0). No source text was copied. Changes: SE(3)
costs replace box costs, binary pose/no-pose targets, small-K local assignment,
and auxiliary pose-decoder losses.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.models.matching.hungarian_assigner import HungarianPoseAssigner
from symm_template_reg.registry import LOSSES


def rotation_geodesic_distance(left: Tensor, right: Tensor) -> Tensor:
    relative = torch.matmul(left.transpose(-2, -1), right)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    # A small interior clamp prevents an infinite acos derivative at an exact identity.
    return torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))


@LOSSES.register_module()
class DirectTranslationLoss(nn.Module):
    def __init__(self, p: int = 1) -> None:
        super().__init__()
        self.p = p

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        return torch.linalg.vector_norm(predicted - target, ord=self.p, dim=-1).mean()


@LOSSES.register_module()
class DirectRotationLoss(nn.Module):
    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        return rotation_geodesic_distance(predicted, target).mean()


@LOSSES.register_module()
class PoseSetLoss(nn.Module):
    """One-to-one matching of direct pose queries to equivalent GT hypotheses."""

    def __init__(
        self,
        translation_weight: float = 10.0,
        rotation_weight: float = 1.0,
        classification_weight: float = 1.0,
        auxiliary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.classification_weight = classification_weight
        self.auxiliary_weight = auxiliary_weight
        self.assigner = HungarianPoseAssigner()

    def _single(
        self,
        predicted_poses: Tensor,
        pose_logits: Tensor,
        target_poses: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if target_poses.shape[0] == 0:
            zero = predicted_poses.sum() * 0.0
            classification = torch.nn.functional.binary_cross_entropy_with_logits(
                pose_logits, torch.zeros_like(pose_logits)
            )
            return self.classification_weight * classification, {
                "translation": zero,
                "rotation": zero,
                "classification": classification,
            }
        translation_cost = torch.cdist(predicted_poses[:, :3, 3], target_poses[:, :3, 3], p=1)
        rotation_cost = rotation_geodesic_distance(
            predicted_poses[:, None, :3, :3], target_poses[None, :, :3, :3]
        )
        cost = self.translation_weight * translation_cost + self.rotation_weight * rotation_cost
        pred_index, target_index = self.assigner(cost)
        matched_translation = translation_cost[pred_index, target_index].mean()
        matched_rotation = rotation_cost[pred_index, target_index].mean()
        class_target = torch.zeros_like(pose_logits)
        class_target[pred_index] = 1.0
        classification = torch.nn.functional.binary_cross_entropy_with_logits(pose_logits, class_target)
        total = (
            self.translation_weight * matched_translation
            + self.rotation_weight * matched_rotation
            + self.classification_weight * classification
        )
        return total, {
            "translation": matched_translation,
            "rotation": matched_rotation,
            "classification": classification,
        }

    def _single_symmetry_aware(
        self,
        predicted_poses: Tensor,
        pose_logits: Tensor,
        gt_pose: Tensor,
        symmetry_metadata: object,
        effective_group: object,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        from symm_template_reg.models.pose.metrics import symmetry_aware_pose_errors

        errors = symmetry_aware_pose_errors(
            predicted_poses,
            gt_pose,
            symmetry_metadata,
            effective_group=effective_group,
            translation_weight=self.translation_weight / max(self.rotation_weight, 1e-12),
        )
        translation_cost = errors["translation_m"]
        rotation_cost = errors["rotation_rad"]
        combined = (
            self.translation_weight * translation_cost
            + self.rotation_weight * rotation_cost
        )
        assigned = torch.argmin(combined)
        classification_target = torch.zeros_like(pose_logits)
        classification_target[assigned] = 1.0
        classification = torch.nn.functional.binary_cross_entropy_with_logits(
            pose_logits, classification_target
        )
        translation = translation_cost[assigned]
        rotation = rotation_cost[assigned]
        total = (
            self.translation_weight * translation
            + self.rotation_weight * rotation
            + self.classification_weight * classification
        )
        return total, {
            "translation": translation,
            "rotation": rotation,
            "classification": classification,
        }, assigned

    def forward(
        self,
        pose_hypotheses: Tensor,
        pose_logits: Tensor,
        target_hypotheses: Tensor | list[Tensor],
        auxiliary_outputs: list[dict[str, Tensor]] | None = None,
        *,
        symmetry_metadata: list[object] | None = None,
        effective_symmetry_groups: list[object] | None = None,
    ) -> dict[str, Tensor]:
        targets = (
            [target_hypotheses[b] for b in range(target_hypotheses.shape[0])]
            if isinstance(target_hypotheses, Tensor)
            else target_hypotheses
        )
        if (symmetry_metadata is None) != (effective_symmetry_groups is None):
            raise ValueError(
                "symmetry_metadata and effective_symmetry_groups must be provided together"
            )
        totals, translation, rotation, classification, assignments = [], [], [], [], []
        for batch_index, target in enumerate(targets):
            if target.numel() == 0:
                # An empty target means that this item has no pose supervision.
                # Keep the historical no-pose classification behaviour and do
                # not try to manufacture a base pose for symmetry matching.
                total, components = self._single(
                    pose_hypotheses[batch_index], pose_logits[batch_index], target
                )
                assigned = torch.full(
                    (), -1, dtype=torch.long, device=pose_hypotheses.device
                )
            elif (
                symmetry_metadata is not None
                and effective_symmetry_groups is not None
                and symmetry_metadata[batch_index] is not None
            ):
                base_target = target[0] if target.ndim == 3 else target
                total, components, assigned = self._single_symmetry_aware(
                    pose_hypotheses[batch_index],
                    pose_logits[batch_index],
                    base_target,
                    symmetry_metadata[batch_index],
                    effective_symmetry_groups[batch_index],
                )
            else:
                if target.ndim == 2:
                    target = target.unsqueeze(0)
                total, components = self._single(
                    pose_hypotheses[batch_index], pose_logits[batch_index], target
                )
                assigned = torch.argmin(
                    torch.linalg.vector_norm(
                        pose_hypotheses[batch_index, :, :3, 3]
                        - target[0, :3, 3],
                        dim=-1,
                    )
                )
            totals.append(total)
            translation.append(components["translation"])
            rotation.append(components["rotation"])
            classification.append(components["classification"])
            assignments.append(assigned)
        total = torch.stack(totals).mean()
        auxiliary = total.new_zeros(())
        if auxiliary_outputs:
            for output in auxiliary_outputs:
                result = self.forward(
                    output["pose_hypotheses"],
                    output["pose_logits"],
                    targets,
                    None,
                    symmetry_metadata=symmetry_metadata,
                    effective_symmetry_groups=effective_symmetry_groups,
                )
                auxiliary = auxiliary + result["loss_pose_set"]
            auxiliary = auxiliary / len(auxiliary_outputs)
            total = total + self.auxiliary_weight * auxiliary
        return {
            "loss_pose_set": total,
            "loss_translation": torch.stack(translation).mean(),
            "loss_rotation": torch.stack(rotation).mean(),
            "loss_pose_classification": torch.stack(classification).mean(),
            "loss_pose_auxiliary": auxiliary,
            "assigned_query_indices": torch.stack(assignments),
        }
