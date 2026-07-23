"""Match the view-to-view pose response structure of ground truth."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.models.pose.rotation import rotation_geodesic_distance
from symm_template_reg.registry import LOSSES


@LOSSES.register_module()
class PairwisePoseResponseLoss(nn.Module):
    def __init__(
        self, rotation_weight: float = 0.25, translation_weight: float = 0.25
    ) -> None:
        super().__init__()
        self.rotation_weight = float(rotation_weight)
        self.translation_weight = float(translation_weight)

    def forward(self, predicted_pose: Tensor, gt_pose: Tensor) -> dict[str, Tensor]:
        if predicted_pose.shape != gt_pose.shape or predicted_pose.ndim != 3:
            raise ValueError("predicted_pose and gt_pose must be equal [B,4,4] tensors")
        pred_rotation_distances = []
        gt_rotation_distances = []
        pred_translation_distances = []
        gt_translation_distances = []
        for left in range(len(predicted_pose)):
            for right in range(left + 1, len(predicted_pose)):
                pred_rotation_distances.append(
                    rotation_geodesic_distance(
                        predicted_pose[left, :3, :3], predicted_pose[right, :3, :3]
                    )
                )
                gt_rotation_distances.append(
                    rotation_geodesic_distance(
                        gt_pose[left, :3, :3], gt_pose[right, :3, :3]
                    )
                )
                pred_translation_distances.append(
                    torch.linalg.vector_norm(
                        predicted_pose[left, :3, 3] - predicted_pose[right, :3, 3]
                    )
                )
                gt_translation_distances.append(
                    torch.linalg.vector_norm(
                        gt_pose[left, :3, 3] - gt_pose[right, :3, 3]
                    )
                )
        if not pred_rotation_distances:
            zero = predicted_pose.sum() * 0.0
            rotation_loss = translation_loss = zero
            rotation_ratio = translation_ratio = zero
        else:
            pred_r = torch.stack(pred_rotation_distances)
            gt_r = torch.stack(gt_rotation_distances)
            pred_t = torch.stack(pred_translation_distances)
            gt_t = torch.stack(gt_translation_distances)
            rotation_loss = torch.nn.functional.smooth_l1_loss(pred_r, gt_r)
            translation_loss = torch.nn.functional.smooth_l1_loss(pred_t, gt_t)
            rotation_ratio = pred_r.mean() / gt_r.mean().clamp_min(1e-8)
            translation_ratio = pred_t.mean() / gt_t.mean().clamp_min(1e-8)
        return {
            "pairwise_pose_response_loss": (
                self.rotation_weight * rotation_loss
                + self.translation_weight * translation_loss
            ),
            "pairwise_rotation_response_loss": rotation_loss,
            "pairwise_translation_response_loss": translation_loss,
            "rotation_response_ratio": rotation_ratio,
            "translation_response_ratio": translation_ratio,
        }


__all__ = ["PairwisePoseResponseLoss"]
