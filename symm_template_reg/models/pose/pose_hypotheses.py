"""Contracts and selection helpers for direct K-pose model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .pose_representation import make_transform
from .rotation import rotation_6d_to_matrix


def _validate_pose_tensors(poses: Tensor, logits: Tensor) -> tuple[tuple[int, ...], int]:
    if poses.ndim < 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"poses must have shape [...,K,4,4], got {tuple(poses.shape)}")
    if logits.shape != poses.shape[:-2]:
        raise ValueError(
            f"logits must have shape {tuple(poses.shape[:-2])}, got {tuple(logits.shape)}"
        )
    return tuple(poses.shape[:-3]), int(poses.shape[-3])


@dataclass
class PoseHypotheses:
    """Validated direct pose-query output.

    Shapes are ``pose_hypotheses=[...,K,4,4]``, ``pose_logits=[...,K]`` and
    optionally ``pose_uncertainty=[...,K,*U]``.  In normal model usage the
    leading shape is ``[B]``.
    """

    pose_hypotheses: Tensor
    pose_logits: Tensor
    pose_uncertainty: Optional[Tensor] = None

    def __post_init__(self) -> None:
        _validate_pose_tensors(self.pose_hypotheses, self.pose_logits)
        if self.pose_uncertainty is not None:
            prefix = self.pose_hypotheses.shape[:-2]
            if self.pose_uncertainty.shape[: len(prefix)] != prefix:
                raise ValueError(
                    "pose_uncertainty must start with the same [...,K] shape as logits"
                )

    @property
    def poses(self) -> Tensor:
        return self.pose_hypotheses

    @property
    def logits(self) -> Tensor:
        return self.pose_logits

    @property
    def uncertainty(self) -> Optional[Tensor]:
        return self.pose_uncertainty

    @property
    def num_hypotheses(self) -> int:
        return int(self.pose_hypotheses.shape[-3])

    @property
    def probabilities(self) -> Tensor:
        # Queries are independent pose/no-pose slots under the BCE set loss.
        return torch.sigmoid(self.pose_logits)

    @property
    def best_indices(self) -> Tensor:
        return torch.argmax(self.pose_logits, dim=-1)

    @property
    def best_poses(self) -> Tensor:
        return select_best_pose(self.pose_hypotheses, self.pose_logits)

    def detach(self) -> "PoseHypotheses":
        return PoseHypotheses(
            pose_hypotheses=self.pose_hypotheses.detach(),
            pose_logits=self.pose_logits.detach(),
            pose_uncertainty=(
                None if self.pose_uncertainty is None else self.pose_uncertainty.detach()
            ),
        )

    def to(self, *args, **kwargs) -> "PoseHypotheses":
        return PoseHypotheses(
            pose_hypotheses=self.pose_hypotheses.to(*args, **kwargs),
            pose_logits=self.pose_logits.to(*args, **kwargs),
            pose_uncertainty=(
                None if self.pose_uncertainty is None else self.pose_uncertainty.to(*args, **kwargs)
            ),
        )


def build_pose_hypotheses(
    rotation_6d: Tensor,
    translation: Tensor,
    pose_logits: Tensor,
    pose_uncertainty: Optional[Tensor] = None,
) -> PoseHypotheses:
    """Convert query outputs ``[...,K,6]``/``[...,K,3]`` to pose matrices."""

    matrices = make_transform(rotation_6d_to_matrix(rotation_6d), translation)
    return PoseHypotheses(matrices, pose_logits, pose_uncertainty)


def select_best_pose(pose_hypotheses: Tensor, pose_logits: Tensor) -> Tensor:
    """Select the maximum-logit pose, returning ``[...,4,4]``."""

    batch_shape, _ = _validate_pose_tensors(pose_hypotheses, pose_logits)
    indices = torch.argmax(pose_logits, dim=-1)
    flat_poses = pose_hypotheses.reshape(-1, pose_hypotheses.shape[-3], 4, 4)
    flat_indices = indices.reshape(-1)
    selected = flat_poses[
        torch.arange(flat_poses.shape[0], device=flat_poses.device), flat_indices
    ]
    return selected.reshape(*batch_shape, 4, 4)


def topk_pose_hypotheses(
    pose_hypotheses: Tensor,
    pose_logits: Tensor,
    k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return poses, logits and original query indices sorted by confidence."""

    batch_shape, count = _validate_pose_tensors(pose_hypotheses, pose_logits)
    if isinstance(k, bool) or not 1 <= int(k) <= count:
        raise ValueError(f"k must be in [1, {count}]")
    top_logits, indices = torch.topk(pose_logits, k=int(k), dim=-1)
    flat_poses = pose_hypotheses.reshape(-1, count, 4, 4)
    flat_indices = indices.reshape(-1, int(k))
    batch_indices = torch.arange(flat_poses.shape[0], device=flat_poses.device)[:, None]
    selected = flat_poses[batch_indices, flat_indices]
    return selected.reshape(*batch_shape, int(k), 4, 4), top_logits, indices


# Descriptive alias for callers that use "set" terminology.
PoseHypothesisSet = PoseHypotheses


__all__ = [
    "PoseHypotheses",
    "PoseHypothesisSet",
    "build_pose_hypotheses",
    "select_best_pose",
    "topk_pose_hypotheses",
]
