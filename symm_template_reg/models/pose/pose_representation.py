"""Pose parameterisation and rigid-transform helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

from symm_template_reg.registry import POSE_MODULES

from .rotation import matrix_to_rotation_6d, rotation_6d_to_matrix


def make_transform(rotation: Tensor, translation: Tensor) -> Tensor:
    """Combine ``[...,3,3]`` rotation and ``[...,3]`` translation into SE(3)."""

    rotation = torch.as_tensor(rotation)
    translation = torch.as_tensor(translation, dtype=rotation.dtype, device=rotation.device)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must have shape [...,3,3], got {tuple(rotation.shape)}")
    if translation.shape[-1:] != (3,):
        raise ValueError(
            f"translation must have shape [...,3], got {tuple(translation.shape)}"
        )
    if not rotation.is_floating_point():
        raise TypeError("rotation and translation must have floating-point dtype")
    batch_shape = torch.broadcast_shapes(rotation.shape[:-2], translation.shape[:-1])
    rotation = rotation.expand(*batch_shape, 3, 3)
    translation = translation.expand(*batch_shape, 3)
    transform = torch.zeros((*batch_shape, 4, 4), dtype=rotation.dtype, device=rotation.device)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


def split_transform(transform: Tensor) -> Tuple[Tensor, Tensor]:
    """Return views of rotation ``[...,3,3]`` and translation ``[...,3]``."""

    transform = torch.as_tensor(transform)
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"transform must have shape [...,4,4], got {tuple(transform.shape)}")
    return transform[..., :3, :3], transform[..., :3, 3]


def pose_9d_to_matrix(pose_9d: Tensor, *, translation_first: bool = False) -> Tensor:
    """Decode ``[...,9]`` values containing rotation-6D and translation.

    The canonical order is ``[rotation_6d, translation_xyz]``.  Set
    ``translation_first=True`` only when adapting an external head using the
    opposite ordering.
    """

    pose = torch.as_tensor(pose_9d)
    if pose.shape[-1:] != (9,):
        raise ValueError(f"pose_9d must have shape [...,9], got {tuple(pose.shape)}")
    if translation_first:
        translation, rotation_6d = pose[..., :3], pose[..., 3:]
    else:
        rotation_6d, translation = pose[..., :6], pose[..., 6:]
    return make_transform(rotation_6d_to_matrix(rotation_6d), translation)


def matrix_to_pose_9d(transform: Tensor, *, translation_first: bool = False) -> Tensor:
    """Encode transforms using the canonical non-unique 6D rotation encoding."""

    rotation, translation = split_transform(transform)
    rotation_6d = matrix_to_rotation_6d(rotation)
    values = (translation, rotation_6d) if translation_first else (rotation_6d, translation)
    return torch.cat(values, dim=-1)


def invert_transform(transform: Tensor) -> Tensor:
    """Invert rigid transforms without a generic matrix inverse."""

    rotation, translation = split_transform(transform)
    rotation_inv = rotation.transpose(-1, -2)
    translation_inv = -torch.matmul(rotation_inv, translation.unsqueeze(-1)).squeeze(-1)
    return make_transform(rotation_inv, translation_inv)


def compose_transforms(transform_a: Tensor, transform_b: Tensor) -> Tensor:
    """Compose transforms so the result applies ``transform_b`` then ``a``."""

    if transform_a.shape[-2:] != (4, 4) or transform_b.shape[-2:] != (4, 4):
        raise ValueError("Both transforms must have shape [...,4,4]")
    return torch.matmul(transform_a, transform_b)


def transform_points(transform: Tensor, points: Tensor) -> Tensor:
    """Apply ``T_target_from_source`` to row-stored points ``[...,N,3]``."""

    rotation, translation = split_transform(transform)
    points = torch.as_tensor(points, dtype=rotation.dtype, device=rotation.device)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [...,N,3], got {tuple(points.shape)}")
    return torch.matmul(points, rotation.transpose(-1, -2)) + translation.unsqueeze(-2)


def validate_transform(
    transform: Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Tensor:
    """Return one boolean SE(3)-validity flag per transform."""

    from .rotation import is_rotation_matrix

    transform = torch.as_tensor(transform)
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"transform must have shape [...,4,4], got {tuple(transform.shape)}")
    rotation_ok = is_rotation_matrix(transform[..., :3, :3], atol=atol, rtol=rtol)
    expected = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], dtype=transform.dtype, device=transform.device
    )
    last_row_ok = torch.isclose(
        transform[..., 3, :], expected, atol=atol, rtol=rtol
    ).all(dim=-1)
    finite = torch.isfinite(transform).all(dim=-1).all(dim=-1)
    return rotation_ok & last_row_ok & finite


@dataclass
class PoseParameters:
    """Direct network outputs before conversion to a homogeneous pose."""

    rotation_6d: Tensor
    translation: Tensor

    def as_matrix(self) -> Tensor:
        return make_transform(rotation_6d_to_matrix(self.rotation_6d), self.translation)


@POSE_MODULES.register_module()
class PoseRepresentation(nn.Module):
    """Stateless module wrapper for config-built model components."""

    def forward(self, rotation_6d: Tensor, translation: Tensor) -> Tensor:
        return make_transform(rotation_6d_to_matrix(rotation_6d), translation)


# Explicit coordinate-frame alias used throughout the dataset/model contract.
compose_pose_matrix = make_transform
invert_pose = invert_transform


__all__ = [
    "PoseParameters",
    "PoseRepresentation",
    "compose_pose_matrix",
    "compose_transforms",
    "invert_pose",
    "invert_transform",
    "make_transform",
    "matrix_to_pose_9d",
    "pose_9d_to_matrix",
    "split_transform",
    "transform_points",
    "validate_transform",
]
