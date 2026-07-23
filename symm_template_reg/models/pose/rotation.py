"""Pure-PyTorch rotation primitives.

The 6D representation follows Zhou et al., "On the Continuity of Rotation
Representations in Neural Networks" (CVPR 2019): two unconstrained 3D vectors
are orthonormalised into the first two rows of a rotation matrix.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _check_floating(value: Tensor, name: str) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _resolved_eps(value: Tensor, eps: Optional[float]) -> float:
    if eps is not None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        return float(eps)
    return max(1e-8, float(torch.finfo(value.dtype).eps))


def rotation_6d_to_matrix(rotation_6d: Tensor, eps: Optional[float] = None) -> Tensor:
    """Convert ``[..., 6]`` continuous representations to ``[..., 3, 3]``.

    Unlike a bare Gram--Schmidt implementation, this function returns a valid
    right-handed rotation even for zero or collinear input vectors.  Those
    degenerate cases use deterministic canonical fallback directions, which is
    useful during initialisation and smoke tests.
    """

    d6 = torch.as_tensor(rotation_6d)
    if d6.shape[-1:] != (6,):
        raise ValueError(f"rotation_6d must have shape [..., 6], got {tuple(d6.shape)}")
    _check_floating(d6, "rotation_6d")
    tolerance = _resolved_eps(d6, eps)

    a1, a2 = d6[..., :3], d6[..., 3:]
    norm1 = torch.linalg.vector_norm(a1, dim=-1, keepdim=True)
    fallback1 = torch.zeros_like(a1)
    fallback1[..., 0] = 1.0
    b1 = torch.where(norm1 > tolerance, a1 / norm1.clamp_min(tolerance), fallback1)

    orthogonal = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    norm2 = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)

    # Choose the canonical basis vector least aligned with b1, then project it
    # into b1's orthogonal plane.  It remains well-conditioned for every b1.
    basis_index = torch.argmin(torch.abs(b1), dim=-1)
    helper = F.one_hot(basis_index, num_classes=3).to(dtype=d6.dtype, device=d6.device)
    fallback2 = helper - torch.sum(helper * b1, dim=-1, keepdim=True) * b1
    fallback2 = F.normalize(fallback2, dim=-1, eps=tolerance)
    b2 = torch.where(
        norm2 > tolerance,
        orthogonal / norm2.clamp_min(tolerance),
        fallback2,
    )
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Convert ``[..., 3, 3]`` matrices to the non-unique 6D representation."""

    matrix = torch.as_tensor(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape [..., 3, 3], got {tuple(matrix.shape)}")
    _check_floating(matrix, "matrix")
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def skew_symmetric(vector: Tensor) -> Tensor:
    """Return cross-product matrices with shape ``[..., 3, 3]``."""

    vector = torch.as_tensor(vector)
    if vector.shape[-1:] != (3,):
        raise ValueError(f"vector must have shape [..., 3], got {tuple(vector.shape)}")
    _check_floating(vector, "vector")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def axis_angle_to_matrix(axis_angle: Tensor, eps: Optional[float] = None) -> Tensor:
    """Convert rotation vectors ``axis * angle`` from ``[..., 3]`` to matrices."""

    vector = torch.as_tensor(axis_angle)
    if vector.shape[-1:] != (3,):
        raise ValueError(f"axis_angle must have shape [..., 3], got {tuple(vector.shape)}")
    _check_floating(vector, "axis_angle")
    tolerance = _resolved_eps(vector, eps)
    theta2 = torch.sum(vector * vector, dim=-1, keepdim=True)
    # Clamp before sqrt: evaluating sqrt(0) in an inactive torch.where branch
    # still creates an infinite derivative during backward.
    safe_theta = torch.sqrt(theta2.clamp_min(tolerance * tolerance))
    safe_theta2 = theta2.clamp_min(tolerance * tolerance)

    # Taylor branches avoid loss of precision and undefined zero divisions.
    a_regular = torch.sin(safe_theta) / safe_theta
    b_regular = (1.0 - torch.cos(safe_theta)) / safe_theta2
    a_taylor = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
    b_taylor = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    small = theta2 <= tolerance * tolerance
    a = torch.where(small, a_taylor, a_regular)[..., None]
    b = torch.where(small, b_taylor, b_regular)[..., None]
    skew = skew_symmetric(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return identity + a * skew + b * torch.matmul(skew, skew)


def project_to_so3(matrix: Tensor) -> Tensor:
    """Project arbitrary ``[..., 3, 3]`` matrices onto ``SO(3)`` by SVD."""

    matrix = torch.as_tensor(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape [..., 3, 3], got {tuple(matrix.shape)}")
    _check_floating(matrix, "matrix")
    u, _, vh = torch.linalg.svd(matrix)
    candidate = torch.matmul(u, vh)
    sign = torch.where(
        torch.linalg.det(candidate) < 0,
        -torch.ones((), dtype=matrix.dtype, device=matrix.device),
        torch.ones((), dtype=matrix.dtype, device=matrix.device),
    )
    correction = torch.ones((*matrix.shape[:-2], 3), dtype=matrix.dtype, device=matrix.device)
    correction[..., -1] = sign
    return torch.matmul(u * correction.unsqueeze(-2), vh)


def rotation_geodesic_distance(matrix_a: Tensor, matrix_b: Tensor) -> Tensor:
    """Return the shortest angular distance in radians with broadcasted batches."""

    matrix_a = torch.as_tensor(matrix_a)
    matrix_b = torch.as_tensor(matrix_b, dtype=matrix_a.dtype, device=matrix_a.device)
    if matrix_a.shape[-2:] != (3, 3) or matrix_b.shape[-2:] != (3, 3):
        raise ValueError("Both rotation matrices must have shape [..., 3, 3]")
    _check_floating(matrix_a, "matrix_a")
    _check_floating(matrix_b, "matrix_b")
    relative = torch.matmul(matrix_a, matrix_b.transpose(-1, -2))
    skew_vector = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sin_angle = 0.5 * torch.linalg.vector_norm(skew_vector, dim=-1)
    cos_angle = 0.5 * (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    )
    return torch.atan2(sin_angle, cos_angle)


def is_rotation_matrix(
    matrix: Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Tensor:
    """Return a boolean validity flag for each ``[..., 3, 3]`` matrix."""

    matrix = torch.as_tensor(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape [..., 3, 3], got {tuple(matrix.shape)}")
    if not matrix.is_floating_point():
        return torch.zeros(matrix.shape[:-2], dtype=torch.bool, device=matrix.device)
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    gram = torch.matmul(matrix, matrix.transpose(-1, -2))
    orthogonal = torch.isclose(gram, identity, atol=atol, rtol=rtol).all(dim=-1).all(dim=-1)
    proper = torch.isclose(
        torch.linalg.det(matrix),
        torch.ones((), dtype=matrix.dtype, device=matrix.device),
        atol=atol,
        rtol=rtol,
    )
    finite = torch.isfinite(matrix).all(dim=-1).all(dim=-1)
    return orthogonal & proper & finite


# Common naming aliases.
rotation_6d_to_rotation_matrix = rotation_6d_to_matrix
rotation_matrix_to_6d = matrix_to_rotation_6d


__all__ = [
    "axis_angle_to_matrix",
    "is_rotation_matrix",
    "matrix_to_rotation_6d",
    "project_to_so3",
    "rotation_6d_to_matrix",
    "rotation_6d_to_rotation_matrix",
    "rotation_geodesic_distance",
    "rotation_matrix_to_6d",
    "skew_symmetric",
]
