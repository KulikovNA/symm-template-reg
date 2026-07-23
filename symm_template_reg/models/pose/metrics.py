"""Pose metrics, including axial symmetry-aware errors."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor

from .pose_representation import split_transform
from .rotation import rotation_geodesic_distance


def translation_error(predicted: Tensor, target: Tensor) -> Tensor:
    """Euclidean translation error in the transforms' coordinate unit."""

    _, predicted_t = split_transform(predicted)
    _, target_t = split_transform(target)
    return torch.linalg.vector_norm(predicted_t - target_t, dim=-1)


def rotation_error_rad(predicted: Tensor, target: Tensor) -> Tensor:
    """SO(3) geodesic pose rotation error in radians."""

    predicted_R, _ = split_transform(predicted)
    target_R, _ = split_transform(target)
    return rotation_geodesic_distance(predicted_R, target_R)


def rotation_error_deg(predicted: Tensor, target: Tensor) -> Tensor:
    """SO(3) geodesic pose rotation error in degrees."""

    return torch.rad2deg(rotation_error_rad(predicted, target))


def axis_direction_error_rad(
    predicted: Tensor,
    target: Tensor,
    axis_O: Union[Tensor, Sequence[float]],
    *,
    undirected: bool = False,
) -> Tensor:
    """Angular error between object-axis directions transformed into camera space."""

    predicted_R, _ = split_transform(predicted)
    target_R, _ = split_transform(target)
    axis = torch.as_tensor(axis_O, dtype=predicted_R.dtype, device=predicted_R.device)
    if axis.shape != (3,):
        raise ValueError(f"axis_O must have shape [3], got {tuple(axis.shape)}")
    axis_norm = torch.linalg.vector_norm(axis)
    if not bool(torch.isfinite(axis_norm)) or bool(axis_norm <= 1e-12):
        raise ValueError("axis_O must be finite and non-zero")
    axis = axis / axis_norm
    predicted_axis = torch.matmul(predicted_R, axis)
    target_axis = torch.matmul(target_R, axis)
    dot = torch.sum(predicted_axis * target_axis, dim=-1).clamp(-1.0, 1.0)
    cross_norm = torch.linalg.vector_norm(
        torch.linalg.cross(predicted_axis, target_axis, dim=-1), dim=-1
    )
    if undirected:
        dot = torch.abs(dot)
    return torch.atan2(cross_norm, dot)


def axis_direction_error_deg(
    predicted: Tensor,
    target: Tensor,
    axis_O: Union[Tensor, Sequence[float]],
    *,
    undirected: bool = False,
) -> Tensor:
    """Axial direction (swing) error in degrees; twist is ignored."""

    return torch.rad2deg(
        axis_direction_error_rad(predicted, target, axis_O, undirected=undirected)
    )


def transformed_axis_origin(
    transform: Tensor,
    origin_O: Union[Tensor, Sequence[float]],
) -> Tensor:
    """Transform one object-frame point on the symmetry axis."""

    rotation, translation = split_transform(transform)
    origin = torch.as_tensor(origin_O, dtype=rotation.dtype, device=rotation.device)
    if origin.shape != (3,):
        raise ValueError(f"origin_O must have shape [3], got {tuple(origin.shape)}")
    return torch.matmul(rotation, origin) + translation


def so2_pose_errors(
    predicted: Tensor,
    target: Tensor,
    *,
    axis_O: Union[Tensor, Sequence[float]],
    origin_O: Union[Tensor, Sequence[float]] = (0.0, 0.0, 0.0),
    undirected_axis: bool = False,
) -> dict[str, Tensor]:
    """Continuous axial pose errors without a twist penalty.

    Translation compares a point on the physical symmetry axis rather than the
    arbitrary object-frame origin.  This stays invariant under right-multipled
    rotations about a non-zero axis origin.
    """

    axis_rad = axis_direction_error_rad(
        predicted, target, axis_O, undirected=undirected_axis
    )
    predicted_origin = transformed_axis_origin(predicted, origin_O)
    target_origin = transformed_axis_origin(target, origin_O)
    translation = torch.linalg.vector_norm(predicted_origin - target_origin, dim=-1)
    return {
        "axis_error_rad": axis_rad,
        "axis_error_deg": torch.rad2deg(axis_rad),
        "translation_m": translation,
    }


def symmetry_aware_pose_errors(
    predicted: Tensor,
    target: Tensor,
    symmetry: Any,
    *,
    active_regions: Optional[Union[Tensor, Sequence[bool]]] = None,
    effective_group: Optional[Any] = None,
    axis: Optional[Union[Tensor, Sequence[float]]] = None,
    origin: Optional[Union[Tensor, Sequence[float]]] = None,
    translation_weight: float = 1.0,
    undirected_so2_axis: bool = False,
) -> dict[str, Tensor]:
    """Return minimum Cn errors or analytic twist-invariant SO2 errors.

    The matching scalar is ``rotation_rad + translation_weight * translation``.
    Its components are also returned separately so callers do not need to mix
    angular and metric units in reports.
    """

    from ..symmetry.groups import SO2Group, parse_rotation_group
    from ..symmetry.hypothesis_expander import equivalent_gt_poses
    from ..symmetry.metadata import SymmetryMetadata
    from ..symmetry.region_assignment import effective_group_from_regions

    if translation_weight < 0:
        raise ValueError("translation_weight must be non-negative")

    if isinstance(symmetry, SymmetryMetadata):
        if effective_group is not None:
            group = parse_rotation_group(effective_group)
        elif active_regions is not None:
            group = effective_group_from_regions(symmetry, active_regions)
        elif symmetry.regions:
            from ..symmetry.groups import intersect_rotation_groups

            group = intersect_rotation_groups(
                [region.rotation_group for region in symmetry.regions]
            )
        else:
            from ..symmetry.groups import CyclicGroup

            group = CyclicGroup(1)
        resolved_axis = symmetry.axis.direction if axis is None else axis
        resolved_origin = symmetry.axis.origin if origin is None else origin
    else:
        group = parse_rotation_group(symmetry)
        if axis is None:
            raise ValueError("axis is required when symmetry is a rotation group")
        resolved_axis = axis
        resolved_origin = (0.0, 0.0, 0.0) if origin is None else origin

    if isinstance(group, SO2Group):
        if predicted.ndim == target.ndim + 1:
            # Native model output is [B,K,4,4], while one GT pose is [B,4,4].
            # Keep the query axis and broadcast the target across it.
            target = target.unsqueeze(-3)
        elif predicted.ndim != target.ndim:
            raise ValueError(
                "predicted and target must have matching pose batch ranks, or "
                "predicted may contain one additional query axis"
            )
        components = so2_pose_errors(
            predicted,
            target,
            axis_O=resolved_axis,
            origin_O=resolved_origin,
            undirected_axis=undirected_so2_axis,
        )
        components["rotation_rad"] = components["axis_error_rad"]
        components["rotation_deg"] = components["axis_error_deg"]
        components["combined"] = (
            components["rotation_rad"]
            + float(translation_weight) * components["translation_m"]
        )
        components["matched_index"] = torch.full_like(
            components["combined"], -1, dtype=torch.long
        )
        return components

    equivalents = equivalent_gt_poses(
        target,
        group,
        axis=resolved_axis,
        origin=resolved_origin,
    )
    predicted_expanded = predicted.unsqueeze(-3)
    if predicted.ndim == target.ndim + 1:
        # [B,K,1,4,4] against [B,1,G,4,4].
        equivalents = equivalents.unsqueeze(-4)
    elif predicted.ndim != target.ndim:
        raise ValueError(
            "predicted and target must have matching pose batch ranks, or "
            "predicted may contain one additional query axis"
        )
    rotations = rotation_error_rad(predicted_expanded, equivalents)
    translations = translation_error(predicted_expanded, equivalents)
    combined = rotations + float(translation_weight) * translations
    best = torch.argmin(combined, dim=-1)
    gather = best.unsqueeze(-1)
    best_rotation = torch.gather(rotations, -1, gather).squeeze(-1)
    best_translation = torch.gather(translations, -1, gather).squeeze(-1)
    best_combined = torch.gather(combined, -1, gather).squeeze(-1)
    return {
        "rotation_rad": best_rotation,
        "rotation_deg": torch.rad2deg(best_rotation),
        "translation_m": best_translation,
        "combined": best_combined,
        "matched_index": best,
    }


# Concise report-facing aliases.
axis_error_deg = axis_direction_error_deg
pose_rotation_error = rotation_error_rad
pose_translation_error = translation_error


__all__ = [
    "axis_direction_error_deg",
    "axis_direction_error_rad",
    "axis_error_deg",
    "pose_rotation_error",
    "pose_translation_error",
    "rotation_error_deg",
    "rotation_error_rad",
    "so2_pose_errors",
    "symmetry_aware_pose_errors",
    "transformed_axis_origin",
    "translation_error",
]
