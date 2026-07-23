"""Assign object-frame points to axial symmetry regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from .groups import CyclicGroup, RotationGroup, intersect_rotation_groups
from .metadata import SymmetryMetadata


@dataclass(frozen=True)
class RegionPartitionValidation:
    """Strict template partition result shared by debug and tests."""

    vertex_memberships: Tensor
    face_memberships: Tensor
    vertex_region_indices: Tensor
    face_region_indices: Tensor
    bbox_axis_min_m: float
    bbox_axis_max_m: float
    coverage_ok: bool
    unassigned_vertices: int
    overlap_vertices: int
    unassigned_faces: int
    overlap_faces: int


def axial_coordinates(points_O: Tensor, metadata: SymmetryMetadata) -> Tensor:
    """Project object-frame points onto the metadata axis.

    Although version-1 interval fields are named ``y_min_m``/``y_max_m``, the
    coordinate is computed along the configured axis direction.  For the
    canonical ``y`` axis this is exactly the object-frame y coordinate.
    """

    points = torch.as_tensor(points_O)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"points_O must have shape [..., N, 3], got {tuple(points.shape)}")
    if not points.is_floating_point():
        points = points.to(torch.get_default_dtype())
    origin = torch.as_tensor(metadata.axis.origin, dtype=points.dtype, device=points.device)
    direction = torch.as_tensor(
        metadata.axis.direction, dtype=points.dtype, device=points.device
    )
    return torch.sum((points - origin) * direction, dim=-1)


def assign_symmetry_regions(
    points_O: Tensor,
    metadata: SymmetryMetadata,
    *,
    atol_m: float = 0.0,
) -> Tensor:
    """Return a boolean membership tensor with shape ``[..., N, R]``."""

    if not isinstance(metadata, SymmetryMetadata):
        raise TypeError("metadata must be SymmetryMetadata")
    if atol_m < 0:
        raise ValueError("atol_m must be non-negative")
    coordinate = axial_coordinates(points_O, metadata)
    if not metadata.regions:
        return torch.empty((*coordinate.shape, 0), dtype=torch.bool, device=coordinate.device)
    # Deterministic interval convention: every internal boundary belongs to
    # the following region. Only the final interval includes its upper bound.
    # ``atol_m`` is restricted to the two outer template bounds so it cannot
    # make adjacent regions overlap.
    memberships = []
    final_index = len(metadata.regions) - 1
    for index, region in enumerate(metadata.regions):
        lower = region.y_min_m - atol_m if index == 0 else region.y_min_m
        upper = region.y_max_m + atol_m if index == final_index else region.y_max_m
        lower_ok = coordinate >= lower
        upper_ok = coordinate <= upper if index == final_index else coordinate < upper
        memberships.append(lower_ok & upper_ok)
    return torch.stack(memberships, dim=-1)


def region_indices_from_membership(
    memberships: Tensor,
    *,
    allow_unassigned: bool = True,
) -> Tensor:
    """Convert ``[...,N,R]`` membership to indices, using ``-1`` outside bands."""

    values = torch.as_tensor(memberships, dtype=torch.bool)
    if values.ndim < 2:
        raise ValueError("memberships must have shape [...,N,R]")
    counts = values.sum(dim=-1)
    if bool((counts > 1).any()):
        raise ValueError("symmetry regions assign at least one point more than once")
    if not allow_unassigned and bool((counts == 0).any()):
        raise ValueError("symmetry regions leave at least one point unassigned")
    indices = values.to(dtype=torch.int64).argmax(dim=-1)
    return torch.where(counts == 0, torch.full_like(indices, -1), indices)


def validate_region_partition(
    points_O: Tensor,
    faces: Tensor,
    metadata: SymmetryMetadata,
    *,
    coverage_tolerance_m: float = 1e-6,
) -> RegionPartitionValidation:
    """Validate that ordered axial bands partition an entire template mesh.

    Faces are assigned by their object-frame centroids. The same half-open
    interval policy as :func:`assign_symmetry_regions` is used for vertices,
    face centroids, Dataset targets, and debug visualization.
    """

    if coverage_tolerance_m < 0:
        raise ValueError("coverage_tolerance_m must be non-negative")
    points = torch.as_tensor(points_O)
    face_indices = torch.as_tensor(faces, dtype=torch.long, device=points.device)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points_O must have shape [V,3]")
    if face_indices.ndim != 2 or face_indices.shape[-1] != 3:
        raise ValueError("faces must have shape [F,3]")
    if face_indices.numel() and (
        bool((face_indices < 0).any()) or bool((face_indices >= len(points)).any())
    ):
        raise ValueError("faces contain out-of-range vertex indices")
    if not metadata.regions:
        raise ValueError("symmetry sidecar must contain at least one region")

    errors: list[str] = []
    for index, region in enumerate(metadata.regions):
        if not region.y_min_m < region.y_max_m:
            errors.append(f"region {region.region_id} has a non-positive interval")
        if index:
            previous = metadata.regions[index - 1]
            if region.y_min_m < previous.y_min_m:
                errors.append("regions are not sorted by lower axial bound")
            delta = region.y_min_m - previous.y_max_m
            if delta > coverage_tolerance_m:
                errors.append(
                    f"gap of {delta:.9g} m between {previous.region_id} and {region.region_id}"
                )
            elif delta < -coverage_tolerance_m:
                errors.append(
                    f"overlap of {-delta:.9g} m between {previous.region_id} and {region.region_id}"
                )

    coordinates = axial_coordinates(points, metadata)
    bbox_min = float(coordinates.min())
    bbox_max = float(coordinates.max())
    coverage_ok = (
        abs(metadata.regions[0].y_min_m - bbox_min) <= coverage_tolerance_m
        and abs(metadata.regions[-1].y_max_m - bbox_max) <= coverage_tolerance_m
    )
    if not coverage_ok:
        errors.append(
            "region bounds do not cover the template axial bbox within "
            f"{coverage_tolerance_m:.9g} m"
        )

    vertex_memberships = assign_symmetry_regions(
        points, metadata, atol_m=coverage_tolerance_m
    )
    centroids = points[face_indices].mean(dim=1)
    face_memberships = assign_symmetry_regions(
        centroids, metadata, atol_m=coverage_tolerance_m
    )
    vertex_counts = vertex_memberships.sum(dim=-1)
    face_counts = face_memberships.sum(dim=-1)
    unassigned_vertices = int((vertex_counts == 0).sum())
    overlap_vertices = int((vertex_counts > 1).sum())
    unassigned_faces = int((face_counts == 0).sum())
    overlap_faces = int((face_counts > 1).sum())
    if unassigned_vertices or overlap_vertices:
        errors.append(
            f"vertex assignment invalid: unassigned={unassigned_vertices}, "
            f"multiply_assigned={overlap_vertices}"
        )
    if unassigned_faces or overlap_faces:
        errors.append(
            f"face assignment invalid: unassigned={unassigned_faces}, "
            f"multiply_assigned={overlap_faces}"
        )
    if errors:
        raise ValueError("; ".join(errors))

    return RegionPartitionValidation(
        vertex_memberships=vertex_memberships,
        face_memberships=face_memberships,
        vertex_region_indices=region_indices_from_membership(
            vertex_memberships, allow_unassigned=False
        ),
        face_region_indices=region_indices_from_membership(
            face_memberships, allow_unassigned=False
        ),
        bbox_axis_min_m=bbox_min,
        bbox_axis_max_m=bbox_max,
        coverage_ok=coverage_ok,
        unassigned_vertices=unassigned_vertices,
        overlap_vertices=overlap_vertices,
        unassigned_faces=unassigned_faces,
        overlap_faces=overlap_faces,
    )


def active_symmetry_regions(
    points_O: Tensor,
    metadata: SymmetryMetadata,
    *,
    valid_mask: Optional[Tensor] = None,
    min_points: int = 1,
    min_fraction: float = 0.0,
    atol_m: float = 0.0,
) -> Tensor:
    """Reduce point memberships to active-region flags ``[..., R]``.

    For an unbatched ``[N, 3]`` point tensor the result is ``[R]``.  Batched
    ``[..., N, 3]`` inputs yield ``[..., R]``.  Padding can be excluded with a
    ``valid_mask`` of shape ``[..., N]``.
    """

    if isinstance(min_points, bool) or int(min_points) < 1:
        raise ValueError("min_points must be a positive integer")
    if not 0.0 <= float(min_fraction) <= 1.0:
        raise ValueError("min_fraction must be between zero and one")

    memberships = assign_symmetry_regions(points_O, metadata, atol_m=atol_m)
    point_shape = memberships.shape[:-1]
    if valid_mask is None:
        valid = torch.ones(point_shape, dtype=torch.bool, device=memberships.device)
    else:
        valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=memberships.device)
        if valid.shape != point_shape:
            raise ValueError(
                f"valid_mask must have shape {tuple(point_shape)}, got {tuple(valid.shape)}"
            )

    masked = memberships & valid.unsqueeze(-1)
    counts = masked.sum(dim=-2)
    valid_counts = valid.sum(dim=-1, keepdim=True)
    fractions = counts.to(torch.float32) / valid_counts.clamp_min(1).to(torch.float32)
    return (
        (counts >= int(min_points))
        & (fractions >= float(min_fraction))
        & (valid_counts > 0)
    )


def _mask_to_bools(active_regions: Union[Tensor, Sequence[bool]], expected: int) -> list[bool]:
    mask = torch.as_tensor(active_regions, dtype=torch.bool)
    if mask.ndim != 1 or mask.numel() != expected:
        raise ValueError(f"active_regions must have shape [{expected}], got {tuple(mask.shape)}")
    return [bool(value) for value in mask.detach().cpu().tolist()]


def effective_group_from_regions(
    metadata: SymmetryMetadata,
    active_regions: Union[Tensor, Sequence[bool]],
) -> RotationGroup:
    """Intersect groups for active regions.

    When a valid sidecar is present but none of its regions is active, only the
    identity is justified, so ``C1`` is returned.  This function must not be
    called for a missing sidecar; absence is represented by ``metadata=None``
    at the dataset boundary.
    """

    if not isinstance(metadata, SymmetryMetadata):
        raise TypeError("metadata must be SymmetryMetadata (not None)")
    mask = _mask_to_bools(active_regions, len(metadata.regions))
    groups = [
        region.rotation_group
        for region, is_active in zip(metadata.regions, mask)
        if is_active
    ]
    if not groups:
        return CyclicGroup(1)
    return intersect_rotation_groups(groups)


def effective_groups_from_regions(
    metadata: SymmetryMetadata,
    active_regions: Union[Tensor, Sequence[Sequence[bool]]],
) -> list[RotationGroup]:
    """Batched convenience wrapper returning one Python group per sample."""

    masks = torch.as_tensor(active_regions, dtype=torch.bool)
    if masks.ndim != 2 or masks.shape[-1] != len(metadata.regions):
        raise ValueError(
            f"active_regions must have shape [B, {len(metadata.regions)}], "
            f"got {tuple(masks.shape)}"
        )
    return [effective_group_from_regions(metadata, row) for row in masks]


# Short aliases useful in dataset code.
assign_regions = assign_symmetry_regions
active_regions = active_symmetry_regions


__all__ = [
    "active_regions",
    "active_symmetry_regions",
    "assign_regions",
    "assign_symmetry_regions",
    "axial_coordinates",
    "effective_group_from_regions",
    "effective_groups_from_regions",
    "region_indices_from_membership",
    "RegionPartitionValidation",
    "validate_region_partition",
]
