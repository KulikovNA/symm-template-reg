"""One production symmetry-target contract shared by data, loss, and debug.

The mesh-aware entrypoint in this module is intentionally the only place that
decides which annotated symmetry regions are active for a fragment.  Dataset
point targets use the same entrypoint without faces; debug tooling passes the
complete fragment mesh and receives the additional area diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

from .groups import RotationGroup, SO2Group, group_to_dict
from .hypothesis_expander import (
    DEFAULT_SO2_NUM_SAMPLES,
    EquivalentPoseSet,
    equivalent_gt_pose_set,
    symmetry_transforms,
)
from .metadata import SymmetryMetadata
from .region_assignment import (
    assign_symmetry_regions,
    effective_group_from_regions,
    region_indices_from_membership,
)


@dataclass(frozen=True)
class SymmetryTargets:
    """Complete region and pose targets for one object-frame fragment."""

    point_region_memberships: Tensor
    point_region_indices: Tensor
    region_point_counts: Tensor
    face_region_memberships: Optional[Tensor]
    face_region_indices: Optional[Tensor]
    region_face_counts: Tensor
    region_surface_areas_m2: Tensor
    region_surface_area_fractions: Tensor
    region_area_sample_counts: Tensor
    active_regions: Tensor
    active_region_decisions: tuple[dict[str, Any], ...]
    effective_group: RotationGroup
    group_elements: Tensor
    equivalent_pose_set: EquivalentPoseSet
    training_target_type: str
    diagnostics: dict[str, Any]

    @property
    def equivalent_poses(self) -> Tensor:
        return self.equivalent_pose_set.poses

    def to_dataset_dict(self) -> dict[str, object]:
        """Return the stable subset consumed by Dataset and losses."""

        return {
            "point_symmetry_region_indices": self.point_region_indices,
            "symmetry_region_point_counts": self.region_point_counts,
            "active_symmetry_regions": self.active_regions,
            "effective_symmetry_group": group_to_dict(self.effective_group),
            "equivalent_T_C_from_O": self.equivalent_poses,
            "symmetry_training_target_type": self.training_target_type,
            "symmetry_pose_set_exhaustive": self.equivalent_pose_set.exhaustive,
        }


def _validate_points(points_O: Tensor) -> Tensor:
    points = torch.as_tensor(points_O)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"fragment_points_O must have shape [N,3], got {tuple(points.shape)}")
    if not points.is_floating_point():
        points = points.to(torch.get_default_dtype())
    if not bool(torch.isfinite(points).all()):
        raise ValueError("fragment_points_O contains NaN/Inf")
    return points


def _validate_faces(faces: Tensor, num_points: int, device: torch.device) -> Tensor:
    triangles = torch.as_tensor(faces, dtype=torch.long, device=device)
    if triangles.ndim != 2 or triangles.shape[-1] != 3:
        raise ValueError(f"fragment_faces must have shape [F,3], got {tuple(triangles.shape)}")
    if triangles.numel() and (
        bool((triangles < 0).any()) or bool((triangles >= num_points).any())
    ):
        raise ValueError("fragment_faces contains out-of-range vertex indices")
    return triangles


def _area_sample_counts(areas: Tensor, sample_count: int) -> Tensor:
    """Allocate a deterministic area-proportional diagnostic sample budget."""

    if sample_count < 1:
        raise ValueError("area_sample_count must be a positive integer")
    if areas.numel() == 0 or float(areas.sum()) <= 0.0:
        return torch.zeros_like(areas, dtype=torch.long)
    expected = areas / areas.sum() * int(sample_count)
    allocated = torch.floor(expected).to(torch.long)
    remainder = int(sample_count - int(allocated.sum()))
    if remainder:
        fractional = expected - allocated.to(expected.dtype)
        # Stable tie-breaking keeps lower region indices first.
        order = sorted(range(len(fractional)), key=lambda i: (-float(fractional[i]), i))
        for index in order[:remainder]:
            allocated[index] += 1
    return allocated


def _mesh_region_statistics(
    points: Tensor,
    faces: Tensor,
    metadata: SymmetryMetadata,
    *,
    assignment_tolerance_m: float,
    area_sample_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    centroids = points[faces].mean(dim=1)
    memberships = assign_symmetry_regions(
        centroids, metadata, atol_m=assignment_tolerance_m
    )
    indices = region_indices_from_membership(memberships)
    triangles = points[faces]
    areas = 0.5 * torch.linalg.vector_norm(
        torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=-1,
        ),
        dim=-1,
    )
    region_count = len(metadata.regions)
    face_counts = torch.zeros(region_count, dtype=torch.long, device=points.device)
    surface_areas = torch.zeros(region_count, dtype=points.dtype, device=points.device)
    assigned = indices >= 0
    if bool(assigned.any()):
        face_counts.scatter_add_(
            0, indices[assigned], torch.ones_like(indices[assigned], dtype=torch.long)
        )
        surface_areas.scatter_add_(0, indices[assigned], areas[assigned])
    total_area = surface_areas.sum()
    fractions = surface_areas / total_area.clamp_min(torch.finfo(points.dtype).eps)
    sampled = _area_sample_counts(surface_areas, area_sample_count)
    return memberships, indices, face_counts, surface_areas, fractions, sampled


def build_fragment_symmetry_targets(
    fragment_points_O: Tensor,
    symmetry_metadata: SymmetryMetadata,
    *,
    fragment_faces: Optional[Tensor] = None,
    base_pose: Optional[Tensor] = None,
    valid_mask: Optional[Tensor] = None,
    min_points: int = 1,
    min_fraction: float = 0.0,
    min_surface_area_m2: float = 0.0,
    min_surface_area_fraction: float = 0.01,
    area_sample_count: int = 2048,
    min_area_sample_count: int = 16,
    assignment_tolerance_m: float = 1e-6,
    so2_num_samples: int = DEFAULT_SO2_NUM_SAMPLES,
) -> SymmetryTargets:
    """Build production region, group, group-element, and pose targets.

    Mesh fragments use robust area evidence: a region is active when at least
    one enabled area criterion passes.  Vertex counts remain diagnostic and do
    not let a lone boundary vertex activate a mesh region.  Point-only Dataset
    inputs retain the established count-and-fraction rule.
    """

    if not isinstance(symmetry_metadata, SymmetryMetadata):
        raise TypeError("symmetry_metadata must be present SymmetryMetadata")
    if isinstance(min_points, bool) or int(min_points) < 1:
        raise ValueError("min_points must be a positive integer")
    if not 0.0 <= float(min_fraction) <= 1.0:
        raise ValueError("min_fraction must be between zero and one")
    if float(min_surface_area_m2) < 0.0:
        raise ValueError("min_surface_area_m2 must be non-negative")
    if not 0.0 <= float(min_surface_area_fraction) <= 1.0:
        raise ValueError("min_surface_area_fraction must be between zero and one")
    if isinstance(min_area_sample_count, bool) or int(min_area_sample_count) < 0:
        raise ValueError("min_area_sample_count must be a non-negative integer")
    if assignment_tolerance_m < 0.0:
        raise ValueError("assignment_tolerance_m must be non-negative")

    points = _validate_points(fragment_points_O)
    memberships = assign_symmetry_regions(
        points, symmetry_metadata, atol_m=assignment_tolerance_m
    )
    point_indices = region_indices_from_membership(memberships)
    if valid_mask is None:
        valid = torch.ones(len(points), dtype=torch.bool, device=points.device)
    else:
        valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=points.device)
        if valid.shape != (len(points),):
            raise ValueError(f"valid_mask must have shape [{len(points)}], got {tuple(valid.shape)}")
    region_point_counts = (memberships & valid.unsqueeze(-1)).sum(dim=0)
    valid_count = valid.sum()
    point_fractions = region_point_counts.to(points.dtype) / valid_count.clamp_min(1).to(points.dtype)

    region_count = len(symmetry_metadata.regions)
    face_memberships: Optional[Tensor] = None
    face_indices: Optional[Tensor] = None
    face_counts = torch.zeros(region_count, dtype=torch.long, device=points.device)
    surface_areas = torch.zeros(region_count, dtype=points.dtype, device=points.device)
    surface_fractions = torch.zeros(region_count, dtype=points.dtype, device=points.device)
    sampled_counts = torch.zeros(region_count, dtype=torch.long, device=points.device)

    if fragment_faces is None:
        active = (
            (region_point_counts >= int(min_points))
            & (point_fractions >= float(min_fraction))
            & (valid_count > 0)
        )
    else:
        triangles = _validate_faces(fragment_faces, len(points), points.device)
        (
            face_memberships,
            face_indices,
            face_counts,
            surface_areas,
            surface_fractions,
            sampled_counts,
        ) = _mesh_region_statistics(
            points,
            triangles,
            symmetry_metadata,
            assignment_tolerance_m=assignment_tolerance_m,
            area_sample_count=int(area_sample_count),
        )
        absolute_pass = (
            surface_areas >= float(min_surface_area_m2)
            if float(min_surface_area_m2) > 0.0
            else torch.zeros_like(surface_areas, dtype=torch.bool)
        )
        relative_pass = (
            surface_fractions >= float(min_surface_area_fraction)
            if float(min_surface_area_fraction) > 0.0
            else torch.zeros_like(surface_areas, dtype=torch.bool)
        )
        sample_pass = (
            sampled_counts >= int(min_area_sample_count)
            if int(min_area_sample_count) > 0
            else torch.zeros_like(surface_areas, dtype=torch.bool)
        )
        if not (
            float(min_surface_area_m2) > 0.0
            or float(min_surface_area_fraction) > 0.0
            or int(min_area_sample_count) > 0
        ):
            active = surface_areas > 0.0
        else:
            active = (absolute_pass | relative_pass | sample_pass) & (surface_areas > 0.0)

    decisions: list[dict[str, Any]] = []
    for index, region in enumerate(symmetry_metadata.regions):
        reasons: list[str] = []
        if fragment_faces is None:
            if int(region_point_counts[index]) >= int(min_points):
                reasons.append("point_count")
            if float(point_fractions[index]) >= float(min_fraction):
                reasons.append("point_fraction")
        else:
            if float(min_surface_area_m2) > 0.0 and float(surface_areas[index]) >= float(min_surface_area_m2):
                reasons.append("surface_area")
            if float(min_surface_area_fraction) > 0.0 and float(surface_fractions[index]) >= float(min_surface_area_fraction):
                reasons.append("surface_area_fraction")
            if int(min_area_sample_count) > 0 and int(sampled_counts[index]) >= int(min_area_sample_count):
                reasons.append("area_weighted_sample_count")
        decisions.append(
            {
                "region_id": region.region_id,
                "active": bool(active[index]),
                "reasons": reasons if bool(active[index]) else ["below_all_enabled_thresholds"],
            }
        )

    effective = effective_group_from_regions(symmetry_metadata, active)
    pose = (
        torch.eye(4, dtype=points.dtype, device=points.device)
        if base_pose is None
        else torch.as_tensor(base_pose, dtype=points.dtype, device=points.device)
    )
    pose_set = equivalent_gt_pose_set(
        pose,
        symmetry_metadata,
        effective_group=effective,
        so2_num_samples=so2_num_samples,
    )
    group_elements = symmetry_transforms(
        effective,
        symmetry_metadata.axis.direction,
        symmetry_metadata.axis.origin,
        so2_num_samples=so2_num_samples if isinstance(effective, SO2Group) else None,
        dtype=points.dtype,
        device=points.device,
    )
    unassigned_points = int(((point_indices < 0) & valid).sum())
    unassigned_faces = int((face_indices < 0).sum()) if face_indices is not None else 0
    diagnostics = {
        "activation_mode": "mesh_area" if fragment_faces is not None else "point_cloud",
        "thresholds": {
            "min_points": int(min_points),
            "min_fraction": float(min_fraction),
            "min_surface_area_m2": float(min_surface_area_m2),
            "min_surface_area_fraction": float(min_surface_area_fraction),
            "area_sample_count": int(area_sample_count),
            "min_area_sample_count": int(min_area_sample_count),
        },
        "unassigned_points": unassigned_points,
        "unassigned_faces": unassigned_faces,
    }
    return SymmetryTargets(
        point_region_memberships=memberships,
        point_region_indices=point_indices,
        region_point_counts=region_point_counts,
        face_region_memberships=face_memberships,
        face_region_indices=face_indices,
        region_face_counts=face_counts,
        region_surface_areas_m2=surface_areas,
        region_surface_area_fractions=surface_fractions,
        region_area_sample_counts=sampled_counts,
        active_regions=active.to(dtype=torch.bool),
        active_region_decisions=tuple(decisions),
        effective_group=effective,
        group_elements=group_elements,
        equivalent_pose_set=pose_set,
        training_target_type=(
            "continuous_analytic" if isinstance(effective, SO2Group) else "finite_exact"
        ),
        diagnostics=diagnostics,
    )


def build_symmetry_targets(
    points_O: Tensor,
    T_C_from_O: Tensor,
    metadata: SymmetryMetadata,
    *,
    valid_mask: Optional[Tensor] = None,
    min_points: int = 1,
    min_fraction: float = 0.0,
    assignment_tolerance_m: float = 1e-6,
    so2_num_samples: int = DEFAULT_SO2_NUM_SAMPLES,
) -> SymmetryTargets:
    """Compatibility wrapper for the established point-only Dataset API."""

    return build_fragment_symmetry_targets(
        points_O,
        metadata,
        base_pose=T_C_from_O,
        valid_mask=valid_mask,
        min_points=min_points,
        min_fraction=min_fraction,
        assignment_tolerance_m=assignment_tolerance_m,
        so2_num_samples=so2_num_samples,
    )


__all__ = [
    "SymmetryTargets",
    "build_fragment_symmetry_targets",
    "build_symmetry_targets",
]
