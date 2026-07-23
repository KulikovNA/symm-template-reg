"""Resolve symmetry from visible geometry conditioned on each predicted pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .groups import RotationGroup, group_to_dict
from .hypothesis_expander import visualization_equivalent_pose_set
from .metadata import SymmetryMetadata
from .region_assignment import (
    active_symmetry_regions,
    assign_symmetry_regions,
    effective_group_from_regions,
    region_indices_from_membership,
)


@dataclass(frozen=True)
class PoseConditionedSymmetryResolution:
    """Ragged production result indexed as ``[batch][base_query]``."""

    point_region_ids_per_pose: list[list[Tensor]]
    region_point_counts_per_pose: list[list[Tensor]]
    region_point_fractions_per_pose: list[list[Tensor]]
    active_regions_per_pose: list[list[Tensor]]
    effective_group_per_pose: list[list[RotationGroup | None]]
    effective_group_status_per_pose: list[list[str]]
    expanded_poses_per_base_pose: list[list[Tensor]]
    out_of_sidecar_bounds_fraction: list[list[float]]
    unresolved_flags: list[list[bool]]
    diagnostics: list[list[dict[str, Any]]]


def _object_points_from_camera(points_C: Tensor, T_C_from_O: Tensor) -> Tensor:
    rotation = T_C_from_O[:3, :3]
    translation = T_C_from_O[:3, 3]
    return torch.matmul(points_C - translation, rotation)


class PoseConditionedSymmetryResolver:
    """Use visible points and each base pose to infer the production group.

    Learned region logits are deliberately absent from this API.  An
    unresolved item expands to its one base pose and carries an explicit
    warning instead of silently substituting a learned ``Cn``.
    """

    def resolve(
        self,
        observed_points_C: Tensor,
        observed_valid_mask: Tensor,
        base_pose_hypotheses: Tensor,
        symmetry_metadata: Sequence[SymmetryMetadata | None] | SymmetryMetadata,
        region_activity_config: Mapping[str, Any] | None,
    ) -> PoseConditionedSymmetryResolution:
        points = torch.as_tensor(observed_points_C)
        valid = torch.as_tensor(
            observed_valid_mask, dtype=torch.bool, device=points.device
        )
        poses = torch.as_tensor(
            base_pose_hypotheses, dtype=points.dtype, device=points.device
        )
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("observed_points_C must have shape [B,N,3]")
        if valid.shape != points.shape[:2]:
            raise ValueError("observed_valid_mask must have shape [B,N]")
        if poses.ndim != 4 or poses.shape[:1] != points.shape[:1] or poses.shape[-2:] != (4, 4):
            raise ValueError("base_pose_hypotheses must have shape [B,K,4,4]")
        if isinstance(symmetry_metadata, SymmetryMetadata):
            metadata_list: list[SymmetryMetadata | None] = [symmetry_metadata]
        else:
            metadata_list = list(symmetry_metadata)
        if len(metadata_list) != len(points):
            raise ValueError("symmetry_metadata length must equal batch size")

        activity = dict(region_activity_config or {})
        min_points = int(activity.get("min_points", 1))
        min_fraction = float(activity.get("min_fraction", 0.0))
        tolerance = float(activity.get("boundary_tolerance_m", 1e-6))
        unresolved_policy = str(
            activity.get("unresolved_group_policy", "base_pose_only")
        )
        if unresolved_policy != "base_pose_only":
            raise ValueError(
                "only unresolved_group_policy='base_pose_only' is safe by default"
            )
        so2_samples = int(activity.get("so2_visualization_samples", 12))

        ids_all: list[list[Tensor]] = []
        counts_all: list[list[Tensor]] = []
        fractions_all: list[list[Tensor]] = []
        active_all: list[list[Tensor]] = []
        groups_all: list[list[RotationGroup | None]] = []
        status_all: list[list[str]] = []
        expanded_all: list[list[Tensor]] = []
        out_all: list[list[float]] = []
        unresolved_all: list[list[bool]] = []
        diagnostics_all: list[list[dict[str, Any]]] = []

        for batch_index, metadata in enumerate(metadata_list):
            ids_row: list[Tensor] = []
            counts_row: list[Tensor] = []
            fractions_row: list[Tensor] = []
            active_row: list[Tensor] = []
            groups_row: list[RotationGroup | None] = []
            status_row: list[str] = []
            expanded_row: list[Tensor] = []
            out_row: list[float] = []
            unresolved_row: list[bool] = []
            diagnostics_row: list[dict[str, Any]] = []
            selected_points_C = points[batch_index, valid[batch_index]]
            for query_index, pose in enumerate(poses[batch_index]):
                warnings: list[str] = []
                points_O = _object_points_from_camera(selected_points_C, pose)
                if metadata is None or not metadata.regions or len(points_O) == 0:
                    region_count = 0 if metadata is None else len(metadata.regions)
                    ids = torch.full(
                        (len(points_O),), -1, dtype=torch.long, device=points.device
                    )
                    counts = torch.zeros(region_count, dtype=torch.long, device=points.device)
                    fractions = torch.zeros(region_count, dtype=points.dtype, device=points.device)
                    active = torch.zeros(region_count, dtype=torch.bool, device=points.device)
                    group = None
                    expanded = pose.unsqueeze(0)
                    out_fraction = 1.0 if len(points_O) else 0.0
                    unresolved = True
                    status = "unresolved"
                    warnings.append("no resolvable visible symmetry-region evidence; base pose only")
                else:
                    memberships = assign_symmetry_regions(
                        points_O, metadata, atol_m=tolerance
                    )
                    ids = region_indices_from_membership(memberships)
                    counts = memberships.sum(dim=0)
                    fractions = counts.to(points.dtype) / max(len(points_O), 1)
                    active = active_symmetry_regions(
                        points_O,
                        metadata,
                        min_points=min_points,
                        min_fraction=min_fraction,
                        atol_m=tolerance,
                    )
                    out_fraction = float(ids.lt(0).float().mean()) if len(ids) else 0.0
                    unresolved = not bool(active.any())
                    if unresolved:
                        group = None
                        status = "unresolved"
                        expanded = pose.unsqueeze(0)
                        warnings.append(
                            "no active sidecar region after thresholds; base pose only"
                        )
                    else:
                        group = effective_group_from_regions(metadata, active)
                        status = "resolved"
                        expanded = visualization_equivalent_pose_set(
                            pose,
                            metadata,
                            effective_group=group,
                            so2_visualization_samples=so2_samples,
                        ).poses
                ids_row.append(ids)
                counts_row.append(counts)
                fractions_row.append(fractions)
                active_row.append(active)
                groups_row.append(group)
                status_row.append(status)
                expanded_row.append(expanded)
                out_row.append(out_fraction)
                unresolved_row.append(unresolved)
                diagnostics_row.append(
                    {
                        "batch_index": batch_index,
                        "base_query_index": query_index,
                        "num_visible_points": len(points_O),
                        "effective_group_status": status,
                        "effective_group": group_to_dict(group) if group is not None else None,
                        "unresolved_group_policy": unresolved_policy,
                        "out_of_sidecar_bounds_fraction": out_fraction,
                        "warnings": warnings,
                    }
                )
            ids_all.append(ids_row)
            counts_all.append(counts_row)
            fractions_all.append(fractions_row)
            active_all.append(active_row)
            groups_all.append(groups_row)
            status_all.append(status_row)
            expanded_all.append(expanded_row)
            out_all.append(out_row)
            unresolved_all.append(unresolved_row)
            diagnostics_all.append(diagnostics_row)

        return PoseConditionedSymmetryResolution(
            point_region_ids_per_pose=ids_all,
            region_point_counts_per_pose=counts_all,
            region_point_fractions_per_pose=fractions_all,
            active_regions_per_pose=active_all,
            effective_group_per_pose=groups_all,
            effective_group_status_per_pose=status_all,
            expanded_poses_per_base_pose=expanded_all,
            out_of_sidecar_bounds_fraction=out_all,
            unresolved_flags=unresolved_all,
            diagnostics=diagnostics_all,
        )


__all__ = [
    "PoseConditionedSymmetryResolution",
    "PoseConditionedSymmetryResolver",
]
