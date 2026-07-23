"""Expand a ground-truth pose by object-frame axial symmetries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from symm_template_reg.registry import SYMMETRY_MODULES

from .groups import (
    RotationGroup,
    SO2Group,
    group_to_dict,
    parse_rotation_group,
    rotation_group_matrices,
)
from .metadata import SymmetryMetadata
from .region_assignment import effective_group_from_regions


DEFAULT_SO2_NUM_SAMPLES = 36
DEFAULT_SO2_VISUALIZATION_SAMPLES = 12


@dataclass(frozen=True)
class EquivalentPoseSet:
    """A finite pose tensor plus metadata describing its completeness."""

    poses: Tensor
    group: RotationGroup
    axis: Tensor
    origin: Tensor
    is_continuous: bool
    exhaustive: bool

    @property
    def num_hypotheses(self) -> int:
        return int(self.poses.shape[-3])

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "effective_group": group_to_dict(self.group),
            "is_continuous": self.is_continuous,
            "finite_hypotheses_exhaustive": self.exhaustive,
            "num_hypotheses": self.num_hypotheses,
            "axis": self.axis.detach().cpu().tolist(),
            "origin": self.origin.detach().cpu().tolist(),
        }


def _validate_transform(T: Tensor) -> Tensor:
    transform = torch.as_tensor(T)
    if transform.shape[-2:] != (4, 4):
        raise ValueError(
            f"T_C_from_O must have shape [..., 4, 4], got {tuple(transform.shape)}"
        )
    if not transform.is_floating_point():
        transform = transform.to(torch.get_default_dtype())
    if not bool(torch.isfinite(transform).all()):
        raise ValueError("T_C_from_O must contain only finite values")
    expected_last_row = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], dtype=transform.dtype, device=transform.device
    )
    if not bool(torch.allclose(transform[..., 3, :], expected_last_row, atol=1e-5, rtol=0.0)):
        raise ValueError("T_C_from_O must be a homogeneous transform with last row [0,0,0,1]")
    return transform


def symmetry_transforms(
    group: Any,
    axis: Union[Tensor, Sequence[float]],
    origin: Union[Tensor, Sequence[float]] = (0.0, 0.0, 0.0),
    *,
    so2_num_samples: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    """Return object-frame homogeneous transforms ``[K, 4, 4]``.

    Each transform rotates about the full axis line, including a non-zero
    origin: ``x' = R @ (x - origin) + origin``.
    """

    if isinstance(axis, Tensor):
        resolved_dtype = dtype or (axis.dtype if axis.is_floating_point() else torch.float32)
        resolved_device = device if device is not None else axis.device
    else:
        resolved_dtype = dtype or torch.float32
        resolved_device = device
    axis_tensor = torch.as_tensor(axis, dtype=resolved_dtype, device=resolved_device)
    origin_tensor = torch.as_tensor(origin, dtype=resolved_dtype, device=resolved_device)
    if origin_tensor.shape != (3,):
        raise ValueError(f"origin must have shape [3], got {tuple(origin_tensor.shape)}")
    rotations = rotation_group_matrices(
        group,
        axis_tensor,
        so2_num_samples=so2_num_samples,
        dtype=resolved_dtype,
        device=resolved_device,
    )
    transforms = torch.eye(
        4, dtype=resolved_dtype, device=resolved_device
    ).expand(rotations.shape[0], 4, 4).clone()
    transforms[:, :3, :3] = rotations
    transforms[:, :3, 3] = origin_tensor - torch.matmul(
        rotations, origin_tensor.unsqueeze(-1)
    ).squeeze(-1)
    return transforms


def _resolve_symmetry(
    symmetry: Any,
    *,
    active_regions: Optional[Union[Tensor, Sequence[bool]]],
    effective_group: Optional[Any],
    axis: Optional[Union[Tensor, Sequence[float]]],
    origin: Optional[Union[Tensor, Sequence[float]]],
) -> tuple[RotationGroup, Union[Tensor, Sequence[float]], Union[Tensor, Sequence[float]]]:
    if isinstance(symmetry, SymmetryMetadata):
        if effective_group is not None:
            group = parse_rotation_group(effective_group)
        elif active_regions is not None:
            group = effective_group_from_regions(symmetry, active_regions)
        elif len(symmetry.regions) == 0:
            from .groups import CyclicGroup

            group = CyclicGroup(1)
        else:
            from .groups import intersect_rotation_groups

            group = intersect_rotation_groups(
                [region.rotation_group for region in symmetry.regions]
            )
        resolved_axis = symmetry.axis.direction if axis is None else axis
        resolved_origin = symmetry.axis.origin if origin is None else origin
        return group, resolved_axis, resolved_origin

    if effective_group is not None:
        raise ValueError(
            "effective_group is only accepted when symmetry is SymmetryMetadata"
        )
    group = parse_rotation_group(symmetry)
    if axis is None:
        raise ValueError("axis is required when symmetry is a rotation group")
    return group, axis, (0.0, 0.0, 0.0) if origin is None else origin


def equivalent_gt_pose_set(
    T_C_from_O: Tensor,
    symmetry: Union[SymmetryMetadata, RotationGroup, Mapping[str, Any], str],
    *,
    active_regions: Optional[Union[Tensor, Sequence[bool]]] = None,
    effective_group: Optional[Any] = None,
    axis: Optional[Union[Tensor, Sequence[float]]] = None,
    origin: Optional[Union[Tensor, Sequence[float]]] = None,
    so2_num_samples: int = DEFAULT_SO2_NUM_SAMPLES,
) -> EquivalentPoseSet:
    """Build equivalent poses ``T_C_from_O @ S`` and expansion metadata.

    ``Cn`` is enumerated exactly.  Since ``SO2`` has infinitely many elements,
    its returned tensor is a deterministic finite sampling and ``exhaustive``
    is false.  Continuous-aware losses should consume the group/axis metadata
    rather than treat the sampling as the complete group.
    """

    transform = _validate_transform(T_C_from_O)
    group, resolved_axis, resolved_origin = _resolve_symmetry(
        symmetry,
        active_regions=active_regions,
        effective_group=effective_group,
        axis=axis,
        origin=origin,
    )
    axis_tensor = torch.as_tensor(
        resolved_axis, dtype=transform.dtype, device=transform.device
    )
    origin_tensor = torch.as_tensor(
        resolved_origin, dtype=transform.dtype, device=transform.device
    )
    sample_count = so2_num_samples if isinstance(group, SO2Group) else None
    symmetry_T = symmetry_transforms(
        group,
        axis_tensor,
        origin_tensor,
        so2_num_samples=sample_count,
        dtype=transform.dtype,
        device=transform.device,
    )
    poses = torch.matmul(transform.unsqueeze(-3), symmetry_T)
    continuous = isinstance(group, SO2Group)
    return EquivalentPoseSet(
        poses=poses,
        group=group,
        axis=axis_tensor,
        origin=origin_tensor,
        is_continuous=continuous,
        exhaustive=not continuous,
    )


def equivalent_gt_poses(
    T_C_from_O: Tensor,
    symmetry: Union[SymmetryMetadata, RotationGroup, Mapping[str, Any], str],
    **kwargs: Any,
) -> Tensor:
    """Return only the ``[..., K, 4, 4]`` tensor from equivalent expansion."""

    return equivalent_gt_pose_set(T_C_from_O, symmetry, **kwargs).poses


def visualization_equivalent_pose_set(
    T_C_from_O: Tensor,
    symmetry: Union[SymmetryMetadata, RotationGroup, Mapping[str, Any], str],
    *,
    so2_visualization_samples: int = DEFAULT_SO2_VISUALIZATION_SAMPLES,
    **kwargs: Any,
) -> EquivalentPoseSet:
    """Return exact Cn poses or a clearly finite SO2 visualization sample.

    Continuous SO2 training semantics remain analytic; this helper only makes
    representative poses for inspection and never labels them exhaustive.
    """

    kwargs["so2_num_samples"] = int(so2_visualization_samples)
    return equivalent_gt_pose_set(T_C_from_O, symmetry, **kwargs)


def place_fragment_for_hypothesis(
    fragment_points_O: Tensor,
    base_pose: Tensor,
    hypothesis_pose: Tensor,
) -> Tensor:
    """Place one fragment exactly as an equivalent GT pose interprets it.

    A physical point rendered under ``base_pose`` has camera coordinate
    ``base_pose @ point_O``.  Re-expressing that same point under an equivalent
    training hypothesis therefore uses ``inverse(hypothesis_pose) @ base_pose``.
    The function accepts one hypothesis ``[4,4]`` or a set ``[K,4,4]`` and
    returns ``[V,3]`` or ``[K,V,3]`` respectively.
    """

    points = torch.as_tensor(fragment_points_O)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(
            f"fragment_points_O must have shape [V,3], got {tuple(points.shape)}"
        )
    if not points.is_floating_point():
        points = points.to(torch.get_default_dtype())
    base = _validate_transform(torch.as_tensor(base_pose, dtype=points.dtype, device=points.device))
    hypotheses = _validate_transform(
        torch.as_tensor(hypothesis_pose, dtype=points.dtype, device=points.device)
    )
    if base.shape != (4, 4):
        raise ValueError("base_pose must have shape [4,4]")
    homogeneous = torch.cat(
        (points, torch.ones((len(points), 1), dtype=points.dtype, device=points.device)),
        dim=-1,
    )
    camera = torch.einsum("ij,vj->vi", base, homogeneous)
    placed = torch.einsum("...ij,vj->...vi", torch.linalg.inv(hypotheses), camera)
    return placed[..., :3]


@SYMMETRY_MODULES.register_module()
class SymmetryHypothesisExpander(nn.Module):
    """Config-buildable wrapper around exact Cn / explicit SO2 expansion."""

    def __init__(self, so2_num_samples: int = DEFAULT_SO2_NUM_SAMPLES) -> None:
        super().__init__()
        self.so2_num_samples = int(so2_num_samples)

    def forward(
        self,
        T_C_from_O: Tensor,
        symmetry: Union[SymmetryMetadata, RotationGroup, Mapping[str, Any], str],
        **kwargs: Any,
    ) -> EquivalentPoseSet:
        kwargs.setdefault("so2_num_samples", self.so2_num_samples)
        return equivalent_gt_pose_set(T_C_from_O, symmetry, **kwargs)


# Naming variants used by dataset and model code.
expand_pose_hypotheses = equivalent_gt_poses
expand_equivalent_poses = equivalent_gt_poses


__all__ = [
    "DEFAULT_SO2_NUM_SAMPLES",
    "DEFAULT_SO2_VISUALIZATION_SAMPLES",
    "EquivalentPoseSet",
    "SymmetryHypothesisExpander",
    "equivalent_gt_pose_set",
    "equivalent_gt_poses",
    "expand_equivalent_poses",
    "expand_pose_hypotheses",
    "place_fragment_for_hypothesis",
    "symmetry_transforms",
    "visualization_equivalent_pose_set",
]
