"""Rotation groups used by template symmetry metadata.

The implementation deliberately models :math:`SO(2)` as a continuous group
instead of silently replacing it with a particular discretisation.  Callers
which need a finite tensor (for visualisation or set targets) must choose the
number of samples explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from math import gcd
from numbers import Integral
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class CyclicGroup:
    """Finite cyclic rotation group ``Cn`` around a configured axis."""

    order: int
    type: str = field(default="C", init=False)

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, Integral):
            raise TypeError("Cyclic group order must be an integer")
        if int(self.order) < 1:
            raise ValueError("Cyclic group order must be at least one")
        object.__setattr__(self, "order", int(self.order))

    @property
    def cardinality(self) -> int:
        return self.order

    @property
    def is_continuous(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return f"C{self.order}"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "C", "order": self.order}


@dataclass(frozen=True)
class SO2Group:
    """Continuous rotations around a configured axis."""

    type: str = field(default="SO2", init=False)

    @property
    def order(self) -> None:
        return None

    @property
    def cardinality(self) -> None:
        return None

    @property
    def is_continuous(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "SO2"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "SO2"}


RotationGroup = Union[CyclicGroup, SO2Group]
SymmetryGroup = RotationGroup

# Descriptive aliases kept public for config code which prefers longer names.
CyclicRotationGroup = CyclicGroup
ContinuousRotationGroup = SO2Group


def parse_rotation_group(value: Any) -> RotationGroup:
    """Parse a rotation group from a dataclass, mapping, or compact string.

    Accepted strings are ``"C4"``, ``"C1"`` and ``"SO2"``.  The canonical
    JSON representation is ``{"type": "C", "order": 4}`` or
    ``{"type": "SO2"}``.
    """

    if isinstance(value, (CyclicGroup, SO2Group)):
        return value

    if isinstance(value, str):
        compact = value.strip().upper().replace("_", "")
        if compact in {"SO2", "SO(2)"}:
            return SO2Group()
        if compact.startswith("C") and compact[1:].isdigit():
            return CyclicGroup(int(compact[1:]))
        raise ValueError(f"Unsupported rotation group string: {value!r}")

    if not isinstance(value, Mapping):
        raise TypeError(
            "Rotation group must be CyclicGroup, SO2Group, a mapping, or a string"
        )

    group_type = str(value.get("type", "")).strip().upper().replace("_", "")
    if group_type == "C":
        if "order" not in value:
            raise ValueError("A cyclic rotation group requires an 'order'")
        return CyclicGroup(value["order"])
    if group_type in {"SO2", "SO(2)"}:
        if value.get("order") not in (None,):
            raise ValueError("SO2 is continuous and must not define an order")
        return SO2Group()
    raise ValueError(f"Unsupported rotation group type: {value.get('type')!r}")


def group_to_dict(group: Any) -> dict[str, Any]:
    """Return the canonical JSON-compatible representation of ``group``."""

    return parse_rotation_group(group).to_dict()


def _coerce_group_sequence(groups: Tuple[Any, ...]) -> list[RotationGroup]:
    if len(groups) == 1:
        candidate = groups[0]
        if not isinstance(candidate, (CyclicGroup, SO2Group, Mapping, str)):
            try:
                candidate = list(candidate)
            except TypeError:
                pass
            else:
                return [parse_rotation_group(group) for group in candidate]
    return [parse_rotation_group(group) for group in groups]


def intersect_rotation_groups(*groups: Any) -> RotationGroup:
    """Intersect collinear axial rotation groups deterministically.

    ``SO2`` acts as the unconstrained axial group and ``Cn ∩ Cm`` is
    ``C_gcd(n, m)``.  Therefore any ``C1`` in the input restricts the result to
    the identity group.  Groups must refer to the same physical axis; the axis
    itself is stored in :class:`~.metadata.SymmetryMetadata` and is intentionally
    not duplicated here.
    """

    parsed = _coerce_group_sequence(groups)
    if not parsed:
        raise ValueError("At least one rotation group is required")

    finite_orders = [group.order for group in parsed if isinstance(group, CyclicGroup)]
    if not finite_orders:
        return SO2Group()
    return CyclicGroup(reduce(gcd, finite_orders))


def intersect_groups(*groups: Any) -> RotationGroup:
    """Alias for :func:`intersect_rotation_groups`."""

    return intersect_rotation_groups(*groups)


def group_angles(
    group: Any,
    *,
    so2_num_samples: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    """Return uniformly spaced group angles in ``[0, 2π)``.

    For ``Cn`` all exact group elements are returned.  For continuous ``SO2``
    the caller must provide ``so2_num_samples``; this makes the approximation
    visible at the API boundary.
    """

    parsed = parse_rotation_group(group)
    if isinstance(parsed, CyclicGroup):
        count = parsed.order
    else:
        if so2_num_samples is None:
            raise ValueError(
                "SO2 is continuous; provide so2_num_samples for a finite approximation"
            )
        if isinstance(so2_num_samples, bool) or int(so2_num_samples) < 1:
            raise ValueError("so2_num_samples must be a positive integer")
        count = int(so2_num_samples)

    indices = torch.arange(count, dtype=dtype, device=device)
    return indices * (2.0 * torch.pi / count)


def axis_angle_rotation_matrices(axis: Tensor, angles: Tensor, eps: float = 1e-12) -> Tensor:
    """Create rotation matrices for rotations about one 3D axis.

    Args:
        axis: Tensor with shape ``[3]``.
        angles: Tensor with arbitrary shape ``[...]`` in radians.

    Returns:
        Tensor with shape ``[..., 3, 3]``.
    """

    axis = torch.as_tensor(axis)
    angles = torch.as_tensor(angles, dtype=axis.dtype, device=axis.device)
    if axis.shape != (3,):
        raise ValueError(f"axis must have shape [3], got {tuple(axis.shape)}")
    if not (axis.is_floating_point() and angles.is_floating_point()):
        raise TypeError("axis and angles must use a floating-point dtype")
    norm = torch.linalg.vector_norm(axis)
    if not bool(torch.isfinite(norm)) or bool(norm <= eps):
        raise ValueError("axis must be finite and have non-zero length")
    unit_axis = axis / norm
    x, y, z = unit_axis.unbind()
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    outer = unit_axis[:, None] * unit_axis[None, :]
    cos = torch.cos(angles)[..., None, None]
    sin = torch.sin(angles)[..., None, None]
    return cos * identity + (1.0 - cos) * outer + sin * skew


def rotation_group_matrices(
    group: Any,
    axis: Union[Tensor, Sequence[float]],
    *,
    so2_num_samples: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    """Return ``[K, 3, 3]`` rotations for a finite group/sampling."""

    if isinstance(axis, Tensor):
        resolved_dtype = dtype or (axis.dtype if axis.is_floating_point() else torch.float32)
        resolved_device = device if device is not None else axis.device
    else:
        resolved_dtype = dtype or torch.float32
        resolved_device = device
    axis_tensor = torch.as_tensor(axis, dtype=resolved_dtype, device=resolved_device)
    angles = group_angles(
        group,
        so2_num_samples=so2_num_samples,
        dtype=resolved_dtype,
        device=resolved_device,
    )
    return axis_angle_rotation_matrices(axis_tensor, angles)


__all__ = [
    "ContinuousRotationGroup",
    "CyclicGroup",
    "CyclicRotationGroup",
    "RotationGroup",
    "SO2Group",
    "SymmetryGroup",
    "axis_angle_rotation_matrices",
    "group_angles",
    "group_to_dict",
    "intersect_groups",
    "intersect_rotation_groups",
    "parse_rotation_group",
    "rotation_group_matrices",
]
