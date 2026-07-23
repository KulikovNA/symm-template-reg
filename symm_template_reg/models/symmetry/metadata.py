"""Typed loader for optional template-level symmetry sidecars."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from .groups import (
    RotationGroup,
    SO2Group,
    group_to_dict,
    parse_rotation_group,
)


PathLike = Union[str, Path]


def _finite_vector3(value: Any, field_name: str) -> Tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{field_name} must be an array of three numbers")
    if any(isinstance(component, bool) or not isinstance(component, Real) for component in value):
        raise ValueError(f"{field_name} must contain only numbers")
    vector = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{field_name} must contain only finite numbers")
    return vector  # type: ignore[return-value]


@dataclass(frozen=True)
class SymmetryAxis:
    """An oriented axis line expressed in the template coordinate frame."""

    name: str
    origin: Tuple[float, float, float]
    direction: Tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("axis.name must be a non-empty string")
        origin = _finite_vector3(self.origin, "axis.origin")
        direction = _finite_vector3(self.direction, "axis.direction")
        norm = math.sqrt(sum(component * component for component in direction))
        if norm <= 1e-12:
            raise ValueError("axis.direction must have non-zero length")
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"axis.direction must be normalized (norm={norm:.9g})"
            )
        normalized = tuple(component / norm for component in direction)
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "direction", normalized)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SymmetryAxis":
        if not isinstance(value, Mapping):
            raise ValueError("axis must be an object")
        required = {"name", "origin", "direction"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"axis is missing required fields: {sorted(missing)}")
        extra = set(value).difference(required)
        if extra:
            raise ValueError(f"axis has unknown fields: {sorted(extra)}")
        return cls(
            name=value["name"],
            origin=_finite_vector3(value["origin"], "axis.origin"),
            direction=_finite_vector3(value["direction"], "axis.direction"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "origin": list(self.origin),
            "direction": list(self.direction),
        }


@dataclass(frozen=True)
class SymmetryRegion:
    """Axial interval whose visible/corresponding points activate a group."""

    region_id: str
    y_min_m: float
    y_max_m: float
    rotation_group: RotationGroup

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise ValueError("region_id must be a non-empty string")
        if (
            isinstance(self.y_min_m, bool)
            or not isinstance(self.y_min_m, Real)
            or isinstance(self.y_max_m, bool)
            or not isinstance(self.y_max_m, Real)
        ):
            raise TypeError("Region bounds must be numbers")
        y_min = float(self.y_min_m)
        y_max = float(self.y_max_m)
        if not (math.isfinite(y_min) and math.isfinite(y_max)):
            raise ValueError("Region bounds must be finite")
        if y_min >= y_max:
            raise ValueError("y_min_m must be strictly less than y_max_m")
        object.__setattr__(self, "region_id", self.region_id.strip())
        object.__setattr__(self, "y_min_m", y_min)
        object.__setattr__(self, "y_max_m", y_max)
        object.__setattr__(self, "rotation_group", parse_rotation_group(self.rotation_group))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SymmetryRegion":
        if not isinstance(value, Mapping):
            raise ValueError("Each regions entry must be an object")
        required = {"region_id", "y_min_m", "y_max_m", "rotation_group"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Symmetry region is missing fields: {sorted(missing)}")
        extra = set(value).difference(required)
        if extra:
            raise ValueError(f"Symmetry region has unknown fields: {sorted(extra)}")
        try:
            return cls(
                region_id=value["region_id"],
                y_min_m=value["y_min_m"],
                y_max_m=value["y_max_m"],
                rotation_group=parse_rotation_group(value["rotation_group"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid symmetry region {value.get('region_id')!r}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "y_min_m": self.y_min_m,
            "y_max_m": self.y_max_m,
            "rotation_group": group_to_dict(self.rotation_group),
        }


@dataclass(frozen=True)
class SymmetryMetadata:
    """Validated version-1 symmetry metadata for one template object."""

    version: int
    object_model_id: str
    coordinate_frame: str
    axis: SymmetryAxis
    regions: Tuple[SymmetryRegion, ...]
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, Integral)
            or int(self.version) != 1
        ):
            raise ValueError(f"Unsupported symmetry metadata version: {self.version!r}")
        if not isinstance(self.object_model_id, str) or not self.object_model_id.strip():
            raise ValueError("object_model_id must be a non-empty string")
        if self.coordinate_frame != "O":
            raise ValueError("coordinate_frame must be 'O' in schema version 1")
        if not isinstance(self.axis, SymmetryAxis):
            raise TypeError("axis must be SymmetryAxis")
        regions = tuple(self.regions)
        if not all(isinstance(region, SymmetryRegion) for region in regions):
            raise TypeError("regions must contain only SymmetryRegion values")
        region_ids = [region.region_id for region in regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region_id values must be unique within a sidecar")
        object.__setattr__(self, "version", 1)
        object.__setattr__(self, "object_model_id", self.object_model_id.strip())
        object.__setattr__(self, "regions", regions)

    @property
    def region_ids(self) -> Tuple[str, ...]:
        return tuple(region.region_id for region in self.regions)

    @property
    def num_regions(self) -> int:
        return len(self.regions)

    @property
    def has_continuous_symmetry(self) -> bool:
        return any(isinstance(region.rotation_group, SO2Group) for region in self.regions)

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        source_path: Optional[PathLike] = None,
    ) -> "SymmetryMetadata":
        if not isinstance(value, Mapping):
            raise ValueError("Symmetry metadata root must be an object")
        required = {"version", "object_model_id", "coordinate_frame", "axis", "regions"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Symmetry metadata is missing fields: {sorted(missing)}")
        extra = set(value).difference(required)
        if extra:
            raise ValueError(f"Symmetry metadata has unknown fields: {sorted(extra)}")
        regions_raw = value["regions"]
        if not isinstance(regions_raw, list):
            raise ValueError("regions must be an array")
        return cls(
            version=value["version"],
            object_model_id=value["object_model_id"],
            coordinate_frame=value["coordinate_frame"],
            axis=SymmetryAxis.from_dict(value["axis"]),
            regions=tuple(SymmetryRegion.from_dict(region) for region in regions_raw),
            source_path=str(source_path) if source_path is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "object_model_id": self.object_model_id,
            "coordinate_frame": self.coordinate_frame,
            "axis": self.axis.to_dict(),
            "regions": [region.to_dict() for region in self.regions],
        }


def canonical_object_model_id(object_model_id: str) -> str:
    """Remove a generated scale suffix for sidecar/template ID comparison."""

    value = str(object_model_id).strip()
    return value.split("__scale_", 1)[0]


def object_model_ids_match(metadata_id: str, requested_id: str) -> bool:
    """Match either exact IDs or their base IDs before ``__scale_*`` suffixes."""

    return metadata_id == requested_id or canonical_object_model_id(metadata_id) == canonical_object_model_id(
        requested_id
    )


def load_symmetry_metadata(
    path: Optional[PathLike],
    *,
    expected_object_model_id: Optional[str] = None,
) -> Optional[SymmetryMetadata]:
    """Load a sidecar, returning ``None`` when it is absent.

    Absence is semantically different from a known ``C1`` object: callers must
    use the ``None`` result to disable symmetry supervision.  Existing but
    malformed files raise ``ValueError`` with their path, because silently
    disabling supervision would hide an annotation error.
    """

    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return None
    try:
        with resolved.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read symmetry metadata {resolved}: {exc}") from exc
    try:
        metadata = SymmetryMetadata.from_dict(payload, source_path=resolved)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid symmetry metadata {resolved}: {exc}") from exc
    if expected_object_model_id is not None and not object_model_ids_match(
        metadata.object_model_id, expected_object_model_id
    ):
        raise ValueError(
            f"Symmetry metadata object_model_id {metadata.object_model_id!r} does not "
            f"match requested template {expected_object_model_id!r}"
        )
    return metadata


def symmetry_sidecar_candidates(
    template_path: PathLike,
    *,
    object_model_id: Optional[str] = None,
) -> Tuple[Path, ...]:
    """Return deterministic sibling sidecar candidates, most specific first."""

    template = Path(template_path)
    directory = template if template.is_dir() else template.parent
    full_id = object_model_id or (None if template.is_dir() else template.stem)
    if full_id is None:
        raise ValueError("object_model_id is required when template_path is a directory")
    base_id = canonical_object_model_id(full_id)
    names = [f"{full_id}.symmetry.json"]
    if base_id != full_id:
        names.append(f"{base_id}.symmetry.json")
    return tuple(directory / name for name in names)


def find_symmetry_sidecar(
    template_path: PathLike,
    *,
    object_model_id: Optional[str] = None,
) -> Optional[Path]:
    """Find the first existing conventional sibling sidecar path."""

    return next(
        (
            candidate
            for candidate in symmetry_sidecar_candidates(
                template_path, object_model_id=object_model_id
            )
            if candidate.is_file()
        ),
        None,
    )


# Compatibility alias with an explicit template-level name.
load_template_symmetry = load_symmetry_metadata


__all__ = [
    "SymmetryAxis",
    "SymmetryMetadata",
    "SymmetryRegion",
    "canonical_object_model_id",
    "find_symmetry_sidecar",
    "load_symmetry_metadata",
    "load_template_symmetry",
    "object_model_ids_match",
    "symmetry_sidecar_candidates",
]
