"""Symmetry debug artifact generation backed only by production target math."""

from __future__ import annotations

import colorsys
import csv
import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from symm_template_reg.datasets import FragmentTemplateRegistrationDataset
from symm_template_reg.datasets.template_repository import TemplateRepository, load_ply
from symm_template_reg.models.pose.pose_representation import invert_transform, transform_points
from symm_template_reg.models.symmetry.groups import (
    SO2Group,
    group_angles,
    group_to_dict,
)
from symm_template_reg.models.symmetry.hypothesis_expander import (
    DEFAULT_SO2_NUM_SAMPLES,
    DEFAULT_SO2_VISUALIZATION_SAMPLES,
    place_fragment_for_hypothesis,
    visualization_equivalent_pose_set,
)
from symm_template_reg.models.symmetry.metadata import (
    SymmetryMetadata,
    find_symmetry_sidecar,
    load_symmetry_metadata,
    object_model_ids_match,
)
from symm_template_reg.models.symmetry.region_assignment import (
    RegionPartitionValidation,
    validate_region_partition,
)
from symm_template_reg.models.symmetry.targets import (
    SymmetryTargets,
    build_fragment_symmetry_targets,
    build_symmetry_targets,
)

from .ply import write_colored_ply


REGION_PALETTE = np.asarray(
    [
        [0, 220, 220],
        [230, 40, 220],
        [125, 250, 255],
        [255, 145, 20],
        [130, 225, 80],
        [170, 100, 255],
        [255, 215, 40],
        [70, 150, 255],
    ],
    dtype=np.uint8,
)
LIGHT_GRAY = np.asarray([205, 205, 205], dtype=np.uint8)
INACTIVE_GRAY = np.asarray([150, 150, 150], dtype=np.uint8)
BOUNDARY_COLOR = np.asarray([255, 255, 255], dtype=np.uint8)
AXIS_COLOR = np.asarray([40, 90, 255], dtype=np.uint8)
ORIGIN_COLOR = np.asarray([255, 35, 35], dtype=np.uint8)
REFERENCE_COLOR = np.asarray([35, 230, 70], dtype=np.uint8)
OBSERVED_CAMERA_COLOR = np.asarray([40, 235, 90], dtype=np.uint8)
FRAGMENT_CAMERA_COLOR = np.asarray([245, 245, 245], dtype=np.uint8)
DEFAULT_DEBUG_OUTPUT_ROOT = Path(
    "/home/nikita/disser/fragment-template-registration-lab/output_debug"
)


def _as_numpy(value: Any, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_npy(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, _as_numpy(value))


def create_unique_run_directory(
    output_root: str | Path,
    *,
    timestamp: str | None = None,
) -> Path:
    """Create a timestamped run directory, adding a suffix on collisions."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / f"symmetry_debug_{stamp}"
    for suffix in range(10000):
        candidate = base if suffix == 0 else Path(f"{base}_{suffix:03d}")
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not allocate a unique run directory below {root}")


@dataclass
class GeometryBuilder:
    vertices: list[np.ndarray] = field(default_factory=list)
    colors: list[np.ndarray] = field(default_factory=list)
    faces: list[np.ndarray] = field(default_factory=list)
    _count: int = 0

    def add_points(self, points: Any, colors: Any) -> None:
        values = _as_numpy(points, np.dtype("f4"))
        rgb = _as_numpy(colors, np.dtype("u1"))
        if values.ndim != 2 or values.shape[-1] != 3:
            raise ValueError("points must have shape [N,3]")
        if rgb.shape == (3,):
            rgb = np.broadcast_to(rgb, (len(values), 3)).copy()
        if rgb.shape != (len(values), 3):
            raise ValueError("point colors must have shape [N,3] or [3]")
        self.vertices.append(values)
        self.colors.append(rgb)
        self._count += len(values)

    def add_mesh(self, vertices: Any, faces: Any, colors: Any) -> None:
        values = _as_numpy(vertices, np.dtype("f4"))
        triangles = _as_numpy(faces, np.dtype("i8"))
        rgb = _as_numpy(colors, np.dtype("u1"))
        if triangles.ndim != 2 or triangles.shape[-1] != 3:
            raise ValueError("faces must have shape [F,3]")
        if rgb.shape == (3,):
            rgb = np.broadcast_to(rgb, (len(values), 3)).copy()
        if rgb.shape != (len(values), 3):
            raise ValueError("mesh colors must have shape [V,3] or [3]")
        self.vertices.append(values)
        self.colors.append(rgb)
        self.faces.append(triangles + self._count)
        self._count += len(values)

    def add_face_colored_mesh(
        self, vertices: Any, faces: Any, face_colors: Any
    ) -> None:
        values = _as_numpy(vertices, np.dtype("f4"))
        triangles = _as_numpy(faces, np.dtype("i8"))
        rgb = _as_numpy(face_colors, np.dtype("u1"))
        if rgb.shape != (len(triangles), 3):
            raise ValueError("face_colors must have shape [F,3]")
        expanded = values[triangles].reshape(-1, 3)
        expanded_colors = np.repeat(rgb, 3, axis=0)
        expanded_faces = np.arange(len(expanded), dtype=np.int64).reshape(-1, 3)
        self.add_mesh(expanded, expanded_faces, expanded_colors)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.vertices:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=np.int64),
            )
        return (
            np.concatenate(self.vertices, axis=0),
            np.concatenate(self.colors, axis=0),
            np.concatenate(self.faces, axis=0)
            if self.faces
            else np.empty((0, 3), dtype=np.int64),
        )

    def write(self, path: Path, *, comments: Sequence[str] | None = None) -> None:
        vertices, colors, faces = self.arrays()
        write_colored_ply(path, vertices, colors, faces=faces, comments=comments)


def _orthogonal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = axis / np.linalg.norm(axis)
    helper = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(unit)))]
    first = np.cross(unit, helper)
    first /= np.linalg.norm(first)
    second = np.cross(unit, first)
    second /= np.linalg.norm(second)
    return first, second


def _tube_segment(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: np.ndarray,
    *,
    sides: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.uint8),
        )
    unit = direction / length
    first, second = _orthogonal_basis(unit)
    angles = np.arange(sides, dtype=np.float64) * (2.0 * math.pi / sides)
    ring = radius * (
        np.cos(angles)[:, None] * first[None]
        + np.sin(angles)[:, None] * second[None]
    )
    vertices = np.concatenate((start[None] + ring, end[None] + ring), axis=0)
    faces: list[list[int]] = []
    for index in range(sides):
        following = (index + 1) % sides
        faces.append([index, following, sides + index])
        faces.append([following, sides + following, sides + index])
    return (
        vertices.astype(np.float32),
        np.asarray(faces, dtype=np.int64),
        np.broadcast_to(color.astype(np.uint8), (len(vertices), 3)).copy(),
    )


def _add_polyline_tubes(
    builder: GeometryBuilder,
    points: np.ndarray,
    radius: float,
    color: np.ndarray,
    *,
    closed: bool = False,
) -> None:
    pairs = list(zip(points[:-1], points[1:]))
    if closed and len(points) > 2:
        pairs.append((points[-1], points[0]))
    for start, end in pairs:
        vertices, faces, colors = _tube_segment(start, end, radius, color)
        if len(vertices):
            builder.add_mesh(vertices, faces, colors)


def _template_scale(
    points: np.ndarray, metadata: SymmetryMetadata
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(metadata.axis.direction, dtype=np.float64)
    origin = np.asarray(metadata.axis.origin, dtype=np.float64)
    relative = points.astype(np.float64) - origin
    axial = relative @ axis
    radial_vectors = relative - axial[:, None] * axis[None]
    radial = max(float(np.linalg.norm(radial_vectors, axis=1).max()), 1e-4)
    extent = max(float(np.ptp(points, axis=0).max()), 1e-4)
    first, second = _orthogonal_basis(axis)
    return radial, extent, max(extent * 0.003, 8e-5), axis, first, second


def _add_axis_boundaries_and_origin(
    builder: GeometryBuilder,
    points: np.ndarray,
    metadata: SymmetryMetadata,
    *,
    offset: np.ndarray | None = None,
    include_boundaries: bool = True,
    include_reference: bool = False,
) -> None:
    shift = np.zeros(3, dtype=np.float64) if offset is None else offset.astype(np.float64)
    origin = np.asarray(metadata.axis.origin, dtype=np.float64) + shift
    radial, extent, tube_radius, axis, first, second = _template_scale(points, metadata)
    coordinates = (points.astype(np.float64) - np.asarray(metadata.axis.origin)) @ axis
    lower = float(coordinates.min()) - extent * 0.08
    upper = float(coordinates.max()) + extent * 0.08
    axis_vertices, axis_faces, axis_colors = _tube_segment(
        origin + axis * lower,
        origin + axis * upper,
        tube_radius * 1.2,
        AXIS_COLOR,
    )
    builder.add_mesh(axis_vertices, axis_faces, axis_colors)

    if include_boundaries:
        bounds = [metadata.regions[0].y_min_m] + [region.y_max_m for region in metadata.regions]
        angles = np.arange(72, dtype=np.float64) * (2.0 * math.pi / 72.0)
        for bound in bounds:
            center = origin + axis * bound
            ring = center[None] + radial * 1.10 * (
                np.cos(angles)[:, None] * first[None]
                + np.sin(angles)[:, None] * second[None]
            )
            _add_polyline_tubes(
                builder, ring, tube_radius, BOUNDARY_COLOR, closed=True
            )

    marker_length = extent * 0.055
    for direction in (axis, first, second):
        vertices, faces, colors = _tube_segment(
            origin - direction * marker_length,
            origin + direction * marker_length,
            tube_radius * 1.45,
            ORIGIN_COLOR,
        )
        builder.add_mesh(vertices, faces, colors)
    if include_reference:
        vertices, faces, colors = _tube_segment(
            origin,
            origin + first * radial * 1.25,
            tube_radius * 1.25,
            REFERENCE_COLOR,
        )
        builder.add_mesh(vertices, faces, colors)


def _region_colors(count: int) -> np.ndarray:
    return np.stack([REGION_PALETTE[index % len(REGION_PALETTE)] for index in range(count)])


def _hypothesis_colors(count: int) -> np.ndarray:
    colors = []
    for index in range(count):
        hue = (0.09 + index * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.98)
        colors.append([round(red * 255), round(green * 255), round(blue * 255)])
    return np.asarray(colors, dtype=np.uint8)


def _face_colors(
    face_regions: np.ndarray,
    metadata: SymmetryMetadata,
    active_regions: Sequence[bool] | None = None,
) -> np.ndarray:
    palette = _region_colors(len(metadata.regions))
    if active_regions is None:
        return palette[face_regions]
    active = np.asarray(active_regions, dtype=bool)
    colors = np.broadcast_to(INACTIVE_GRAY, (len(face_regions), 3)).copy()
    for index in range(len(metadata.regions)):
        if active[index]:
            colors[face_regions == index] = palette[index]
    return colors


def _triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces].astype(np.float64)
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=-1,
    )


@dataclass(frozen=True)
class _RegionSplitTemplateMesh:
    vertices: np.ndarray
    faces: np.ndarray
    region_indices: np.ndarray
    source_face_indices: np.ndarray
    split_source_faces: int


def _split_template_at_region_boundaries(
    points: np.ndarray,
    faces: np.ndarray,
    metadata: SymmetryMetadata,
) -> _RegionSplitTemplateMesh:
    """Cut template triangles exactly at every axial symmetry-band plane."""

    template_points = np.asarray(points, dtype=np.float64)
    template_faces = np.asarray(faces, dtype=np.int64)
    origin = np.asarray(metadata.axis.origin, dtype=np.float64)
    axis = np.asarray(metadata.axis.direction, dtype=np.float64)
    output_triangles: list[np.ndarray] = []
    output_regions: list[int] = []
    output_sources: list[int] = []
    split_sources: set[int] = set()

    def coordinate(values: np.ndarray) -> np.ndarray:
        return (values - origin) @ axis

    def clip_polygon(
        polygon: np.ndarray,
        boundary: float,
        *,
        keep_greater: bool,
    ) -> np.ndarray:
        if len(polygon) == 0:
            return polygon
        coordinates = coordinate(polygon)
        inside = (
            coordinates >= boundary - 1e-12
            if keep_greater
            else coordinates <= boundary + 1e-12
        )
        clipped: list[np.ndarray] = []
        for index in range(len(polygon)):
            following = (index + 1) % len(polygon)
            current_point = polygon[index]
            following_point = polygon[following]
            current_inside = bool(inside[index])
            following_inside = bool(inside[following])
            if current_inside:
                clipped.append(current_point)
            if current_inside != following_inside:
                current_coordinate = float(coordinates[index])
                following_coordinate = float(coordinates[following])
                denominator = following_coordinate - current_coordinate
                if abs(denominator) <= 1e-15:
                    continue
                parameter = (boundary - current_coordinate) / denominator
                clipped.append(
                    current_point
                    + np.clip(parameter, 0.0, 1.0)
                    * (following_point - current_point)
                )
        return np.asarray(clipped, dtype=np.float64).reshape(-1, 3)

    for source_face, triangle_indices in enumerate(template_faces):
        triangle = template_points[triangle_indices]
        emitted_before = len(output_triangles)
        for region_index, region in enumerate(metadata.regions):
            polygon = clip_polygon(
                triangle, region.y_min_m, keep_greater=True
            )
            polygon = clip_polygon(
                polygon, region.y_max_m, keep_greater=False
            )
            if len(polygon) < 3:
                continue
            for index in range(1, len(polygon) - 1):
                piece = np.stack((polygon[0], polygon[index], polygon[index + 1]))
                area_vector = np.cross(piece[1] - piece[0], piece[2] - piece[0])
                if float(np.dot(area_vector, area_vector)) <= 1e-24:
                    continue
                output_triangles.append(piece.astype(np.float32))
                output_regions.append(region_index)
                output_sources.append(source_face)
        if len(output_triangles) - emitted_before > 1:
            split_sources.add(source_face)

    triangles = np.asarray(output_triangles, dtype=np.float32)
    vertices = triangles.reshape(-1, 3)
    output_faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return _RegionSplitTemplateMesh(
        vertices=vertices,
        faces=output_faces,
        region_indices=np.asarray(output_regions, dtype=np.int64),
        source_face_indices=np.asarray(output_sources, dtype=np.int64),
        split_source_faces=len(split_sources),
    )


class _TriangleSurfaceIndex:
    """Small dependency-free spatial index for a triangle surface."""

    def __init__(self, triangles: np.ndarray, distance_m: float) -> None:
        self.triangles = np.asarray(triangles, dtype=np.float64)
        self.distance_m = float(distance_m)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3):
            raise ValueError("triangles must have shape [F,3,3]")
        if len(self.triangles) == 0:
            raise ValueError("surface index requires at least one triangle")
        if not np.isfinite(self.triangles).all() or self.distance_m <= 0:
            raise ValueError("surface triangles and distance must be finite and positive")
        self.cell_size = self.distance_m * 2.0
        self.cells: dict[tuple[int, int, int], list[int]] = {}
        lower = np.floor(
            (self.triangles.min(axis=1) - self.distance_m) / self.cell_size
        ).astype(np.int64)
        upper = np.floor(
            (self.triangles.max(axis=1) + self.distance_m) / self.cell_size
        ).astype(np.int64)
        for triangle_index, (minimum, maximum) in enumerate(zip(lower, upper)):
            for x_index in range(int(minimum[0]), int(maximum[0]) + 1):
                for y_index in range(int(minimum[1]), int(maximum[1]) + 1):
                    for z_index in range(int(minimum[2]), int(maximum[2]) + 1):
                        self.cells.setdefault((x_index, y_index, z_index), []).append(
                            triangle_index
                        )

    def contains_projected_surface(self, points: np.ndarray) -> np.ndarray:
        """Test orthogonal projection onto triangle interiors without edge dilation."""

        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("query points must have shape [N,3]")
        result = np.zeros(len(values), dtype=bool)
        keys = np.floor(values / self.cell_size).astype(np.int64)
        threshold_squared = self.distance_m * self.distance_m
        for point_index, (point, key) in enumerate(zip(values, keys)):
            candidates = self.cells.get(tuple(int(value) for value in key))
            if not candidates:
                continue
            triangles = self.triangles[np.asarray(candidates, dtype=np.int64)]
            first = triangles[:, 0]
            edge0 = triangles[:, 1] - first
            edge1 = triangles[:, 2] - first
            relative = point - first
            dot00 = np.einsum("ij,ij->i", edge0, edge0)
            dot01 = np.einsum("ij,ij->i", edge0, edge1)
            dot11 = np.einsum("ij,ij->i", edge1, edge1)
            dot20 = np.einsum("ij,ij->i", relative, edge0)
            dot21 = np.einsum("ij,ij->i", relative, edge1)
            denominator = dot00 * dot11 - dot01 * dot01
            valid = denominator > 1e-24
            coordinate1 = np.divide(
                dot11 * dot20 - dot01 * dot21,
                denominator,
                out=np.zeros_like(denominator),
                where=valid,
            )
            coordinate2 = np.divide(
                dot00 * dot21 - dot01 * dot20,
                denominator,
                out=np.zeros_like(denominator),
                where=valid,
            )
            normal = np.cross(edge0, edge1)
            normal_squared = np.einsum("ij,ij->i", normal, normal)
            plane_numerator = np.einsum("ij,ij->i", relative, normal)
            plane_distance_squared = np.divide(
                plane_numerator * plane_numerator,
                normal_squared,
                out=np.full_like(normal_squared, np.inf),
                where=normal_squared > 1e-24,
            )
            inside = (
                valid
                & (coordinate1 >= -1e-9)
                & (coordinate2 >= -1e-9)
                & (coordinate1 + coordinate2 <= 1.0 + 1e-9)
                & (plane_distance_squared <= threshold_squared)
            )
            result[point_index] = bool(np.any(inside))
        return result


def _project_fragment_shell_to_template(
    *,
    template_points: np.ndarray,
    template_faces: np.ndarray,
    shell_surface: _TriangleSurfaceIndex,
    template_to_fragment_surface: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select template vertices/faces covered by the fragment shell surface."""

    points = np.asarray(template_points, dtype=np.float64)
    faces = np.asarray(template_faces, dtype=np.int64)
    if template_to_fragment_surface is not None:
        transform = np.asarray(template_to_fragment_surface, dtype=np.float64)
        points = points @ transform[:3, :3].T + transform[:3, 3]
    vertex_mask = shell_surface.contains_projected_surface(points)
    centers = points[faces].mean(axis=1)
    face_mask = shell_surface.contains_projected_surface(centers)
    return vertex_mask, face_mask


@dataclass(frozen=True)
class _RefinedTemplateMesh:
    vertices: np.ndarray
    faces: np.ndarray
    source_face_indices: np.ndarray
    projected_faces: np.ndarray
    split_source_faces: int


def _refine_template_at_fragment_boundary(
    *,
    template_points: np.ndarray,
    template_faces: np.ndarray,
    shell_surface: _TriangleSurfaceIndex,
    template_to_fragment_surface: np.ndarray | None,
    boundary_resolution_m: float,
    max_depth: int,
) -> _RefinedTemplateMesh:
    """Split only boundary faces and classify their pieces on the template."""

    points = np.asarray(template_points, dtype=np.float64)
    faces = np.asarray(template_faces, dtype=np.int64)
    if boundary_resolution_m <= 0 or max_depth < 0:
        raise ValueError("boundary refinement parameters must be positive")
    transform = (
        np.eye(4, dtype=np.float64)
        if template_to_fragment_surface is None
        else np.asarray(template_to_fragment_surface, dtype=np.float64)
    )
    cache: dict[bytes, bool] = {}

    def classify(query_points: np.ndarray) -> np.ndarray:
        queries = np.asarray(query_points, dtype=np.float64).reshape(-1, 3)
        result = np.empty(len(queries), dtype=bool)
        missing_points: list[np.ndarray] = []
        missing_indices: list[int] = []
        missing_keys: list[bytes] = []
        for index, query in enumerate(queries):
            key = query.tobytes()
            cached = cache.get(key)
            if cached is None:
                missing_points.append(query)
                missing_indices.append(index)
                missing_keys.append(key)
            else:
                result[index] = cached
        if missing_points:
            local = np.asarray(missing_points, dtype=np.float64)
            transformed = local @ transform[:3, :3].T + transform[:3, 3]
            classified = shell_surface.contains_projected_surface(transformed)
            for index, key, value in zip(missing_indices, missing_keys, classified):
                boolean = bool(value)
                cache[key] = boolean
                result[index] = boolean
        return result

    output_triangles: list[np.ndarray] = []
    output_sources: list[int] = []
    output_projected: list[bool] = []
    split_sources: set[int] = set()

    def emit(triangle: np.ndarray, source_face: int, projected: bool) -> None:
        area_vector = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if float(np.dot(area_vector, area_vector)) <= 1e-24:
            return
        output_triangles.append(np.asarray(triangle, dtype=np.float32))
        output_sources.append(source_face)
        output_projected.append(projected)

    def boundary_point(
        first: np.ndarray, second: np.ndarray, first_state: bool
    ) -> np.ndarray:
        lower = first.copy()
        upper = second.copy()
        for _ in range(18):
            middle = (lower + upper) * 0.5
            if bool(classify(middle[None])[0]) == first_state:
                lower = middle
            else:
                upper = middle
        return (lower + upper) * 0.5

    def emit_clipped(
        triangle: np.ndarray,
        states: np.ndarray,
        source_face: int,
    ) -> None:
        if bool(np.all(states == states[0])):
            emit(triangle, source_face, bool(states[0]))
            return
        polygons: dict[bool, list[np.ndarray]] = {False: [], True: []}
        for index in range(3):
            following = (index + 1) % 3
            state = bool(states[index])
            following_state = bool(states[following])
            polygons[state].append(triangle[index])
            if state != following_state:
                crossing = boundary_point(
                    triangle[index], triangle[following], state
                )
                polygons[False].append(crossing)
                polygons[True].append(crossing)
        for projected, polygon_values in polygons.items():
            if len(polygon_values) < 3:
                continue
            polygon = np.asarray(polygon_values, dtype=np.float64)
            for index in range(1, len(polygon) - 1):
                emit(
                    np.stack((polygon[0], polygon[index], polygon[index + 1])),
                    source_face,
                    projected,
                )

    def refine(triangle: np.ndarray, source_face: int, depth: int) -> None:
        edge_midpoints = np.stack(
            (
                (triangle[0] + triangle[1]) * 0.5,
                (triangle[1] + triangle[2]) * 0.5,
                (triangle[2] + triangle[0]) * 0.5,
            )
        )
        center = triangle.mean(axis=0, keepdims=True)
        states = classify(np.concatenate((triangle, edge_midpoints, center), axis=0))
        if bool(np.all(states == states[0])):
            emit(triangle, source_face, bool(states[0]))
            return
        split_sources.add(source_face)
        edge_lengths = np.linalg.norm(
            triangle[[1, 2, 0]] - triangle[[0, 1, 2]], axis=1
        )
        if depth < max_depth and float(edge_lengths.max()) > boundary_resolution_m:
            first, second, third = triangle
            midpoint01, midpoint12, midpoint20 = edge_midpoints
            refine(np.stack((first, midpoint01, midpoint20)), source_face, depth + 1)
            refine(np.stack((midpoint01, second, midpoint12)), source_face, depth + 1)
            refine(np.stack((midpoint20, midpoint12, third)), source_face, depth + 1)
            refine(np.stack((midpoint01, midpoint12, midpoint20)), source_face, depth + 1)
            return
        vertex_states = states[:3]
        if bool(np.all(vertex_states == vertex_states[0])):
            center_point = center[0]
            for index in range(3):
                subtriangle = np.stack(
                    (triangle[index], triangle[(index + 1) % 3], center_point)
                )
                emit_clipped(
                    subtriangle,
                    classify(subtriangle),
                    source_face,
                )
            return
        emit_clipped(triangle, vertex_states, source_face)

    initial_triangles = points[faces]
    initial_centers = initial_triangles.mean(axis=1)
    vertex_states = classify(points)
    center_states = classify(initial_centers)
    for source_face, triangle_indices in enumerate(faces):
        states = np.concatenate(
            (vertex_states[triangle_indices], center_states[source_face : source_face + 1])
        )
        if bool(np.all(states == states[0])):
            emit(initial_triangles[source_face], source_face, bool(states[0]))
        else:
            refine(initial_triangles[source_face], source_face, 0)

    triangles = np.asarray(output_triangles, dtype=np.float32)
    vertices = triangles.reshape(-1, 3)
    output_faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return _RefinedTemplateMesh(
        vertices=vertices,
        faces=output_faces,
        source_face_indices=np.asarray(output_sources, dtype=np.int64),
        projected_faces=np.asarray(output_projected, dtype=bool),
        split_source_faces=len(split_sources),
    )


def export_template_visualization(
    output_dir: Path,
    *,
    object_model_id: str,
    template_path: Path,
    sidecar_path: Path,
    points_O: np.ndarray,
    faces: np.ndarray,
    metadata: SymmetryMetadata,
    partition: RegionPartitionValidation,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    face_regions = _as_numpy(partition.face_region_indices, np.dtype("i8"))
    vertex_regions = _as_numpy(partition.vertex_region_indices, np.dtype("i8"))
    visual_mesh = _split_template_at_region_boundaries(
        points_O, faces, metadata
    )
    palette = _region_colors(len(metadata.regions))
    colors = palette[visual_mesh.region_indices]

    regions_builder = GeometryBuilder()
    regions_builder.add_face_colored_mesh(
        visual_mesh.vertices, visual_mesh.faces, colors
    )
    regions_builder.write(output_dir / "template_symmetry_regions.ply")

    marker_builder = GeometryBuilder()
    _add_axis_boundaries_and_origin(marker_builder, points_O, metadata)
    marker_builder.write(output_dir / "template_region_boundaries.ply")

    combined = GeometryBuilder()
    combined.add_face_colored_mesh(
        visual_mesh.vertices, visual_mesh.faces, colors
    )
    _add_axis_boundaries_and_origin(combined, points_O, metadata)
    combined.write(output_dir / "template_symmetry_regions_with_boundaries.ply")

    _save_npy(output_dir / "template_face_regions.npy", face_regions)
    _save_npy(output_dir / "template_vertex_regions.npy", vertex_regions)
    _save_npy(
        output_dir / "template_visual_face_regions.npy",
        visual_mesh.region_indices,
    )
    _save_npy(
        output_dir / "template_visual_source_face_indices.npy",
        visual_mesh.source_face_indices,
    )
    original_areas = _triangle_areas(points_O, faces)
    visual_areas = _triangle_areas(visual_mesh.vertices, visual_mesh.faces)
    region_summaries = []
    for index, region in enumerate(metadata.regions):
        region_summaries.append(
            {
                "region_id": region.region_id,
                "y_min_m": region.y_min_m,
                "y_max_m": region.y_max_m,
                "group": region.rotation_group.name,
                "vertices": int((vertex_regions == index).sum()),
                "faces": int((face_regions == index).sum()),
                "visual_faces_after_boundary_split": int(
                    (visual_mesh.region_indices == index).sum()
                ),
                "surface_area_m2": float(
                    visual_areas[visual_mesh.region_indices == index].sum()
                ),
                "color_rgb": palette[index].tolist(),
            }
        )
    summary = {
        "object_model_id": object_model_id,
        "template_path": str(template_path),
        "sidecar_path": str(sidecar_path),
        "axis": metadata.axis.to_dict(),
        "bbox_y_min_m": partition.bbox_axis_min_m,
        "bbox_y_max_m": partition.bbox_axis_max_m,
        "coverage_ok": partition.coverage_ok,
        "interval_policy": "all regions [y_min,y_max), final region [y_min,y_max]",
        "visual_boundary_policy": "template triangles are cut exactly at every internal axial region plane",
        "original_template_faces": int(len(faces)),
        "visual_template_faces": int(len(visual_mesh.faces)),
        "boundary_split_source_faces": visual_mesh.split_source_faces,
        "original_surface_area_m2": float(original_areas.sum()),
        "visual_surface_area_m2": float(visual_areas.sum()),
        "regions": region_summaries,
        "unassigned_vertices": partition.unassigned_vertices,
        "unassigned_faces": partition.unassigned_faces,
        "overlap_vertices": partition.overlap_vertices,
        "overlap_faces": partition.overlap_faces,
        "warnings": [],
    }
    legend = {
        "regions": [
            {
                "region_id": region.region_id,
                "group": region.rotation_group.name,
                "color_rgb": palette[index].tolist(),
            }
            for index, region in enumerate(metadata.regions)
        ],
        "boundary_rings_rgb": BOUNDARY_COLOR.tolist(),
        "symmetry_axis_rgb": AXIS_COLOR.tolist(),
        "origin_rgb": ORIGIN_COLOR.tolist(),
    }
    _write_json(output_dir / "template_symmetry_summary.json", summary)
    _write_json(output_dir / "template_symmetry_legend.json", legend)
    return summary


def gallery_offsets(
    count: int,
    *,
    columns: int,
    spacing_m: float,
) -> np.ndarray:
    """Return deterministic XZ gallery offsets without modifying local geometry."""

    if count < 0 or columns < 1 or spacing_m <= 0:
        raise ValueError("invalid gallery layout")
    return np.asarray(
        [
            [(index % columns) * spacing_m, 0.0, (index // columns) * spacing_m]
            for index in range(count)
        ],
        dtype=np.float32,
    )


def camera_points_in_hypothesis_object(points_C: Tensor, T_C_from_O: Tensor) -> Tensor:
    """Apply the exact inverse generated pose used by gallery construction."""

    return transform_points(invert_transform(T_C_from_O), points_C)


def apply_gallery_offset(points: Tensor, offset: Tensor | Sequence[float]) -> Tensor:
    return points + torch.as_tensor(offset, dtype=points.dtype, device=points.device)


def _resolve_template_and_sidecar(
    dataset_root: Path,
    object_model_id: str,
    template_mesh: str,
    symmetry_sidecar: str,
) -> tuple[Path, Path, np.ndarray, np.ndarray, SymmetryMetadata]:
    if template_mesh == "auto":
        repository = TemplateRepository(
            dataset_root / "models", fine_points=None, coarse_points=None
        )
        template = repository.get(object_model_id)
        template_path = Path(str(template["mesh_path"])).resolve()
        points = _as_numpy(template["points_O"], np.dtype("f4"))
        if template["faces"] is None:
            raise ValueError("template mesh has no triangular faces")
        faces = _as_numpy(template["faces"], np.dtype("i8"))
    else:
        template_path = Path(template_mesh).expanduser().resolve()
        mesh = load_ply(template_path)
        points = _as_numpy(mesh["points"], np.dtype("f4"))
        if mesh["faces"] is None:
            raise ValueError("template mesh has no triangular faces")
        faces = _as_numpy(mesh["faces"], np.dtype("i8"))

    if symmetry_sidecar == "auto":
        found = find_symmetry_sidecar(template_path, object_model_id=template_path.stem)
        if found is None:
            raise FileNotFoundError(f"no symmetry sidecar found beside {template_path}")
        sidecar_path = found.resolve()
    else:
        sidecar_path = Path(symmetry_sidecar).expanduser().resolve()
    metadata = load_symmetry_metadata(
        sidecar_path, expected_object_model_id=template_path.stem
    )
    if metadata is None:
        raise FileNotFoundError(sidecar_path)
    if not object_model_ids_match(metadata.object_model_id, object_model_id):
        raise ValueError(
            f"sidecar object {metadata.object_model_id!r} does not match requested "
            f"object {object_model_id!r}"
        )
    return template_path, sidecar_path, points, faces, metadata


def _parse_sample_indices(values: Sequence[str] | None) -> list[int]:
    result: list[int] = []
    for value in values or ():
        for item in value.split(","):
            if item.strip():
                result.append(int(item))
    return result


def select_dataset_indices(
    dataset: FragmentTemplateRegistrationDataset,
    *,
    object_model_id: str,
    sample_index: int | None = None,
    sample_indices: Sequence[str] | None = None,
    scene_id: str | None = None,
    frame_id: int | None = None,
    fragment_id: int | None = None,
    all_fragments: bool = False,
    max_samples: int = 16,
) -> list[int]:
    explicit = _parse_sample_indices(sample_indices)
    if sample_index is not None:
        explicit.append(sample_index)
    if explicit:
        selected = explicit
    else:
        selected = [
            index
            for index, record in enumerate(dataset.sample_records)
            if object_model_ids_match(record.object_model_id, object_model_id)
            and (scene_id is None or record.scene_id == scene_id)
            and (frame_id is None or record.frame_id == frame_id)
            and (fragment_id is None or record.fragment_id == fragment_id)
        ]
        if not all_fragments and selected:
            selected = selected[:1]
    unique: list[int] = []
    for index in selected:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"sample index {index} outside [0,{len(dataset)})")
        if index not in unique:
            unique.append(index)
    if not unique:
        raise ValueError("sample selection is empty")
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    return unique[:max_samples]


def _fragment_mesh_for_sample(
    dataset: FragmentTemplateRegistrationDataset,
    index: int,
) -> tuple[np.ndarray, np.ndarray, Path] | None:
    record = dataset.sample_records[index]
    path = (
        record.visible_points_path.parent.parent
        / "fragments"
        / f"fragment_{record.fragment_id:04d}.ply"
    )
    if not path.is_file():
        return None
    mesh = load_ply(path)
    if mesh["faces"] is None:
        return None
    return (
        _as_numpy(mesh["points"], np.dtype("f4")),
        _as_numpy(mesh["faces"], np.dtype("i8")),
        path,
    )


def _target_parity(sample: dict[str, Any], targets: SymmetryTargets) -> None:
    dataset_active = sample["gt"].get("active_symmetry_regions")
    dataset_group = sample["gt"].get("effective_symmetry_group")
    dataset_poses = sample["gt"].get("equivalent_T_C_from_O")
    if not isinstance(dataset_active, Tensor) or not torch.equal(
        dataset_active, targets.active_regions
    ):
        raise RuntimeError("Dataset/debug active-region target parity failed")
    if dataset_group != group_to_dict(targets.effective_group):
        raise RuntimeError("Dataset/debug effective-group target parity failed")
    if not isinstance(dataset_poses, Tensor) or not torch.allclose(
        dataset_poses, targets.equivalent_poses, atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError("Dataset/debug pose-hypothesis target parity failed")


def _gallery_pose_set(
    targets: SymmetryTargets,
    T_C_from_O: Tensor,
    metadata: SymmetryMetadata,
    so2_visualization_samples: int,
):
    if not isinstance(targets.effective_group, SO2Group):
        return targets.equivalent_pose_set
    return visualization_equivalent_pose_set(
        T_C_from_O,
        metadata,
        effective_group=targets.effective_group,
        so2_visualization_samples=so2_visualization_samples,
    )


def export_sample_visualization(
    output_dir: Path,
    *,
    sample: dict[str, Any],
    dataset: FragmentTemplateRegistrationDataset,
    dataset_index: int,
    points_O: np.ndarray,
    faces: np.ndarray,
    metadata: SymmetryMetadata,
    partition: RegionPartitionValidation,
    so2_visualization_samples: int,
    gallery_columns: int,
    gallery_spacing_scale: float,
    include_fragment_mesh: bool,
    include_observed_points: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    observed_C = sample["observed"]["points_C"]
    corresponding_O = sample["gt"]["points_O_corresponding"]
    T_C_from_O = sample["gt"]["T_C_from_O"]
    if not isinstance(corresponding_O, Tensor):
        raise ValueError("sample has no corresponding object-frame points")
    sample_metadata = sample["template"].get("symmetry_metadata")
    if not isinstance(sample_metadata, SymmetryMetadata):
        raise ValueError("Dataset sample has no symmetry metadata")
    if sample_metadata.to_dict() != metadata.to_dict():
        raise ValueError("selected sidecar differs from Dataset production sidecar")

    targets = build_symmetry_targets(corresponding_O, T_C_from_O, metadata)
    _target_parity(sample, targets)
    gallery_pose_set = _gallery_pose_set(
        targets, T_C_from_O, metadata, so2_visualization_samples
    )
    pose_count = gallery_pose_set.num_hypotheses
    hypothesis_colors = _hypothesis_colors(pose_count)
    region_palette = _region_colors(len(metadata.regions))
    face_regions = _as_numpy(partition.face_region_indices, np.dtype("i8"))
    active = _as_numpy(targets.active_regions, np.dtype("bool"))
    active_face_colors = _face_colors(face_regions, metadata, active)

    active_builder = GeometryBuilder()
    active_builder.add_face_colored_mesh(points_O, faces, active_face_colors)
    _add_axis_boundaries_and_origin(active_builder, points_O, metadata)
    active_builder.write(output_dir / "active_regions_on_template.ply")

    point_indices = _as_numpy(targets.point_region_indices, np.dtype("i8"))
    point_colors = np.broadcast_to(ORIGIN_COLOR, (len(point_indices), 3)).copy()
    assigned = point_indices >= 0
    point_colors[assigned] = region_palette[point_indices[assigned]]
    write_colored_ply(
        output_dir / "observed_point_regions.ply",
        corresponding_O,
        point_colors,
    )

    template_extent = max(float(np.ptp(points_O, axis=0).max()), 1e-4)
    offsets = gallery_offsets(
        pose_count,
        columns=gallery_columns,
        spacing_m=template_extent * gallery_spacing_scale,
    )
    angles = group_angles(
        targets.effective_group,
        so2_num_samples=pose_count if isinstance(targets.effective_group, SO2Group) else None,
        dtype=T_C_from_O.dtype,
        device=T_C_from_O.device,
    )
    gallery = GeometryBuilder()
    fragment_gallery = GeometryBuilder()
    fragment_mesh = _fragment_mesh_for_sample(dataset, dataset_index) if include_fragment_mesh else None
    fragment_vertices_C: Tensor | None = None
    fragment_faces: np.ndarray | None = None
    fragment_path: Path | None = None
    warnings: list[str] = []
    if fragment_mesh is not None:
        fragment_vertices_F_np, fragment_faces, fragment_path = fragment_mesh
        T_C_from_F = sample["gt"].get("T_C_from_F")
        if not isinstance(T_C_from_F, Tensor):
            warnings.append("fragment mesh omitted: Dataset has no T_C_from_F")
            fragment_mesh = None
        else:
            fragment_vertices_C = transform_points(
                T_C_from_F, torch.from_numpy(fragment_vertices_F_np).to(T_C_from_F)
            )
    elif include_fragment_mesh:
        warnings.append("fragment mesh requested but not found")

    hypothesis_entries: list[dict[str, Any]] = []
    for index, pose in enumerate(gallery_pose_set.poses):
        offset = offsets[index]
        gallery.add_face_colored_mesh(
            points_O + offset,
            faces,
            active_face_colors,
        )
        if include_observed_points:
            observed_O_hyp = camera_points_in_hypothesis_object(observed_C, pose)
            observed_gallery = apply_gallery_offset(observed_O_hyp, offset)
            gallery.add_points(observed_gallery, hypothesis_colors[index])
        if fragment_mesh is not None and fragment_vertices_C is not None and fragment_faces is not None:
            fragment_O_hyp = camera_points_in_hypothesis_object(fragment_vertices_C, pose)
            fragment_gallery_points = apply_gallery_offset(fragment_O_hyp, offset)
            gallery.add_mesh(fragment_gallery_points, fragment_faces, hypothesis_colors[index])
            fragment_gallery.add_mesh(
                fragment_gallery_points, fragment_faces, hypothesis_colors[index]
            )
        _add_axis_boundaries_and_origin(
            gallery,
            points_O,
            metadata,
            offset=offset,
            include_boundaries=False,
            include_reference=True,
        )
        group_name = targets.effective_group.name
        group_element = (
            f"SO2:sample={index}"
            if isinstance(targets.effective_group, SO2Group)
            else f"{group_name}:k={index}"
        )
        hypothesis_entries.append(
            {
                "hypothesis_index": index,
                "group_element": group_element,
                "angle_deg": float(torch.rad2deg(angles[index])),
                "color_rgb": hypothesis_colors[index].tolist(),
                "gallery_offset_m": offset.tolist(),
                "T_C_from_O": pose.detach().cpu().tolist(),
                "active_regions": [
                    region.region_id
                    for region, is_active in zip(metadata.regions, active.tolist())
                    if is_active
                ],
                "effective_group": group_name,
            }
        )
    gallery.write(output_dir / "gt_hypotheses_gallery.ply")
    if fragment_gallery.vertices:
        fragment_gallery.write(output_dir / "optional_fragment_mesh_hypotheses.ply")

    camera_builder = GeometryBuilder()
    if include_observed_points:
        camera_builder.add_points(observed_C, OBSERVED_CAMERA_COLOR)
    if fragment_mesh is not None and fragment_vertices_C is not None and fragment_faces is not None:
        camera_builder.add_mesh(fragment_vertices_C, fragment_faces, FRAGMENT_CAMERA_COLOR)
    sample_stride = max(1, math.ceil(len(points_O) / 1200))
    sparse_template = torch.from_numpy(points_O[::sample_stride]).to(T_C_from_O)
    axis = torch.as_tensor(metadata.axis.direction, dtype=T_C_from_O.dtype)
    origin = torch.as_tensor(metadata.axis.origin, dtype=T_C_from_O.dtype)
    coordinates = (torch.from_numpy(points_O).to(T_C_from_O) - origin) @ axis
    lower, upper = float(coordinates.min()), float(coordinates.max())
    tube_radius = max(template_extent * 0.0025, 8e-5)
    for index, pose in enumerate(gallery_pose_set.poses):
        template_C = transform_points(pose, sparse_template)
        camera_builder.add_points(template_C, hypothesis_colors[index])
        axis_endpoints_O = torch.stack((origin + axis * lower, origin + axis * upper))
        axis_endpoints_C = transform_points(pose, axis_endpoints_O).detach().cpu().numpy()
        vertices, axis_faces, colors = _tube_segment(
            axis_endpoints_C[0], axis_endpoints_C[1], tube_radius, hypothesis_colors[index]
        )
        camera_builder.add_mesh(vertices, axis_faces, colors)
    camera_builder.write(output_dir / "gt_hypotheses_camera_frame.ply")

    _write_json(output_dir / "hypothesis_index.json", {"hypotheses": hypothesis_entries})
    region_counts = {
        region.region_id: int(targets.region_point_counts[index])
        for index, region in enumerate(metadata.regions)
    }
    active_region_ids = [
        region.region_id
        for region, enabled in zip(metadata.regions, active.tolist())
        if enabled
    ]
    active_groups = [
        region.rotation_group.name
        for region, enabled in zip(metadata.regions, active.tolist())
        if enabled
    ]
    continuous = isinstance(targets.effective_group, SO2Group)
    summary = {
        "sample_id": sample["sample_id"],
        "dataset_index": dataset_index,
        "scene_id": sample["scene_id"],
        "frame_id": int(sample["frame_id"]),
        "fragment_id": int(sample["fragment_id"]),
        "num_observed_points": int(len(observed_C)),
        "region_point_counts": region_counts,
        "unassigned_observed_points": int((point_indices < 0).sum()),
        "active_regions": active_region_ids,
        "active_groups": active_groups,
        "effective_group": group_to_dict(targets.effective_group),
        "num_training_hypotheses": targets.equivalent_pose_set.num_hypotheses,
        "num_gallery_hypotheses": gallery_pose_set.num_hypotheses,
        "training_target_type": targets.training_target_type,
        "training_pose_set_exhaustive": targets.equivalent_pose_set.exhaustive,
        "gallery_is_finite_visualization_only": continuous,
        "visualization_samples": gallery_pose_set.num_hypotheses,
        "hypotheses": [
            {
                "index": entry["hypothesis_index"],
                "angle_deg": entry["angle_deg"],
                "T_C_from_O": entry["T_C_from_O"],
            }
            for entry in hypothesis_entries
        ],
        "camera_frame_hypotheses_may_overlap": True,
        "fragment_mesh_path": str(fragment_path) if fragment_path else None,
        "production_function_sources": {
            "target_builder": "symm_template_reg.models.symmetry.targets.build_symmetry_targets",
            "region_assignment": "symm_template_reg.models.symmetry.region_assignment.assign_symmetry_regions",
            "group_intersection": "symm_template_reg.models.symmetry.groups.intersect_rotation_groups",
            "hypothesis_expansion": "symm_template_reg.models.symmetry.hypothesis_expander.equivalent_gt_pose_set",
            "camera_to_hypothesis_object": "symm_template_reg.models.pose.pose_representation.invert_transform/transform_points",
        },
        "dataset_debug_target_parity": True,
        "warnings": warnings,
    }
    _write_json(output_dir / "sample_summary.json", summary)
    return summary


def _math_semantics(so2_visualization_samples: int) -> dict[str, Any]:
    return {
        "region_interval_policy": {
            "non_final": "[y_min_m, y_max_m)",
            "final": "[y_min_m, y_max_m]",
            "production_source": "symm_template_reg.models.symmetry.region_assignment.assign_symmetry_regions",
        },
        "effective_group": {
            "operation": "intersection of all active region groups",
            "production_source": "symm_template_reg.models.symmetry.groups.intersect_rotation_groups",
        },
        "SO2": {
            "training_target_type": "continuous_analytic",
            "loss_semantics": "twist ignored; axis swing and transformed axis-origin translation are penalized",
            "analytic_loss_source": "symm_template_reg.models.losses.symmetry_pose_loss.SymmetryPoseLoss",
            "dataset_set_prediction_support_samples": DEFAULT_SO2_NUM_SAMPLES,
            "dataset_finite_support_is_exhaustive": False,
            "gallery_is_finite_visualization_only": True,
            "visualization_samples": so2_visualization_samples,
            "visualization_sampler_source": "symm_template_reg.models.symmetry.hypothesis_expander.visualization_equivalent_pose_set",
        },
        "pose_convention": "T_C_from_O_hypothesis = T_C_from_O @ S_O; gallery observed points use inverse(T_C_from_O_hypothesis) @ P_C",
    }


def run_symmetry_debug(
    *,
    dataset_root: str | Path,
    object_model_id: str,
    template_mesh: str = "auto",
    symmetry_sidecar: str = "auto",
    mode: str = "all",
    sample_index: int | None = None,
    sample_indices: Sequence[str] | None = None,
    scene_id: str | None = None,
    frame_id: int | None = None,
    fragment_id: int | None = None,
    all_fragments: bool = False,
    max_samples: int = 16,
    so2_visualization_samples: int = DEFAULT_SO2_VISUALIZATION_SAMPLES,
    gallery_columns: int = 4,
    gallery_spacing_scale: float = 2.5,
    include_fragment_mesh: bool = False,
    include_observed_points: bool = True,
    output_root: str | Path = "output_debug",
    timestamp: str | None = None,
) -> Path:
    if mode not in {"template", "samples", "all"}:
        raise ValueError("mode must be template, samples, or all")
    if so2_visualization_samples < 1 or gallery_columns < 1 or gallery_spacing_scale <= 0:
        raise ValueError("visualization sample/layout parameters must be positive")
    root = Path(dataset_root).expanduser().resolve()
    template_path, sidecar_path, points_O, faces, metadata = _resolve_template_and_sidecar(
        root, object_model_id, template_mesh, symmetry_sidecar
    )
    partition = validate_region_partition(
        torch.from_numpy(points_O),
        torch.from_numpy(faces),
        metadata,
        coverage_tolerance_m=1e-6,
    )
    run_dir = create_unique_run_directory(output_root, timestamp=timestamp)
    _write_json(run_dir / "math_semantics.json", _math_semantics(so2_visualization_samples))

    template_summary = None
    if mode in {"template", "all"}:
        template_summary = export_template_visualization(
            run_dir / "template",
            object_model_id=object_model_id,
            template_path=template_path,
            sidecar_path=sidecar_path,
            points_O=points_O,
            faces=faces,
            metadata=metadata,
            partition=partition,
        )

    sample_summaries: list[dict[str, Any]] = []
    if mode in {"samples", "all"}:
        dataset = FragmentTemplateRegistrationDataset(
            root,
            observed_policy="farthest_point_up_to_max",
            min_observed_points=128,
            max_observed_points=4096,
            template_fine_points=2048,
            template_coarse_points=512,
        )
        indices = select_dataset_indices(
            dataset,
            object_model_id=object_model_id,
            sample_index=sample_index,
            sample_indices=sample_indices,
            scene_id=scene_id,
            frame_id=frame_id,
            fragment_id=fragment_id,
            all_fragments=all_fragments,
            max_samples=max_samples,
        )
        samples_root = run_dir / "samples"
        samples_root.mkdir(parents=True, exist_ok=False)
        for index in indices:
            sample = dataset[index]
            sample_name = (
                f"{sample['scene_id']}_frame_{int(sample['frame_id']):06d}_"
                f"fragment_{int(sample['fragment_id']):04d}"
            )
            sample_summaries.append(
                export_sample_visualization(
                    samples_root / sample_name,
                    sample=sample,
                    dataset=dataset,
                    dataset_index=index,
                    points_O=points_O,
                    faces=faces,
                    metadata=metadata,
                    partition=partition,
                    so2_visualization_samples=so2_visualization_samples,
                    gallery_columns=gallery_columns,
                    gallery_spacing_scale=gallery_spacing_scale,
                    include_fragment_mesh=include_fragment_mesh,
                    include_observed_points=include_observed_points,
                )
            )

    index_path = run_dir / "samples_index.csv"
    with index_path.open("x", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "sample_id",
            "scene_id",
            "frame_id",
            "fragment_id",
            "num_observed_points",
            "active_regions",
            "effective_group",
            "num_hypotheses",
            "gallery_ply",
            "camera_frame_ply",
            "summary_json",
            "warnings",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for summary in sample_summaries:
            sample_name = (
                f"{summary['scene_id']}_frame_{summary['frame_id']:06d}_"
                f"fragment_{summary['fragment_id']:04d}"
            )
            relative = Path("samples") / sample_name
            writer.writerow(
                {
                    "sample_id": summary["sample_id"],
                    "scene_id": summary["scene_id"],
                    "frame_id": summary["frame_id"],
                    "fragment_id": summary["fragment_id"],
                    "num_observed_points": summary["num_observed_points"],
                    "active_regions": ";".join(summary["active_regions"]),
                    "effective_group": json.dumps(summary["effective_group"], separators=(",", ":")),
                    "num_hypotheses": summary["num_gallery_hypotheses"],
                    "gallery_ply": str(relative / "gt_hypotheses_gallery.ply"),
                    "camera_frame_ply": str(relative / "gt_hypotheses_camera_frame.ply"),
                    "summary_json": str(relative / "sample_summary.json"),
                    "warnings": ";".join(summary["warnings"]),
                }
            )

    run_summary = {
        "run_directory": str(run_dir),
        "dataset_root": str(root),
        "object_model_id": object_model_id,
        "template_path": str(template_path),
        "sidecar_path": str(sidecar_path),
        "mode": mode,
        "template_generated": template_summary is not None,
        "num_samples": len(sample_summaries),
        "samples": [
            {
                "sample_id": summary["sample_id"],
                "active_regions": summary["active_regions"],
                "effective_group": summary["effective_group"],
                "num_hypotheses": summary["num_gallery_hypotheses"],
            }
            for summary in sample_summaries
        ],
        "source_data_modified": False,
        "recommended_open_order": [
            "template/template_symmetry_regions_with_boundaries.ply",
            "samples/<sample>/active_regions_on_template.ply",
            "samples/<sample>/gt_hypotheses_gallery.ply",
        ],
    }
    _write_json(run_dir / "run_summary.json", run_summary)
    return run_dir


@dataclass(frozen=True)
class AnnotatedFragmentMesh:
    """One fragment mesh resolved from the scene-level annotation contract."""

    scene_id: str
    fragment_id: int
    fragment_key: str
    annotation_path: Path
    mesh_path: Path
    coordinate_frame: str
    units: str
    T_O_from_F: np.ndarray
    points_F: np.ndarray
    points_O: np.ndarray
    faces: np.ndarray
    face_labels_path: Path
    face_labels: np.ndarray


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def _validate_rigid_transform(value: Any, field_name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{field_name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7, rtol=0.0):
        raise ValueError(f"{field_name} must end with [0,0,0,1]")
    rotation = transform[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"{field_name} rotation determinant is {determinant:.9g}, not 1")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0):
        raise ValueError(f"{field_name} rotation is not orthonormal")
    return transform


def _annotation_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Fragment annotation audit",
        "",
        f"Dataset: `{audit['dataset_root']}`",
        "",
        "Contract: fragment PLY vertices are in local frame `F`; each selected entry "
        "must provide an explicit `T_O_from_F`, which is applied before region assignment.",
        "Mesh paths are scene-relative; the annotated object-model path is relative to "
        "`fragments/fragment_annotations.json`.",
        "",
        "| scene | fragment | mesh frame | transform | vertices | faces | status |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for scene in audit["scenes"]:
        for fragment in scene.get("fragments", []):
            lines.append(
                "| {scene} | {fragment_id} | {frame} | {transform} | {vertices} | "
                "{faces} | {status} |".format(
                    scene=scene["scene_id"],
                    fragment_id=fragment.get("fragment_id", ""),
                    frame=fragment.get("fragment_coordinate_frame", ""),
                    transform=fragment.get("transform_to_object", ""),
                    vertices=fragment.get("num_vertices_actual", ""),
                    faces=fragment.get("num_faces_actual", ""),
                    status=fragment.get("status", ""),
                )
            )
    lines.extend(
        [
            "",
            f"Selected entries: {audit['totals']['selected_entries']}",
            f"Contract-valid entries: {audit['totals']['contract_valid']}",
            f"Contract errors: {audit['totals']['annotation_contract_errors']}",
            "",
        ]
    )
    return "\n".join(lines)


def audit_fragment_annotations(
    dataset_root: str | Path,
    *,
    scene_ids: Sequence[str],
    object_model_id: str,
    template_path: Path,
    fragment_ids: Sequence[int] | None = None,
    max_fragments_per_scene: int | None = None,
    all_annotated_fragments: bool = True,
) -> tuple[dict[str, Any], list[AnnotatedFragmentMesh]]:
    """Read only the three scene annotation files and validate their mesh joins."""

    root = Path(dataset_root).expanduser().resolve()
    requested_ids = None if fragment_ids is None else {int(value) for value in fragment_ids}
    if not all_annotated_fragments and requested_ids is None:
        raise ValueError("select --all-annotated-fragments or provide --fragment-ids")
    if max_fragments_per_scene is not None and int(max_fragments_per_scene) < 1:
        raise ValueError("max_fragments_per_scene must be positive")

    audit_scenes: list[dict[str, Any]] = []
    valid_fragments: list[AnnotatedFragmentMesh] = []
    selected_entries = 0
    contract_errors = 0
    for scene_id in scene_ids:
        scene_root = root / scene_id
        annotation_path = scene_root / "fragments" / "fragment_annotations.json"
        scene_audit: dict[str, Any] = {
            "scene_id": scene_id,
            "fragment_annotation_path": str(annotation_path),
            "status": "ok",
            "fragments": [],
        }
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("annotation root must be an object")
            if not object_model_ids_match(str(payload.get("object_id", "")), object_model_id):
                raise ValueError(
                    f"object_id {payload.get('object_id')!r} does not match {object_model_id!r}"
                )
            coordinate_systems = payload.get("coordinate_systems")
            if not isinstance(coordinate_systems, dict):
                raise ValueError("coordinate_systems must be an object")
            if coordinate_systems.get("object") != "O" or coordinate_systems.get("fragment_prefix") != "F":
                raise ValueError("coordinate_systems must declare object=O and fragment_prefix=F")
            object_model_rel = payload.get("object_model")
            if not isinstance(object_model_rel, str):
                raise ValueError("object_model must be a relative path string")
            annotated_model = (annotation_path.parent / object_model_rel).resolve()
            if annotated_model != template_path.resolve():
                raise ValueError(
                    f"object_model resolves to {annotated_model}, expected {template_path.resolve()}"
                )
            geometry = payload.get("geometry")
            if not isinstance(geometry, dict):
                raise ValueError("geometry must be an object")
            if not bool(geometry.get("fragment_output_scale_baked_to_mesh")):
                raise ValueError("fragment_output_scale_baked_to_mesh must be true")
            scene_audit.update(
                {
                    "object_id": payload.get("object_id"),
                    "object_model_annotation": object_model_rel,
                    "object_model_resolved": str(annotated_model),
                    "coordinate_systems": coordinate_systems,
                    "geometry": geometry,
                    "mesh_coordinate_contract": "fragment PLY vertices are F; T_O_from_F maps F to O",
                    "units": "scene_unit (meters for this dataset)",
                }
            )
            entries = payload.get("fragments")
            if not isinstance(entries, list):
                raise ValueError("fragments must be an array")
            seen_ids: set[int] = set()
            selected_in_scene = 0
            for entry in entries:
                if not isinstance(entry, dict) or "fragment_id" not in entry:
                    continue
                fragment_id = int(entry["fragment_id"])
                if fragment_id in seen_ids:
                    raise ValueError(f"duplicate fragment_id {fragment_id}")
                seen_ids.add(fragment_id)
                if entry.get("annotation_status") != "annotated":
                    continue
                if requested_ids is not None and fragment_id not in requested_ids:
                    continue
                if (
                    max_fragments_per_scene is not None
                    and selected_in_scene >= int(max_fragments_per_scene)
                ):
                    continue
                selected_in_scene += 1
                selected_entries += 1
                name = str(entry.get("name", f"fragment_{fragment_id:04d}"))
                fragment_audit: dict[str, Any] = {
                    "fragment_id": fragment_id,
                    "fragment_key": f"{scene_id}:{name}",
                    "fragment_annotation_path": str(annotation_path),
                    "fragment_coordinate_frame": "F",
                    "transform_to_object": "T_O_from_F",
                    "status": "ok",
                    "warnings": [],
                }
                try:
                    expected_name = f"fragment_{fragment_id:04d}"
                    if name != expected_name:
                        raise ValueError(f"name {name!r} does not match {expected_name!r}")
                    mesh_rel = entry.get("mesh")
                    if not isinstance(mesh_rel, str):
                        raise ValueError("mesh must be a scene-relative path string")
                    mesh_path = (scene_root / mesh_rel).resolve()
                    if mesh_path.name != f"{expected_name}.ply":
                        raise ValueError("mesh basename does not match fragment_id")
                    if not mesh_path.is_file():
                        raise FileNotFoundError(mesh_path)
                    transform = _validate_rigid_transform(
                        entry.get("T_O_from_F"), "T_O_from_F"
                    )
                    mesh = load_ply(mesh_path)
                    points_F = np.asarray(mesh["points"], dtype=np.float32)
                    faces_value = mesh.get("faces")
                    if faces_value is None:
                        raise ValueError("fragment mesh contains no faces")
                    faces = np.asarray(faces_value, dtype=np.int64)
                    if len(points_F) != int(entry.get("num_vertices", -1)):
                        raise ValueError("PLY vertex count differs from annotation")
                    if len(faces) != int(entry.get("num_faces", -1)):
                        raise ValueError("PLY face count differs from annotation")
                    face_labels_rel = entry.get("face_labels")
                    if not isinstance(face_labels_rel, str):
                        raise ValueError("face_labels must be a scene-relative path string")
                    face_labels_path = (scene_root / face_labels_rel).resolve()
                    face_labels = np.asarray(np.load(face_labels_path), dtype=np.uint8)
                    if face_labels.shape != (len(faces),):
                        raise ValueError("face-label count differs from PLY face count")
                    unexpected_labels = np.setdiff1d(
                        np.unique(face_labels), np.asarray([0, 1, 255], dtype=np.uint8)
                    )
                    if len(unexpected_labels):
                        raise ValueError(
                            f"unsupported face labels: {unexpected_labels.tolist()}"
                        )
                    if not bool(np.any(face_labels == 0)):
                        raise ValueError("fragment has no shell faces to project")
                    points_O = (
                        points_F.astype(np.float64) @ transform[:3, :3].T
                        + transform[:3, 3]
                    ).astype(np.float32)
                    fragment_audit.update(
                        {
                            "fragment_mesh_path": str(mesh_path),
                            "mesh_path_base": "scene directory",
                            "num_vertices_annotated": int(entry["num_vertices"]),
                            "num_vertices_actual": int(len(points_F)),
                            "num_faces_annotated": int(entry["num_faces"]),
                            "num_faces_actual": int(len(faces)),
                            "face_labels_path": str(face_labels_path),
                            "num_shell_faces": int(np.count_nonzero(face_labels == 0)),
                            "num_fracture_faces": int(np.count_nonzero(face_labels == 1)),
                            "num_unknown_faces": int(np.count_nonzero(face_labels == 255)),
                            "T_O_from_F": transform.tolist(),
                            "rotation_determinant": float(np.linalg.det(transform[:3, :3])),
                            "scale_baked_to_mesh": True,
                            "object_model_id": payload.get("object_id"),
                        }
                    )
                    valid_fragments.append(
                        AnnotatedFragmentMesh(
                            scene_id=scene_id,
                            fragment_id=fragment_id,
                            fragment_key=f"{scene_id}:{expected_name}",
                            annotation_path=annotation_path,
                            mesh_path=mesh_path,
                            coordinate_frame="F",
                            units="m",
                            T_O_from_F=transform,
                            points_F=points_F,
                            points_O=points_O,
                            faces=faces,
                            face_labels_path=face_labels_path,
                            face_labels=face_labels,
                        )
                    )
                except Exception as exc:
                    contract_errors += 1
                    fragment_audit["status"] = "annotation_contract_error"
                    fragment_audit["error"] = str(exc)
                scene_audit["fragments"].append(fragment_audit)
        except Exception as exc:
            scene_audit["status"] = "annotation_contract_error"
            scene_audit["error"] = str(exc)
            contract_errors += 1
        audit_scenes.append(scene_audit)
    audit = {
        "dataset_root": str(root),
        "object_model_id": object_model_id,
        "scene_ids": list(scene_ids),
        "selection": {
            "all_annotated_fragments": bool(all_annotated_fragments),
            "fragment_ids": sorted(requested_ids) if requested_ids is not None else None,
            "max_fragments_per_scene": max_fragments_per_scene,
        },
        "schema": {
            "fragment_list": "root.fragments[]",
            "fragment_id": "fragments[].fragment_id",
            "fragment_mesh_path": "fragments[].mesh, relative to scene directory",
            "fragment_mesh_coordinate_frame": "F",
            "fragment_face_labels": "fragments[].face_labels, relative to scene directory; 0=shell, 1=fracture, 255=unknown",
            "object_transform": "fragments[].T_O_from_F",
            "object_model_id": "root.object_id",
            "object_model_path": "root.object_model, relative to annotation directory",
            "units": "root.geometry.units=scene_unit; dataset contract defines scene units as meters",
        },
        "scenes": audit_scenes,
        "totals": {
            "selected_entries": selected_entries,
            "contract_valid": len(valid_fragments),
            "annotation_contract_errors": contract_errors,
        },
    }
    return audit, valid_fragments


def _template_gallery_colors(
    face_regions: np.ndarray, active_regions: Sequence[bool]
) -> np.ndarray:
    colors = np.broadcast_to(LIGHT_GRAY, (len(face_regions), 3)).copy()
    active = np.asarray(active_regions, dtype=bool)
    for index, enabled in enumerate(active):
        if enabled:
            colors[face_regions == index] = INACTIVE_GRAY
    return colors


def _validate_gallery_math(
    fragment_points_O: Tensor,
    base_pose: Tensor,
    poses: Tensor,
    placed: Tensor,
    metadata: SymmetryMetadata,
) -> dict[str, Any]:
    determinants = torch.linalg.det(poses[:, :3, :3])
    if not bool(torch.allclose(determinants, torch.ones_like(determinants), atol=1e-5, rtol=0.0)):
        raise RuntimeError("a hypothesis rotation determinant differs from 1")
    if not bool(torch.allclose(placed[0], fragment_points_O, atol=1e-6, rtol=1e-6)):
        raise RuntimeError("identity hypothesis does not reproduce fragment_points_O")
    origin = torch.as_tensor(metadata.axis.origin, dtype=poses.dtype, device=poses.device)
    axis = torch.as_tensor(metadata.axis.direction, dtype=poses.dtype, device=poses.device)
    homogeneous_origin = torch.cat((origin, origin.new_ones(1)))
    transformed_origins = torch.einsum("kij,j->ki", poses, homogeneous_origin)[:, :3]
    transformed_axes = torch.einsum("kij,j->ki", poses[:, :3, :3], axis)
    if not bool(torch.allclose(transformed_origins, origin.expand_as(transformed_origins), atol=1e-5, rtol=0.0)):
        raise RuntimeError("a hypothesis does not rotate around the configured origin")
    if not bool(torch.allclose(transformed_axes, axis.expand_as(transformed_axes), atol=1e-5, rtol=0.0)):
        raise RuntimeError("a hypothesis does not preserve the configured axis")
    if len(poses) > 1 and bool(torch.allclose(poses[-1], poses[0], atol=1e-6, rtol=0.0)):
        raise RuntimeError("last hypothesis duplicates the identity")
    if len(fragment_points_O) > 1:
        reference = torch.linalg.vector_norm(fragment_points_O[1:] - fragment_points_O[:-1], dim=-1)
        placed_distances = torch.linalg.vector_norm(placed[:, 1:] - placed[:, :-1], dim=-1)
        if not bool(torch.allclose(placed_distances, reference.expand_as(placed_distances), atol=1e-5, rtol=1e-5)):
            raise RuntimeError("hypothesis placement changes internal fragment distances")
    last_differs = len(poses) == 1 or not bool(
        torch.allclose(poses[-1], poses[0], atol=1e-6, rtol=0.0)
    )
    return {
        "identity_matches_original": True,
        "rotation_determinants": determinants.detach().cpu().tolist(),
        "axis_and_origin_fixed": True,
        "internal_distances_preserved": True,
        "last_hypothesis_differs_from_identity": last_differs,
        "placement_source": "symm_template_reg.models.symmetry.hypothesis_expander.place_fragment_for_hypothesis",
    }


def export_annotated_fragment_visualization(
    output_dir: Path,
    *,
    fragment: AnnotatedFragmentMesh,
    template_points_O: np.ndarray,
    template_faces: np.ndarray,
    metadata: SymmetryMetadata,
    partition: RegionPartitionValidation,
    so2_visualization_samples: int,
    gallery_columns: int,
    gallery_spacing_scale: float,
    template_projection_distance_m: float,
    template_boundary_resolution_m: float,
    template_boundary_max_depth: int,
    min_surface_area_m2: float,
    min_surface_area_fraction: float,
    area_sample_count: int,
    min_area_sample_count: int,
) -> dict[str, Any]:
    """Color the template footprint of a fragment for every symmetry hypothesis."""

    output_dir.mkdir(parents=True, exist_ok=False)
    copied_fragment_path = output_dir / fragment.mesh_path.name
    if copied_fragment_path.exists():
        raise FileExistsError(f"refusing to overwrite {copied_fragment_path}")
    shutil.copyfile(fragment.mesh_path, copied_fragment_path)
    fragment_points = torch.from_numpy(fragment.points_O)
    fragment_faces = torch.from_numpy(fragment.faces)
    identity = torch.eye(4, dtype=fragment_points.dtype)
    targets = build_fragment_symmetry_targets(
        fragment_points,
        metadata,
        fragment_faces=fragment_faces,
        base_pose=identity,
        min_surface_area_m2=min_surface_area_m2,
        min_surface_area_fraction=min_surface_area_fraction,
        area_sample_count=area_sample_count,
        min_area_sample_count=min_area_sample_count,
    )
    if targets.face_region_indices is None:
        raise RuntimeError("mesh-aware target builder returned no face regions")
    active = targets.active_regions.detach().cpu().numpy().astype(bool)
    template_face_regions = partition.face_region_indices.detach().cpu().numpy()
    template_colors = _template_gallery_colors(template_face_regions, active)
    palette = _region_colors(len(metadata.regions))
    shell_faces = fragment.faces[fragment.face_labels == 0]
    shell_surface = _TriangleSurfaceIndex(
        fragment.points_O[shell_faces], template_projection_distance_m
    )
    identity_vertex_mask, identity_face_mask = _project_fragment_shell_to_template(
        template_points=template_points_O,
        template_faces=template_faces,
        shell_surface=shell_surface,
    )
    identity_refined = _refine_template_at_fragment_boundary(
        template_points=template_points_O,
        template_faces=template_faces,
        shell_surface=shell_surface,
        template_to_fragment_surface=None,
        boundary_resolution_m=template_boundary_resolution_m,
        max_depth=template_boundary_max_depth,
    )
    fragment_shell_area_m2 = float(
        _triangle_areas(fragment.points_O, shell_faces).sum()
    )
    identity_refined_areas = _triangle_areas(
        identity_refined.vertices, identity_refined.faces
    )
    identity_projected_area_m2 = float(
        identity_refined_areas[identity_refined.projected_faces].sum()
    )
    projected_region_colors = np.broadcast_to(
        LIGHT_GRAY, (len(identity_refined.faces), 3)
    ).copy()
    projected_region_colors[identity_refined.projected_faces] = palette[
        template_face_regions[
            identity_refined.source_face_indices[identity_refined.projected_faces]
        ]
    ]

    regions_builder = GeometryBuilder()
    regions_builder.add_face_colored_mesh(
        identity_refined.vertices, identity_refined.faces, projected_region_colors
    )
    _add_axis_boundaries_and_origin(
        regions_builder, template_points_O, metadata, include_reference=True
    )
    regions_builder.write(output_dir / "fragment_regions_on_template.ply")
    _save_npy(output_dir / "template_projected_vertex_mask.npy", identity_vertex_mask)
    _save_npy(output_dir / "template_projected_face_mask.npy", identity_face_mask)

    gallery_pose_set = (
        visualization_equivalent_pose_set(
            identity,
            metadata,
            effective_group=targets.effective_group,
            so2_visualization_samples=so2_visualization_samples,
        )
        if isinstance(targets.effective_group, SO2Group)
        else targets.equivalent_pose_set
    )
    pose_count = gallery_pose_set.num_hypotheses
    extent = max(float(np.ptp(template_points_O, axis=0).max()), 1e-4)
    spacing_m = extent * gallery_spacing_scale
    offsets = gallery_offsets(pose_count, columns=gallery_columns, spacing_m=spacing_m)
    colors = _hypothesis_colors(pose_count)
    placed = place_fragment_for_hypothesis(
        fragment_points, identity, gallery_pose_set.poses
    )
    math_checks = _validate_gallery_math(
        fragment_points, identity, gallery_pose_set.poses, placed, metadata
    )
    angles = group_angles(
        targets.effective_group,
        so2_num_samples=pose_count if isinstance(targets.effective_group, SO2Group) else None,
        dtype=fragment_points.dtype,
    )

    gallery = GeometryBuilder()
    hypothesis_entries: list[dict[str, Any]] = []
    for index in range(pose_count):
        offset = offsets[index]
        hypothesis_transform = gallery_pose_set.poses[index].detach().cpu().numpy()
        projected_vertex_mask, projected_face_mask = _project_fragment_shell_to_template(
            template_points=template_points_O,
            template_faces=template_faces,
            shell_surface=shell_surface,
            template_to_fragment_surface=hypothesis_transform,
        )
        refined = (
            identity_refined
            if index == 0
            else _refine_template_at_fragment_boundary(
                template_points=template_points_O,
                template_faces=template_faces,
                shell_surface=shell_surface,
                template_to_fragment_surface=hypothesis_transform,
                boundary_resolution_m=template_boundary_resolution_m,
                max_depth=template_boundary_max_depth,
            )
        )
        hypothesis_template_colors = template_colors[
            refined.source_face_indices
        ].copy()
        hypothesis_template_colors[refined.projected_faces] = colors[index]
        refined_areas = _triangle_areas(refined.vertices, refined.faces)
        gallery.add_face_colored_mesh(
            refined.vertices + offset, refined.faces, hypothesis_template_colors
        )
        _add_axis_boundaries_and_origin(
            gallery,
            template_points_O,
            metadata,
            offset=offset,
            include_boundaries=False,
            include_reference=True,
        )
        group_element = (
            f"SO2:sample={index}"
            if isinstance(targets.effective_group, SO2Group)
            else f"{targets.effective_group.name}:k={index}"
        )
        hypothesis_entries.append(
            {
                "index": index,
                "group_element": group_element,
                "angle_deg": float(torch.rad2deg(angles[index])),
                "color_rgb": colors[index].tolist(),
                "projected_template_vertices": int(projected_vertex_mask.sum()),
                "projected_template_faces": int(projected_face_mask.sum()),
                "output_template_faces": int(len(refined.faces)),
                "output_projected_face_pieces": int(refined.projected_faces.sum()),
                "boundary_split_source_faces": refined.split_source_faces,
                "output_template_surface_area_m2": float(refined_areas.sum()),
                "output_projected_surface_area_m2": float(
                    refined_areas[refined.projected_faces].sum()
                ),
                "gallery_offset_m": offset.tolist(),
                "transform": gallery_pose_set.poses[index].detach().cpu().tolist(),
            }
        )
    gallery.write(output_dir / "hypothesis_gallery.ply")
    hypothesis_index = {
        "scene_id": fragment.scene_id,
        "fragment_id": fragment.fragment_id,
        "effective_group": group_to_dict(targets.effective_group),
        "training_semantics": targets.training_target_type,
        "finite_visualization_of_continuous_group": isinstance(
            targets.effective_group, SO2Group
        ),
        "gallery_layout": {
            "columns": gallery_columns,
            "spacing_m": spacing_m,
            "template_copy_count": pose_count,
            "fragment_copy_count": 0,
            "fragments_per_template_copy": 0,
            "representation": "fragment shell footprint colored on template faces",
        },
        "hypotheses": hypothesis_entries,
    }
    _write_json(output_dir / "hypothesis_index.json", hypothesis_index)

    region_summaries: list[dict[str, Any]] = []
    for index, region in enumerate(metadata.regions):
        decision = targets.active_region_decisions[index]
        region_summaries.append(
            {
                "region_id": region.region_id,
                "group": region.rotation_group.name,
                "vertex_count": int(targets.region_point_counts[index]),
                "face_count": int(targets.region_face_counts[index]),
                "surface_area_m2": float(targets.region_surface_areas_m2[index]),
                "surface_area_fraction": float(targets.region_surface_area_fractions[index]),
                "area_weighted_sample_count": int(targets.region_area_sample_counts[index]),
                "active": decision["active"],
                "decision_reasons": decision["reasons"],
            }
        )
    active_region_ids = [
        region.region_id
        for region, enabled in zip(metadata.regions, active.tolist())
        if enabled
    ]
    summary = {
        "scene_id": fragment.scene_id,
        "fragment_id": fragment.fragment_id,
        "fragment_key": fragment.fragment_key,
        "fragment_mesh_path": str(fragment.mesh_path),
        "copied_fragment_mesh_path": str(copied_fragment_path),
        "fragment_face_labels_path": str(fragment.face_labels_path),
        "fragment_annotation_path": str(fragment.annotation_path),
        "fragment_coordinate_frame": fragment.coordinate_frame,
        "transform_applied": "T_O_from_F",
        "T_O_from_F": fragment.T_O_from_F.tolist(),
        "num_vertices": int(len(fragment.points_O)),
        "num_faces": int(len(fragment.faces)),
        "num_shell_faces": int(np.count_nonzero(fragment.face_labels == 0)),
        "num_fracture_faces": int(np.count_nonzero(fragment.face_labels == 1)),
        "fragment_shell_surface_area_m2": fragment_shell_area_m2,
        "template_projection_distance_m": float(template_projection_distance_m),
        "template_boundary_resolution_m": float(template_boundary_resolution_m),
        "template_boundary_max_depth": int(template_boundary_max_depth),
        "identity_projected_template_vertices": int(identity_vertex_mask.sum()),
        "identity_projected_template_faces": int(identity_face_mask.sum()),
        "identity_output_template_faces": int(len(identity_refined.faces)),
        "identity_output_projected_face_pieces": int(
            identity_refined.projected_faces.sum()
        ),
        "identity_boundary_split_source_faces": identity_refined.split_source_faces,
        "identity_output_template_surface_area_m2": float(
            identity_refined_areas.sum()
        ),
        "identity_projected_template_surface_area_m2": identity_projected_area_m2,
        "identity_projected_area_relative_error_vs_fragment_shell": float(
            abs(identity_projected_area_m2 - fragment_shell_area_m2)
            / max(fragment_shell_area_m2, 1e-12)
        ),
        "projection_uses_shell_faces_only": True,
        "regions": region_summaries,
        "active_regions": active_region_ids,
        "effective_group": group_to_dict(targets.effective_group),
        "effective_group_name": targets.effective_group.name,
        "training_target_type": targets.training_target_type,
        "training_pose_set_exhaustive": targets.equivalent_pose_set.exhaustive,
        "training_group_elements": int(targets.group_elements.shape[0]),
        "num_hypotheses": pose_count,
        "gallery_is_finite_visualization_only": isinstance(
            targets.effective_group, SO2Group
        ),
        "so2_visualization_samples": (
            pose_count if isinstance(targets.effective_group, SO2Group) else None
        ),
        "diagnostics": targets.diagnostics,
        "hypothesis_math_checks": math_checks,
        "gallery_template_copy_count": pose_count,
        "gallery_fragment_copy_count": 0,
        "fragments_per_template_copy": 0,
        "gallery_representation": "fragment shell footprint colored on template faces",
        "production_function_sources": {
            "target_builder": "symm_template_reg.models.symmetry.targets.build_fragment_symmetry_targets",
            "region_assignment": "symm_template_reg.models.symmetry.region_assignment.assign_symmetry_regions",
            "group_intersection": "symm_template_reg.models.symmetry.groups.intersect_rotation_groups",
            "hypothesis_expansion": "symm_template_reg.models.symmetry.hypothesis_expander.equivalent_gt_pose_set",
            "fragment_placement": "symm_template_reg.models.symmetry.hypothesis_expander.place_fragment_for_hypothesis",
            "template_projection": "orthogonal projection into shell-triangle interiors plus adaptive template-boundary splitting",
        },
        "warnings": [],
        "status": "ok",
    }
    _write_json(output_dir / "fragment_summary.json", summary)
    return summary


def _fragment_math_semantics(
    so2_visualization_samples: int,
    *,
    template_projection_distance_m: float,
    template_boundary_resolution_m: float,
    template_boundary_max_depth: int,
    min_surface_area_m2: float,
    min_surface_area_fraction: float,
    area_sample_count: int,
    min_area_sample_count: int,
) -> dict[str, Any]:
    semantics = _math_semantics(so2_visualization_samples)
    semantics.update(
        {
            "analysis_unit": "scene_id + fragment_id",
            "source_contract": "fragment mesh F -> explicit annotation T_O_from_F -> O",
            "fragment_activation": {
                "operation": "OR over enabled mesh-area criteria; vertex counts are diagnostic only",
                "min_surface_area_m2": min_surface_area_m2,
                "min_surface_area_fraction": min_surface_area_fraction,
                "area_sample_count": area_sample_count,
                "min_area_sample_count": min_area_sample_count,
                "production_source": "symm_template_reg.models.symmetry.targets.build_fragment_symmetry_targets",
            },
            "fragment_hypothesis_placement": "inverse(hypothesis_pose) @ base_pose @ fragment_points_O",
            "visualization_projection": {
                "source_faces": "fragment face_labels == 0 (shell only)",
                "target_geometry": "template vertices and triangle centers",
                "distance_m": template_projection_distance_m,
                "boundary_resolution_m": template_boundary_resolution_m,
                "boundary_max_depth": template_boundary_max_depth,
                "output": "locally split and colored template faces; fragment mesh is copied separately",
            },
            "forbidden_inputs_not_read": [
                "RGB",
                "depth",
                "instance masks",
                "surface masks",
                "visible_points/frame_*.npz",
            ],
        }
    )
    return semantics


def run_annotated_fragment_symmetry_debug(
    *,
    dataset_root: str | Path,
    object_model_id: str,
    scene_ids: Sequence[str] = ("scene_000000", "scene_000001", "scene_000002"),
    fragment_ids: Sequence[int] | None = None,
    max_fragments_per_scene: int | None = None,
    all_annotated_fragments: bool = True,
    template_mesh: str = "auto",
    symmetry_sidecar: str = "auto",
    mode: str = "all",
    so2_visualization_samples: int = DEFAULT_SO2_VISUALIZATION_SAMPLES,
    gallery_columns: int = 4,
    gallery_spacing_scale: float = 2.5,
    template_projection_distance_m: float = 5e-4,
    template_boundary_resolution_m: float = 1e-4,
    template_boundary_max_depth: int = 2,
    min_surface_area_m2: float = 0.0,
    min_surface_area_fraction: float = 0.01,
    area_sample_count: int = 2048,
    min_area_sample_count: int = 16,
    output_root: str | Path = DEFAULT_DEBUG_OUTPUT_ROOT,
    timestamp: str | None = None,
) -> Path:
    """Run the annotation-level debug without enumerating Dataset frames."""

    if mode not in {"template", "fragments", "all"}:
        raise ValueError("mode must be template, fragments, or all")
    if (
        so2_visualization_samples < 1
        or gallery_columns < 1
        or gallery_spacing_scale <= 0
        or template_projection_distance_m <= 0
        or template_boundary_resolution_m <= 0
        or template_boundary_max_depth < 0
    ):
        raise ValueError("visualization sample/layout parameters must be positive")
    root = Path(dataset_root).expanduser().resolve()
    run_dir = create_unique_run_directory(output_root, timestamp=timestamp)
    template_path, sidecar_path, template_points, template_faces, metadata = (
        _resolve_template_and_sidecar(
            root, object_model_id, template_mesh, symmetry_sidecar
        )
    )
    partition = validate_region_partition(
        torch.from_numpy(template_points),
        torch.from_numpy(template_faces),
        metadata,
        coverage_tolerance_m=1e-6,
    )
    _write_json(
        run_dir / "math_semantics.json",
        _fragment_math_semantics(
            so2_visualization_samples,
            template_projection_distance_m=template_projection_distance_m,
            template_boundary_resolution_m=template_boundary_resolution_m,
            template_boundary_max_depth=template_boundary_max_depth,
            min_surface_area_m2=min_surface_area_m2,
            min_surface_area_fraction=min_surface_area_fraction,
            area_sample_count=area_sample_count,
            min_area_sample_count=min_area_sample_count,
        ),
    )

    if mode == "template":
        annotation_audit = {
            "dataset_root": str(root),
            "object_model_id": object_model_id,
            "scene_ids": [],
            "selection": {"mode": "template", "annotations_read": False},
            "schema": {},
            "scenes": [],
            "totals": {
                "selected_entries": 0,
                "contract_valid": 0,
                "annotation_contract_errors": 0,
            },
        }
        fragments: list[AnnotatedFragmentMesh] = []
    else:
        annotation_audit, fragments = audit_fragment_annotations(
            root,
            scene_ids=scene_ids,
            object_model_id=object_model_id,
            template_path=template_path,
            fragment_ids=fragment_ids,
            max_fragments_per_scene=max_fragments_per_scene,
            all_annotated_fragments=all_annotated_fragments,
        )
    _write_json(run_dir / "annotation_audit.json", annotation_audit)
    _write_text(run_dir / "annotation_audit.md", _annotation_audit_markdown(annotation_audit))

    template_summary = None
    if mode in {"template", "all"}:
        template_summary = export_template_visualization(
            run_dir / "template",
            object_model_id=object_model_id,
            template_path=template_path,
            sidecar_path=sidecar_path,
            points_O=template_points,
            faces=template_faces,
            metadata=metadata,
            partition=partition,
        )

    summaries: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    if mode in {"fragments", "all"}:
        for fragment in fragments:
            relative = (
                Path("scenes")
                / fragment.scene_id
                / "fragments"
                / f"fragment_{fragment.fragment_id:04d}"
            )
            try:
                summary = export_annotated_fragment_visualization(
                    run_dir / relative,
                    fragment=fragment,
                    template_points_O=template_points,
                    template_faces=template_faces,
                    metadata=metadata,
                    partition=partition,
                    so2_visualization_samples=so2_visualization_samples,
                    gallery_columns=gallery_columns,
                    gallery_spacing_scale=gallery_spacing_scale,
                    template_projection_distance_m=template_projection_distance_m,
                    template_boundary_resolution_m=template_boundary_resolution_m,
                    template_boundary_max_depth=template_boundary_max_depth,
                    min_surface_area_m2=min_surface_area_m2,
                    min_surface_area_fraction=min_surface_area_fraction,
                    area_sample_count=area_sample_count,
                    min_area_sample_count=min_area_sample_count,
                )
                summaries.append(summary)
                index_rows.append(
                    {
                        "scene_id": fragment.scene_id,
                        "fragment_id": fragment.fragment_id,
                        "fragment_key": fragment.fragment_key,
                        "fragment_mesh_path": str(fragment.mesh_path),
                        "fragment_annotation_path": str(fragment.annotation_path),
                        "fragment_coordinate_frame": fragment.coordinate_frame,
                        "num_vertices": len(fragment.points_O),
                        "num_faces": len(fragment.faces),
                        "active_regions": ";".join(summary["active_regions"]),
                        "effective_group": summary["effective_group_name"],
                        "num_hypotheses": summary["num_hypotheses"],
                        "copied_fragment_ply": str(relative / fragment.mesh_path.name),
                        "gallery_ply": str(relative / "hypothesis_gallery.ply"),
                        "summary_json": str(relative / "fragment_summary.json"),
                        "status": "ok",
                        "warnings": "",
                    }
                )
            except Exception as exc:
                index_rows.append(
                    {
                        "scene_id": fragment.scene_id,
                        "fragment_id": fragment.fragment_id,
                        "fragment_key": fragment.fragment_key,
                        "fragment_mesh_path": str(fragment.mesh_path),
                        "fragment_annotation_path": str(fragment.annotation_path),
                        "fragment_coordinate_frame": fragment.coordinate_frame,
                        "num_vertices": len(fragment.points_O),
                        "num_faces": len(fragment.faces),
                        "active_regions": "",
                        "effective_group": "",
                        "num_hypotheses": 0,
                        "copied_fragment_ply": "",
                        "gallery_ply": "",
                        "summary_json": "",
                        "status": "processing_error",
                        "warnings": str(exc),
                    }
                )
        for scene in annotation_audit.get("scenes", []):
            for entry in scene.get("fragments", []):
                if entry.get("status") != "annotation_contract_error":
                    continue
                index_rows.append(
                    {
                        "scene_id": scene["scene_id"],
                        "fragment_id": entry.get("fragment_id", ""),
                        "fragment_key": entry.get("fragment_key", ""),
                        "fragment_mesh_path": entry.get("fragment_mesh_path", ""),
                        "fragment_annotation_path": entry.get("fragment_annotation_path", ""),
                        "fragment_coordinate_frame": entry.get("fragment_coordinate_frame", ""),
                        "num_vertices": entry.get("num_vertices_actual", ""),
                        "num_faces": entry.get("num_faces_actual", ""),
                        "active_regions": "",
                        "effective_group": "",
                        "num_hypotheses": 0,
                        "copied_fragment_ply": "",
                        "gallery_ply": "",
                        "summary_json": "",
                        "status": "annotation_contract_error",
                        "warnings": entry.get("error", ""),
                    }
                )

    fieldnames = [
        "scene_id",
        "fragment_id",
        "fragment_key",
        "fragment_mesh_path",
        "fragment_annotation_path",
        "fragment_coordinate_frame",
        "num_vertices",
        "num_faces",
        "active_regions",
        "effective_group",
        "num_hypotheses",
        "copied_fragment_ply",
        "gallery_ply",
        "summary_json",
        "status",
        "warnings",
    ]
    with (run_dir / "fragments_index.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    status_counts: dict[str, int] = {}
    for row in index_rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    run_summary = {
        "run_directory": str(run_dir),
        "dataset_root": str(root),
        "object_model_id": object_model_id,
        "template_path": str(template_path),
        "sidecar_path": str(sidecar_path),
        "mode": mode,
        "scene_ids": list(scene_ids) if mode != "template" else [],
        "analysis_unit": "scene_id + fragment_id",
        "template_generated": template_summary is not None,
        "num_fragments_selected": len(index_rows),
        "num_fragments_processed": len(summaries),
        "status_counts": status_counts,
        "fragments": [
            {
                "scene_id": summary["scene_id"],
                "fragment_id": summary["fragment_id"],
                "active_regions": summary["active_regions"],
                "effective_group": summary["effective_group_name"],
                "num_hypotheses": summary["num_hypotheses"],
                "copied_fragment_mesh_path": summary["copied_fragment_mesh_path"],
                "continuous_training_semantics": summary["training_target_type"]
                == "continuous_analytic",
            }
            for summary in summaries
        ],
        "source_data_modified": False,
        "frame_files_enumerated": False,
        "forbidden_modalities_read": [],
        "recommended_open_order": [
            "template/template_symmetry_regions_with_boundaries.ply",
            "scenes/scene_000000/fragments/<fragment>/fragment_regions_on_template.ply",
            "scenes/scene_000000/fragments/<fragment>/hypothesis_gallery.ply",
            "scenes/scene_000000/fragments/<fragment>/<fragment>.ply",
        ],
    }
    _write_json(run_dir / "run_summary.json", run_summary)
    return run_dir


__all__ = [
    "AnnotatedFragmentMesh",
    "apply_gallery_offset",
    "audit_fragment_annotations",
    "camera_points_in_hypothesis_object",
    "create_unique_run_directory",
    "export_sample_visualization",
    "export_template_visualization",
    "gallery_offsets",
    "run_annotated_fragment_symmetry_debug",
    "run_symmetry_debug",
    "select_dataset_indices",
]
