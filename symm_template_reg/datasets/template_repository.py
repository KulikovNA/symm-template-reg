"""Cached template mesh loading with a dependency-free PLY fallback."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np
import torch

from .transforms import farthest_point_indices


_PLY_SCALARS: dict[str, tuple[str, np.dtype[Any]]] = {
    "char": ("b", np.dtype("i1")),
    "int8": ("b", np.dtype("i1")),
    "uchar": ("B", np.dtype("u1")),
    "uint8": ("B", np.dtype("u1")),
    "short": ("h", np.dtype("i2")),
    "int16": ("h", np.dtype("i2")),
    "ushort": ("H", np.dtype("u2")),
    "uint16": ("H", np.dtype("u2")),
    "int": ("i", np.dtype("i4")),
    "int32": ("i", np.dtype("i4")),
    "uint": ("I", np.dtype("u4")),
    "uint32": ("I", np.dtype("u4")),
    "float": ("f", np.dtype("f4")),
    "float32": ("f", np.dtype("f4")),
    "double": ("d", np.dtype("f8")),
    "float64": ("d", np.dtype("f8")),
}


def _read_ply_header(stream: BinaryIO) -> tuple[str, list[dict[str, Any]], list[str]]:
    if stream.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    file_format: str | None = None
    elements: list[dict[str, Any]] = []
    comments: list[str] = []
    current: dict[str, Any] | None = None
    while True:
        raw = stream.readline()
        if not raw:
            raise ValueError("truncated PLY header")
        line = raw.decode("ascii").strip()
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "format":
            file_format = fields[1]
        elif fields[0] in {"comment", "obj_info"}:
            comments.append(" ".join(fields[1:]))
        elif fields[0] == "element":
            current = {"name": fields[1], "count": int(fields[2]), "properties": []}
            elements.append(current)
        elif fields[0] == "property":
            if current is None:
                raise ValueError("PLY property declared before an element")
            if fields[1] == "list":
                current["properties"].append(("list", fields[2], fields[3], fields[4]))
            else:
                current["properties"].append(("scalar", fields[1], fields[2]))
        elif fields[0] == "end_header":
            break
    if file_format not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported PLY format {file_format!r}")
    return file_format, elements, comments


def inspect_ply_header(path: str | Path) -> dict[str, Any]:
    """Return PLY structure without loading its potentially large payload."""

    with Path(path).open("rb") as stream:
        file_format, elements, comments = _read_ply_header(stream)
    return {"format": file_format, "elements": elements, "comments": comments}


def _read_ascii_element(stream: BinaryIO, element: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _ in range(element["count"]):
        fields = stream.readline().decode("ascii").split()
        if not fields:
            raise ValueError(f"truncated ASCII PLY element {element['name']!r}")
        cursor = 0
        row: dict[str, Any] = {}
        for prop in element["properties"]:
            if prop[0] == "scalar":
                dtype = _PLY_SCALARS[prop[1]][1]
                row[prop[2]] = np.asarray(fields[cursor], dtype=dtype).item()
                cursor += 1
            else:
                count = int(fields[cursor])
                cursor += 1
                dtype = _PLY_SCALARS[prop[2]][1]
                row[prop[3]] = np.asarray(fields[cursor : cursor + count], dtype=dtype)
                cursor += count
        rows.append(row)
    return rows


def _unpack_scalar(stream: BinaryIO, scalar_type: str, endian: str) -> Any:
    fmt = endian + _PLY_SCALARS[scalar_type][0]
    data = stream.read(struct.calcsize(fmt))
    if len(data) != struct.calcsize(fmt):
        raise ValueError("truncated binary PLY payload")
    return struct.unpack(fmt, data)[0]


def _read_binary_element(
    stream: BinaryIO, element: dict[str, Any], endian: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _ in range(element["count"]):
        row: dict[str, Any] = {}
        for prop in element["properties"]:
            if prop[0] == "scalar":
                row[prop[2]] = _unpack_scalar(stream, prop[1], endian)
            else:
                count = int(_unpack_scalar(stream, prop[1], endian))
                row[prop[3]] = np.asarray(
                    [_unpack_scalar(stream, prop[2], endian) for _ in range(count)],
                    dtype=_PLY_SCALARS[prop[2]][1],
                )
        rows.append(row)
    return rows


def load_ply(path: str | Path) -> dict[str, Any]:
    """Load vertices, optional normals and triangular faces from a PLY file."""

    path = Path(path)
    with path.open("rb") as stream:
        file_format, elements, comments = _read_ply_header(stream)
        data: dict[str, list[dict[str, Any]]] = {}
        for element in elements:
            if file_format == "ascii":
                rows = _read_ascii_element(stream, element)
            else:
                endian = "<" if file_format == "binary_little_endian" else ">"
                rows = _read_binary_element(stream, element, endian)
            data[element["name"]] = rows
    vertices = data.get("vertex", [])
    if not vertices or not {"x", "y", "z"}.issubset(vertices[0]):
        raise ValueError(f"PLY {path} has no x/y/z vertex element")
    points = np.asarray(
        [[row["x"], row["y"], row["z"]] for row in vertices], dtype=np.float32
    )
    normals: np.ndarray | None = None
    if {"nx", "ny", "nz"}.issubset(vertices[0]):
        normals = np.asarray(
            [[row["nx"], row["ny"], row["nz"]] for row in vertices],
            dtype=np.float32,
        )
    faces_list: list[list[int]] = []
    face_polygons: list[np.ndarray] = []
    for row in data.get("face", []):
        indices = row.get("vertex_indices", row.get("vertex_index"))
        if indices is None:
            continue
        values = [int(value) for value in indices]
        face_polygons.append(np.asarray(values, dtype=np.int64))
        if len(values) == 3:
            faces_list.append(values)
        elif len(values) > 3:
            # Deterministic fan triangulation is sufficient for template meshes.
            faces_list.extend([[values[0], values[i], values[i + 1]] for i in range(1, len(values) - 1)])
    faces = np.asarray(faces_list, dtype=np.int64) if faces_list else None
    return {
        "points": points,
        "normals": normals,
        "faces": faces,
        "face_polygons": face_polygons,
        "num_face_records": len(face_polygons),
        "polygon_size_distribution": {
            str(size): sum(len(polygon) == size for polygon in face_polygons)
            for size in sorted({len(polygon) for polygon in face_polygons})
        },
        "format": file_format,
        "comments": comments,
    }


def compute_vertex_normals(points: np.ndarray, faces: np.ndarray | None) -> np.ndarray | None:
    """Compute area-weighted unit vertex normals for a triangular mesh."""

    if faces is None or len(faces) == 0:
        return None
    normals = np.zeros_like(points, dtype=np.float64)
    triangles = points[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-12
    normals[valid] /= lengths[valid]
    normals[~valid] = 0.0
    return normals.astype(np.float32)


def _load_symmetry_sidecar(path: Path, expected_id: str) -> Any | None:
    if not path.exists():
        return None
    try:
        from symm_template_reg.models.symmetry.metadata import load_symmetry_metadata
    except (ImportError, AttributeError):
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    try:
        return load_symmetry_metadata(path, expected_object_model_id=expected_id)
    except TypeError:
        return load_symmetry_metadata(path)


class TemplateRepository:
    """Resolve templates by model id and reuse tensor objects across samples."""

    def __init__(
        self,
        models_dir: str | Path,
        *,
        fine_points: int | None = 4096,
        coarse_points: int | None = 1024,
        fine_sample_count: int | None = None,
        coarse_sample_count: int | None = None,
        compute_missing_normals: bool = True,
        recompute_normals: bool = False,
    ) -> None:
        resolved = Path(models_dir).expanduser().resolve()
        if (resolved / "models").is_dir() and not any(resolved.glob("*.ply")):
            resolved = resolved / "models"
        self.models_dir = resolved
        if not self.models_dir.is_dir():
            raise FileNotFoundError(f"template models directory not found: {self.models_dir}")
        self.fine_points = fine_sample_count if fine_sample_count is not None else fine_points
        self.coarse_points = coarse_sample_count if coarse_sample_count is not None else coarse_points
        for name, count in (("fine_points", self.fine_points), ("coarse_points", self.coarse_points)):
            if count is not None and int(count) <= 0:
                raise ValueError(f"{name} must be positive or None")
        self.compute_missing_normals = compute_missing_normals
        self.recompute_normals = bool(recompute_normals)
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_counts: dict[str, int] = {}
        self._feature_cache: dict[tuple[str, str], Any] = {}

    def clear(self) -> None:
        self._cache.clear()
        self._feature_cache.clear()

    @property
    def cache_size(self) -> int:
        return len({id(value) for value in self._cache.values()})

    def __len__(self) -> int:
        return self.cache_size

    def cache_info(self) -> dict[str, Any]:
        return {
            "templates": self.cache_size,
            "keys": sorted(self._cache),
            "load_counts": dict(self._load_counts),
            "feature_entries": len(self._feature_cache),
        }

    def load_count(self, object_model_id: str) -> int:
        return self._load_counts.get(object_model_id, 0)

    def cache_feature(self, object_model_id: str, key: str, value: Any) -> None:
        canonical = self._canonical_id(object_model_id)
        self._feature_cache[(canonical, key)] = value

    def get_cached_feature(self, object_model_id: str, key: str) -> Any | None:
        canonical = self._canonical_id(object_model_id)
        return self._feature_cache.get((canonical, key))

    def _canonical_id(self, object_model_id: str) -> str:
        cached = self._cache.get(object_model_id)
        if cached is not None:
            return str(cached["object_model_id"])
        try:
            return self._resolve_mesh(object_model_id).stem
        except (FileNotFoundError, ValueError):
            return object_model_id

    def _resolve_mesh(self, object_model_id: str) -> Path:
        exact = self.models_dir / f"{object_model_id}.ply"
        if exact.is_file():
            return exact
        candidates = sorted(self.models_dir.glob(f"{object_model_id}__*.ply"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(
                f"no template PLY for {object_model_id!r} in {self.models_dir}"
            )
        raise ValueError(
            f"ambiguous template id {object_model_id!r}: {[path.name for path in candidates]}"
        )

    def get(self, object_model_id: str) -> dict[str, Any]:
        if object_model_id in self._cache:
            return self._cache[object_model_id]
        mesh_path = self._resolve_mesh(object_model_id)
        canonical_id = mesh_path.stem
        if canonical_id in self._cache:
            self._cache[object_model_id] = self._cache[canonical_id]
            return self._cache[canonical_id]
        mesh = load_ply(mesh_path)
        points = mesh["points"]
        normals = mesh["normals"]
        if self.recompute_normals or (
            normals is None and self.compute_missing_normals
        ):
            normals = compute_vertex_normals(points, mesh["faces"])
        fine_indices = self._sample_indices(points, self.fine_points)
        coarse_indices = self._sample_indices(points, self.coarse_points)
        meta_path = mesh_path.with_suffix(".meta.json")
        metadata: dict[str, Any] = {}
        if meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
        source_object_id = str(metadata.get("object_id", canonical_id.split("__", 1)[0]))
        sidecar_candidates = [
            mesh_path.with_suffix(".symmetry.json"),
            self.models_dir / f"{source_object_id}.symmetry.json",
        ]
        sidecar_path = next((path for path in sidecar_candidates if path.is_file()), None)
        symmetry = (
            _load_symmetry_sidecar(sidecar_path, source_object_id)
            if sidecar_path is not None
            else None
        )
        result: dict[str, Any] = {
            "points_O": torch.from_numpy(points.copy()),
            "normals_O": torch.from_numpy(normals.copy()) if normals is not None else None,
            "faces": torch.from_numpy(mesh["faces"].copy()) if mesh["faces"] is not None else None,
            "object_model_id": canonical_id,
            "source_object_id": source_object_id,
            "mesh_path": str(mesh_path),
            "metadata": metadata,
            "symmetry_metadata": symmetry,
            "symmetry_sidecar_path": str(sidecar_path) if sidecar_path else None,
            "fine_indices": torch.from_numpy(fine_indices),
            "coarse_indices": torch.from_numpy(coarse_indices),
            "fine_points_O": torch.from_numpy(points[fine_indices].copy()),
            "coarse_points_O": torch.from_numpy(points[coarse_indices].copy()),
            "fine_normals_O": (
                torch.from_numpy(normals[fine_indices].copy()) if normals is not None else None
            ),
            "coarse_normals_O": (
                torch.from_numpy(normals[coarse_indices].copy()) if normals is not None else None
            ),
        }
        self._cache[canonical_id] = result
        self._cache[object_model_id] = result
        self._load_counts[canonical_id] = self._load_counts.get(canonical_id, 0) + 1
        if object_model_id != canonical_id:
            self._load_counts[object_model_id] = self._load_counts[canonical_id]
        return result

    load = get
    get_template = get

    def __getitem__(self, object_model_id: str) -> dict[str, Any]:
        return self.get(object_model_id)

    @staticmethod
    def _sample_indices(points: np.ndarray, count: int | None) -> np.ndarray:
        if count is None or count >= len(points):
            return np.arange(len(points), dtype=np.int64)
        return np.sort(farthest_point_indices(points, count))


__all__ = [
    "TemplateRepository",
    "compute_vertex_normals",
    "inspect_ply_header",
    "load_ply",
]
