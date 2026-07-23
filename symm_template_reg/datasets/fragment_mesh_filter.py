"""Physical fragment-mesh metadata, caching, and training-data filtering."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .template_repository import load_ply


REQUIRED_TEST_SPLIT_FLAGS = {
    "debug_training_on_test_split": True,
    "results_are_not_final_evaluation": True,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


@dataclass(frozen=True)
class FragmentMeshMetadata:
    scene_id: str
    fragment_id: int
    fragment_key: str
    mesh_path: Path
    num_vertices: int
    num_faces: int
    surface_area_m2: float
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_diagonal_m: float
    sha256: str
    polygon_size_distribution: dict[str, int]
    file_size: int
    mtime_ns: int
    annotation_path: Path
    annotation_sha256: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["mesh_path"] = str(self.mesh_path)
        result["annotation_path"] = str(self.annotation_path)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FragmentMeshMetadata":
        payload = dict(value)
        payload["mesh_path"] = Path(str(payload["mesh_path"])).resolve()
        payload["annotation_path"] = Path(str(payload["annotation_path"])).resolve()
        payload["bbox_min"] = tuple(float(item) for item in payload["bbox_min"])
        payload["bbox_max"] = tuple(float(item) for item in payload["bbox_max"])
        payload["polygon_size_distribution"] = {
            str(key): int(count)
            for key, count in dict(payload.get("polygon_size_distribution", {})).items()
        }
        return cls(**payload)


@dataclass(frozen=True)
class FragmentFilterDecision:
    accepted: bool
    reasons: list[str]
    metadata: FragmentMeshMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "metadata": self.metadata.to_dict(),
        }


DEFAULT_FRAGMENT_MESH_FILTER = {
    "enabled": False,
    "min_num_faces": None,
    "max_num_faces": None,
    "min_num_vertices": None,
    "min_surface_area_m2": None,
    "min_bbox_diagonal_m": None,
    "exclude_entire_fragment": True,
    "missing_mesh_policy": "error",
    "manifest_mismatch_policy": "error",
    "cache_metadata": False,
    "train_policy": "exclude",
    "debug_eval_policy": "exclude",
    "validation_policy": "report_only",
}


class FragmentMeshFilter:
    """Evaluate physical fragment meshes independently of camera observations."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        merged = dict(DEFAULT_FRAGMENT_MESH_FILTER)
        if config is not None:
            unknown = set(config).difference(merged)
            if unknown:
                raise ValueError(f"unknown fragment_mesh_filter fields: {sorted(unknown)}")
            merged.update(dict(config))
        self.config = merged
        if bool(merged["enabled"]) and merged["min_num_faces"] is None:
            raise ValueError(
                "Fragment face threshold is enabled but min_num_faces is not configured."
            )
        if bool(merged["enabled"]) and not bool(merged["exclude_entire_fragment"]):
            raise ValueError("training fragment filter requires exclude_entire_fragment=True")
        for key in ("min_num_faces", "max_num_faces", "min_num_vertices"):
            value = merged[key]
            if value is not None and int(value) < 0:
                raise ValueError(f"{key} must be non-negative or None")
        for key in ("min_surface_area_m2", "min_bbox_diagonal_m"):
            value = merged[key]
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{key} must be finite, non-negative, or None")
        for key in ("missing_mesh_policy", "manifest_mismatch_policy"):
            if merged[key] not in {"error", "exclude", "report_only"}:
                raise ValueError(f"unsupported {key}={merged[key]!r}")
        self._decisions: list[FragmentFilterDecision] = []

    def evaluate(self, metadata: FragmentMeshMetadata) -> FragmentFilterDecision:
        reasons: list[str] = []
        if bool(self.config["enabled"]):
            minimum_faces = self.config["min_num_faces"]
            maximum_faces = self.config["max_num_faces"]
            minimum_vertices = self.config["min_num_vertices"]
            minimum_area = self.config["min_surface_area_m2"]
            minimum_diagonal = self.config["min_bbox_diagonal_m"]
            if minimum_faces is not None and metadata.num_faces < int(minimum_faces):
                reasons.append("physical_fragment_num_faces_below_min")
            if maximum_faces is not None and metadata.num_faces > int(maximum_faces):
                reasons.append("physical_fragment_num_faces_above_max")
            if minimum_vertices is not None and metadata.num_vertices < int(minimum_vertices):
                reasons.append("physical_fragment_num_vertices_below_min")
            if minimum_area is not None and metadata.surface_area_m2 < float(minimum_area):
                reasons.append("physical_fragment_surface_area_below_min")
            if (
                minimum_diagonal is not None
                and metadata.bbox_diagonal_m < float(minimum_diagonal)
            ):
                reasons.append("physical_fragment_bbox_diagonal_below_min")
        decision = FragmentFilterDecision(not reasons, reasons, metadata)
        return decision

    def filter_fragments(
        self, metadata_by_id: Mapping[tuple[str, int], FragmentMeshMetadata]
    ) -> dict[tuple[str, int], FragmentFilterDecision]:
        decisions = {
            key: self.evaluate(metadata)
            for key, metadata in sorted(metadata_by_id.items())
        }
        self._decisions = list(decisions.values())
        return decisions

    def explain(self, metadata: FragmentMeshMetadata) -> str:
        decision = self.evaluate(metadata)
        return "accepted" if decision.accepted else ";".join(decision.reasons)

    def to_report(self) -> dict[str, Any]:
        accepted = sum(decision.accepted for decision in self._decisions)
        return {
            "config": dict(self.config),
            "total_physical_fragments": len(self._decisions),
            "accepted_physical_fragments": accepted,
            "rejected_physical_fragments": len(self._decisions) - accepted,
            "decisions": [decision.to_dict() for decision in self._decisions],
        }


def _mesh_metadata(
    *,
    scene_id: str,
    fragment_id: int,
    mesh_path: Path,
    annotation_path: Path,
    annotation_sha256: str,
) -> FragmentMeshMetadata:
    stat = mesh_path.stat()
    mesh_hash = sha256_file(mesh_path)
    mesh = load_ply(mesh_path)
    points = np.asarray(mesh["points"], dtype=np.float64)
    polygons = [
        np.asarray(values, dtype=np.int64)
        for values in mesh.get("face_polygons", [])
    ]
    if not polygons and mesh.get("faces") is not None:
        polygons = [np.asarray(values, dtype=np.int64) for values in mesh["faces"]]
    if not polygons:
        raise ValueError(f"fragment mesh has no polygon faces: {mesh_path}")
    distribution: dict[str, int] = {}
    area = 0.0
    for polygon in polygons:
        size = len(polygon)
        distribution[str(size)] = distribution.get(str(size), 0) + 1
        if size < 3:
            continue
        first = points[polygon[0]]
        for index in range(1, size - 1):
            second = points[polygon[index]]
            third = points[polygon[index + 1]]
            area += 0.5 * float(np.linalg.norm(np.cross(second - first, third - first)))
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    return FragmentMeshMetadata(
        scene_id=scene_id,
        fragment_id=fragment_id,
        fragment_key=f"{scene_id}:fragment_{fragment_id:04d}",
        mesh_path=mesh_path.resolve(),
        num_vertices=len(points),
        num_faces=len(polygons),
        surface_area_m2=area,
        bbox_min=tuple(float(value) for value in bbox_min),
        bbox_max=tuple(float(value) for value in bbox_max),
        bbox_diagonal_m=float(np.linalg.norm(bbox_max - bbox_min)),
        sha256=mesh_hash,
        polygon_size_distribution=distribution,
        file_size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        annotation_path=annotation_path.resolve(),
        annotation_sha256=annotation_sha256,
    )


class FragmentMeshMetadataCache:
    """Persistent cache whose hits still verify file SHA256 as required."""

    def __init__(self, dataset_root: str | Path, cache_dir: str | Path) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        root_hash = hashlib.sha256(str(self.dataset_root).encode("utf-8")).hexdigest()[:16]
        self.path = Path(cache_dir).expanduser().resolve() / (
            f"fragment_mesh_metadata_{root_hash}.json"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            payload = _json(self.path)
            if Path(str(payload.get("dataset_root", ""))).resolve() == self.dataset_root:
                self._entries = dict(payload.get("entries", {}))

    def get_or_compute(
        self,
        *,
        scene_id: str,
        fragment_id: int,
        mesh_path: Path,
        annotation_path: Path,
        annotation_sha256: str,
    ) -> FragmentMeshMetadata:
        key = f"{scene_id}:{fragment_id}"
        stat = mesh_path.stat()
        current_sha = sha256_file(mesh_path)
        cached = self._entries.get(key)
        if cached is not None:
            metadata = FragmentMeshMetadata.from_dict(cached)
            valid = (
                metadata.mesh_path == mesh_path.resolve()
                and metadata.file_size == int(stat.st_size)
                and metadata.mtime_ns == int(stat.st_mtime_ns)
                and metadata.sha256 == current_sha
                and metadata.annotation_path == annotation_path.resolve()
                and metadata.annotation_sha256 == annotation_sha256
            )
            if valid:
                self.hits += 1
                return metadata
        self.misses += 1
        metadata = _mesh_metadata(
            scene_id=scene_id,
            fragment_id=fragment_id,
            mesh_path=mesh_path,
            annotation_path=annotation_path,
            annotation_sha256=annotation_sha256,
        )
        self._entries[key] = metadata.to_dict()
        return metadata

    def save(self) -> None:
        payload = {
            "version": 1,
            "dataset_root": str(self.dataset_root),
            "entries": self._entries,
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


def scan_fragment_mesh_metadata(
    dataset_root: str | Path,
    *,
    filter_config: Mapping[str, Any] | None = None,
    cache_dir: str | Path = "work_dirs/cache",
) -> tuple[
    dict[tuple[str, int], FragmentMeshMetadata],
    dict[str, Any],
]:
    root = Path(dataset_root).expanduser().resolve()
    config = dict(DEFAULT_FRAGMENT_MESH_FILTER)
    if filter_config:
        config.update(dict(filter_config))
    cache = (
        FragmentMeshMetadataCache(root, cache_dir)
        if bool(config.get("cache_metadata"))
        else None
    )
    metadata_by_id: dict[tuple[str, int], FragmentMeshMetadata] = {}
    annotation_hashes: dict[str, str] = {}
    for scene_dir in sorted(root.glob("scene_*")):
        annotation_path = scene_dir / "fragments" / "fragment_annotations.json"
        if not annotation_path.is_file():
            if config.get("missing_mesh_policy") == "error":
                raise FileNotFoundError(annotation_path)
            continue
        annotation_sha = sha256_file(annotation_path)
        annotation_hashes[str(annotation_path.resolve())] = annotation_sha
        payload = _json(annotation_path)
        scene_id = scene_dir.name
        for entry in payload.get("fragments", []):
            if not isinstance(entry, Mapping) or "fragment_id" not in entry:
                continue
            fragment_id = int(entry["fragment_id"])
            mesh_value = entry.get("mesh")
            mesh_path = (
                (scene_dir / str(mesh_value)).resolve()
                if isinstance(mesh_value, str)
                else (scene_dir / "fragments" / f"fragment_{fragment_id:04d}.ply").resolve()
            )
            if not mesh_path.is_file():
                if config.get("missing_mesh_policy") == "error":
                    raise FileNotFoundError(
                        f"physical fragment mesh is required but missing: {mesh_path}"
                    )
                continue
            metadata = (
                cache.get_or_compute(
                    scene_id=scene_id,
                    fragment_id=fragment_id,
                    mesh_path=mesh_path,
                    annotation_path=annotation_path,
                    annotation_sha256=annotation_sha,
                )
                if cache is not None
                else _mesh_metadata(
                    scene_id=scene_id,
                    fragment_id=fragment_id,
                    mesh_path=mesh_path,
                    annotation_path=annotation_path,
                    annotation_sha256=annotation_sha,
                )
            )
            if config.get("manifest_mismatch_policy") == "error":
                annotated_vertices = entry.get("num_vertices")
                annotated_faces = entry.get("num_faces")
                if annotated_vertices is not None and int(annotated_vertices) != metadata.num_vertices:
                    raise ValueError(
                        f"fragment manifest num_vertices mismatch for {metadata.fragment_key}"
                    )
                if annotated_faces is not None and int(annotated_faces) != metadata.num_faces:
                    raise ValueError(
                        f"fragment manifest num_faces mismatch for {metadata.fragment_key}"
                    )
            key = (scene_id, fragment_id)
            if key in metadata_by_id:
                raise ValueError(f"duplicate physical fragment metadata for {key}")
            metadata_by_id[key] = metadata
    if cache is not None:
        cache.save()
    report = {
        "dataset_root": str(root),
        "annotation_sha256": annotation_hashes,
        "fragment_mesh_cache_path": str(cache.path) if cache is not None else None,
        "fragment_mesh_cache_hits": cache.hits if cache is not None else 0,
        "fragment_mesh_cache_misses": cache.misses if cache is not None else len(metadata_by_id),
        "num_physical_fragments": len(metadata_by_id),
    }
    return metadata_by_id, report


__all__ = [
    "DEFAULT_FRAGMENT_MESH_FILTER",
    "FragmentFilterDecision",
    "FragmentMeshFilter",
    "FragmentMeshMetadata",
    "FragmentMeshMetadataCache",
    "REQUIRED_TEST_SPLIT_FLAGS",
    "scan_fragment_mesh_metadata",
    "sha256_file",
]
