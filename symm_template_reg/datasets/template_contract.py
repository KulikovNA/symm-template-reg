"""Cross-split contract for the single production template and symmetry sidecar."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .template_repository import compute_vertex_normals, load_ply


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_template_sha256(path: str | Path) -> str:
    """Hash exact vertices/faces; stored normals are not authoritative."""

    mesh = load_ply(Path(path))
    digest = hashlib.sha256()
    for name, value, dtype in (
        ("vertices", mesh["points"], "<f4"),
        ("faces", mesh["faces"], "<i8"),
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def deterministic_template_normals(path: str | Path) -> np.ndarray:
    mesh = load_ply(Path(path))
    normals = compute_vertex_normals(mesh["points"], mesh["faces"])
    if normals is None:
        raise ValueError(f"production template has no triangular faces: {path}")
    return normals


def inspect_template_contract(dataset_root: str | Path) -> dict[str, Any]:
    """Validate one semantic template and sidecar across train/val/test."""

    root = Path(dataset_root).expanduser().resolve()
    rows: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        models = root / split / "models"
        templates = sorted(models.glob("object_*.ply"))
        if len(templates) != 1:
            raise ValueError(
                f"{split} requires exactly one object_*.ply; found {len(templates)}"
            )
        mesh_path = templates[0]
        sidecar_path = mesh_path.with_suffix(".symmetry.json")
        meta_path = mesh_path.with_suffix(".meta.json")
        if not sidecar_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"{split} template requires .symmetry.json and .meta.json"
            )
        mesh = load_ply(mesh_path)
        points = np.asarray(mesh["points"], dtype=np.float32)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        rows[split] = {
            "template_path": str(mesh_path),
            "template_sha256": semantic_template_sha256(mesh_path),
            "template_file_sha256": sha256_file(mesh_path),
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "metadata_path": str(meta_path),
            "metadata_file_sha256": sha256_file(meta_path),
            "object_id": metadata.get("object_id"),
            "units": metadata.get("units", metadata.get("coordinate_unit")),
            "object_frame": metadata.get(
                "coordinate_frame", metadata.get("object_frame")
            ),
            "bbox_min": points.min(axis=0).tolist(),
            "bbox_max": points.max(axis=0).tolist(),
        }
    exact_fields = (
        "template_sha256",
        "sidecar_sha256",
        "object_id",
        "units",
        "object_frame",
    )
    field_matches = {
        field: len({json.dumps(row[field], sort_keys=True) for row in rows.values()})
        == 1
        for field in exact_fields
    }
    bbox_matches = all(
        np.array_equal(
            np.asarray(rows["train"][field], dtype=np.float32),
            np.asarray(rows[split][field], dtype=np.float32),
        )
        for split in ("val", "test")
        for field in ("bbox_min", "bbox_max")
    )
    raw_file_hash_matches = (
        len({row["template_file_sha256"] for row in rows.values()}) == 1
    )
    passed = all(field_matches.values()) and bbox_matches
    report = {
        "dataset_root": str(root),
        "hash_semantics": (
            "template_sha256 hashes exact float32 vertices and int64 faces; "
            "stored vertex normals are recomputed deterministically"
        ),
        "splits": rows,
        "field_matches": field_matches,
        "bbox_matches": bbox_matches,
        "raw_template_file_hash_matches": raw_file_hash_matches,
        "raw_template_file_hash_warning": (
            None
            if raw_file_hash_matches
            else "PLY bytes differ only outside the authoritative vertices/faces hash."
        ),
        "passed": passed,
    }
    if not passed:
        raise ValueError(f"cross-split template contract failed: {report}")
    return report


__all__ = [
    "deterministic_template_normals",
    "inspect_template_contract",
    "semantic_template_sha256",
    "sha256_file",
]
