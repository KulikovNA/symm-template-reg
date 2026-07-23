"""Small on-disk fixture matching the generated dataset's actual joins."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_ascii_tetrahedron(path: Path, *, with_normals: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normal_properties = "property float nx\nproperty float ny\nproperty float nz\n" if with_normals else ""
    normal_values = " 0 0 1" if with_normals else ""
    vertices = [
        f"0 0 0{normal_values}",
        f"1 0 0{normal_values}",
        f"0 1 0{normal_values}",
        f"0 0 1{normal_values}",
    ]
    payload = (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"{normal_properties}"
        "element face 4\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
        + "\n".join(vertices)
        + "\n3 0 2 1\n3 0 1 3\n3 0 3 2\n3 1 2 3\n"
    )
    path.write_text(payload, encoding="ascii")


def build_dataset(root: Path, *, with_sidecar: bool = False) -> Path:
    model_id = "object_000004__scale_0p05"
    model_path = root / "models" / f"{model_id}.ply"
    write_ascii_tetrahedron(model_path)
    _write_json(
        model_path.with_suffix(".meta.json"),
        {
            "object_id": "object_000004",
            "mesh": model_path.name,
            "coordinate_frame": "O",
            "units": "scene_unit",
        },
    )
    if with_sidecar:
        _write_json(
            model_path.with_suffix(".symmetry.json"),
            {
                "version": 1,
                "object_model_id": "object_000004",
                "coordinate_frame": "O",
                "axis": {
                    "name": "y",
                    "origin": [0.0, 0.0, 0.0],
                    "direction": [0.0, 1.0, 0.0],
                },
                "regions": [
                    {
                        "region_id": "all",
                        "y_min_m": -10.0,
                        "y_max_m": 10.0,
                        "rotation_group": {"type": "C", "order": 4},
                    }
                ],
            },
        )

    scene = root / "scene_000000"
    (scene / "visible_points").mkdir(parents=True)
    for fragment_id in (0, 1):
        fragment_mesh = scene / "fragments" / f"fragment_{fragment_id:04d}.ply"
        write_ascii_tetrahedron(fragment_mesh)
    object_points_0 = np.asarray(
        [[0.00, 0.00, 0.00], [0.01, 0.01, 0.00], [0.02, 0.02, 0.00]],
        dtype=np.float32,
    )
    object_points_1 = np.asarray(
        [
            [0.00, -0.02, 0.00],
            [0.01, -0.01, 0.00],
            [0.02, 0.00, 0.00],
            [0.03, 0.01, 0.00],
            [0.04, 0.02, 0.00],
        ],
        dtype=np.float32,
    )
    transforms = [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]
    transforms[0][:3, 3] = [0.1, 0.0, 0.5]
    transforms[1][:3, 3] = [-0.1, 0.1, 0.6]
    points_C_0 = object_points_0 + transforms[0][:3, 3]
    points_C_1 = object_points_1 + transforms[1][:3, 3]
    points_O = np.concatenate([object_points_0, object_points_1])
    points_C = np.concatenate([points_C_0, points_C_1])
    fragment_ids = np.asarray([0] * len(object_points_0) + [1] * len(object_points_1), dtype=np.int32)
    labels = np.asarray([0, 1, 0, 0, 0, 1, 0, 0], dtype=np.uint8)
    np.savez(
        scene / "visible_points" / "frame_000000.npz",
        u=np.arange(len(points_C), dtype=np.int32),
        v=np.zeros(len(points_C), dtype=np.int32),
        fragment_id=fragment_ids,
        surface_label=labels,
        points_C=points_C,
        points_F=points_O,
        points_O=points_O,
        face_id=np.zeros(len(points_C), dtype=np.int32),
        barycentric=np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (len(points_C), 1)),
        shell_indices=np.flatnonzero(labels == 0).astype(np.int32),
        fracture_indices=np.flatnonzero(labels == 1).astype(np.int32),
    )
    fragments = [
        {"fragment_id": index, "T_C_from_O": transform.tolist()}
        for index, transform in enumerate(transforms)
    ]
    _write_json(
        scene / "gt_annotations.json",
        {
            "scene_id": "scene_000000",
            "frames": [
                {
                    "frame_id": 0,
                    "visible_points": "visible_points/frame_000000.npz",
                    "fragments": fragments,
                }
            ],
        },
    )
    _write_json(
        scene / "scene_meta.json",
        {
            "scene_id": "scene_000000",
            "split": root.name,
            "object_id": "object_000004",
            "object_model": f"../models/{model_id}.ply",
            "units": "scene_unit",
            "matrix_convention": "BOP/OpenCV, column-vector homogeneous transforms",
        },
    )
    _write_json(
        scene / "fragments" / "fragment_annotations.json",
        {
            "object_id": "object_000004",
            "object_model": f"../../models/{model_id}.ply",
            "fragments": [
                {
                    "fragment_id": fragment_id,
                    "name": f"fragment_{fragment_id:04d}",
                    "mesh": f"fragments/fragment_{fragment_id:04d}.ply",
                    "num_vertices": 4,
                    "num_faces": 4,
                }
                for fragment_id in (0, 1)
            ],
        },
    )
    return root
