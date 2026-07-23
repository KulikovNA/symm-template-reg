from __future__ import annotations

from copy import deepcopy

import numpy as np

from symm_template_reg.datasets.boundary_augmentation import (
    BoundaryMaskAugmentation,
)


def case() -> dict:
    height = width = 13
    shell_mask = np.zeros((height, width), dtype=bool)
    shell_mask[4:9, 4:9] = True
    instance = np.zeros((height, width), dtype=np.uint16)
    instance[2:11, 2:11] = 7
    surface = np.zeros((height, width), dtype=np.uint8)
    surface[shell_mask] = 1
    surface[3, 4:9] = 2
    depth = np.zeros((height, width), dtype=np.float32)
    depth[1:12, 1:12] = 1.0
    intrinsics = np.asarray(
        [[10.0, 0.0, 6.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    vu = np.argwhere(shell_mask)
    uv = vu[:, ::-1].copy()
    z = depth[vu[:, 0], vu[:, 1]]
    points_C = np.stack(
        (
            (uv[:, 0] - 6.0) * z / 10.0,
            (uv[:, 1] - 6.0) * z / 10.0,
            z,
        ),
        axis=-1,
    ).astype(np.float32)
    T_C_from_O = np.eye(4, dtype=np.float32)
    T_C_from_O[2, 3] = 1.0
    targets_O = points_C.copy()
    targets_O[:, 2] -= 1.0
    vertices = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return {
        "shell_points_C": points_C,
        "shell_targets_O": targets_O,
        "shell_uv": uv,
        "shell_mask": shell_mask,
        "instance_mask": instance,
        "surface_mask": surface,
        "depth_m": depth,
        "intrinsics": intrinsics,
        "instance_value": 7,
        "T_C_from_O": T_C_from_O,
        "template_vertices_O": vertices,
        "template_faces": faces,
        "seed": 17,
        "epoch": 3,
    }


def run(mode: str, **overrides):
    config = {
        "enabled": True,
        "apply_probability": 1.0,
        "mode": mode,
        "radius_px": {"min": 1, "max": 1},
        "max_removed_fraction": 0.20,
        "max_added_fraction": 0.20,
        "min_points_after_augmentation": 5,
        "partial_boundary_probability": 0.0,
        "max_pseudo_target_distance_m": 0.002,
        "max_local_depth_difference_m": 0.01,
        "local_depth_window_px": 2,
    }
    config.update(deepcopy(overrides))
    return BoundaryMaskAugmentation(config).apply(**case())
