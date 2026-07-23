"""Train-only image-space shell-boundary erosion and gated dilation."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import torch

from symm_template_reg.geometry.triangle_surface import (
    closest_points_on_triangle_mesh,
)


DEFAULT_BOUNDARY_AUGMENTATION = {
    "enabled": True,
    "apply_probability": 0.6,
    "mode": "mixed",
    "mode_probabilities": {
        "none": 0.10,
        "erode": 0.35,
        "dilate": 0.35,
        "mixed": 0.20,
    },
    "radius_px": {"min": 1, "max": 2},
    "max_removed_fraction": 0.08,
    "max_added_fraction": 0.05,
    "min_points_after_augmentation": 128,
    "partial_boundary_probability": 0.7,
    "boundary_arc_fraction_range": [0.15, 0.50],
    "max_pseudo_target_distance_m": 0.002,
    "max_local_depth_difference_m": 0.010,
    "local_depth_window_px": 3,
    "include_fracture_candidates": True,
    "include_depth_ring_candidates": True,
}


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    output = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    target_y0, target_y1 = source_y0 + dy, source_y1 + dy
    target_x0, target_x1 = source_x0 + dx, source_x1 + dx
    output[target_y0:target_y1, target_x0:target_x1] = mask[
        source_y0:source_y1, source_x0:source_x1
    ]
    return output


def _disk_offsets(radius: int) -> list[tuple[int, int]]:
    return [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy * dy + dx * dx <= radius * radius
    ]


def binary_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    for dy, dx in _disk_offsets(int(radius)):
        result |= _shift(mask, dy, dx)
    return result


def binary_erosion(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.ones_like(mask, dtype=bool)
    for dy, dx in _disk_offsets(int(radius)):
        result &= _shift(mask, dy, dx)
    return result


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask)
    best: list[tuple[int, int]] = []
    for start_y, start_x in np.argwhere(mask):
        if seen[start_y, start_x]:
            continue
        component: list[tuple[int, int]] = []
        queue = deque([(int(start_y), int(start_x))])
        seen[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if (
                    0 <= yy < mask.shape[0]
                    and 0 <= xx < mask.shape[1]
                    and mask[yy, xx]
                    and not seen[yy, xx]
                ):
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        if len(component) > len(best):
            best = component
    output = np.zeros_like(mask)
    for y, x in best:
        output[y, x] = True
    return output


def _backproject(
    u: np.ndarray, v: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    z = depth_m[v, u].astype(np.float32)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return np.stack((x, y, z), axis=-1)


def _transform_inverse(points_C: np.ndarray, T_C_from_O: np.ndarray) -> np.ndarray:
    rotation = T_C_from_O[:3, :3].astype(np.float32)
    translation = T_C_from_O[:3, 3].astype(np.float32)
    return (points_C - translation) @ rotation


class BoundaryMaskAugmentation:
    """Apply bounded boundary corruption while preserving valid correspondences."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        merged = deepcopy(DEFAULT_BOUNDARY_AUGMENTATION)
        for key, value in dict(config or {}).items():
            if key not in merged:
                raise ValueError(f"unknown boundary augmentation field: {key}")
            if isinstance(merged[key], dict) and isinstance(value, Mapping):
                unknown = set(value).difference(merged[key])
                if unknown:
                    raise ValueError(
                        f"unknown {key} fields: {sorted(unknown)}"
                    )
                merged[key].update(dict(value))
            else:
                merged[key] = value
        if merged["mode"] not in {"none", "erode", "dilate", "mixed", "random"}:
            raise ValueError("mode must be none, erode, dilate, mixed, or random")
        if not 0 <= float(merged["max_removed_fraction"]) <= 1:
            raise ValueError("max_removed_fraction must be in [0, 1]")
        if not 0 <= float(merged["max_added_fraction"]) <= 1:
            raise ValueError("max_added_fraction must be in [0, 1]")
        self.config = merged

    def _choose_mode(self, rng: np.random.Generator) -> str:
        if not self.config["enabled"]:
            return "none"
        if rng.random() > float(self.config["apply_probability"]):
            return "none"
        configured = str(self.config["mode"])
        if configured != "random":
            return configured
        probabilities = dict(self.config["mode_probabilities"])
        modes = ("none", "erode", "dilate", "mixed")
        values = np.asarray([float(probabilities[name]) for name in modes])
        if np.any(values < 0) or values.sum() <= 0:
            raise ValueError("mode probabilities must be non-negative with positive sum")
        return str(rng.choice(modes, p=values / values.sum()))

    def apply(
        self,
        *,
        shell_points_C: np.ndarray,
        shell_targets_O: np.ndarray,
        shell_uv: np.ndarray,
        shell_mask: np.ndarray,
        instance_mask: np.ndarray,
        surface_mask: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        instance_value: int,
        T_C_from_O: np.ndarray,
        template_vertices_O: np.ndarray,
        template_faces: np.ndarray,
        seed: int,
        epoch: int = 0,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        mode = self._choose_mode(rng)
        original_points = np.asarray(shell_points_C, dtype=np.float32)
        original_targets = np.asarray(shell_targets_O, dtype=np.float32)
        original_uv = np.asarray(shell_uv, dtype=np.int64)
        if (
            original_points.shape != original_targets.shape
            or original_points.ndim != 2
            or original_points.shape[1] != 3
            or original_uv.shape != (len(original_points), 2)
        ):
            raise ValueError("shell points, targets and uv are not row-aligned")
        radius_cfg = self.config["radius_px"]
        radius = int(
            rng.integers(int(radius_cfg["min"]), int(radius_cfg["max"]) + 1)
        )
        shell_mask = np.asarray(shell_mask, dtype=bool)
        boundary = shell_mask & ~binary_erosion(shell_mask, radius)
        retained = np.ones(len(original_points), dtype=bool)
        remove_rows = np.empty(0, dtype=np.int64)
        max_remove = int(
            np.floor(len(original_points) * float(self.config["max_removed_fraction"]))
        )
        max_remove = min(
            max_remove,
            max(
                0,
                len(original_points)
                - int(self.config["min_points_after_augmentation"]),
            ),
        )
        if mode in {"erode", "mixed"} and max_remove > 0:
            selected_boundary = boundary.copy()
            if (
                np.any(boundary)
                and rng.random()
                < float(self.config["partial_boundary_probability"])
            ):
                coordinates = np.argwhere(boundary)
                pivot = coordinates[int(rng.integers(len(coordinates)))]
                low, high = map(
                    float, self.config["boundary_arc_fraction_range"]
                )
                fraction = float(rng.uniform(low, high))
                count = max(1, int(np.ceil(len(coordinates) * fraction)))
                distance2 = np.sum((coordinates - pivot) ** 2, axis=1)
                chosen = coordinates[np.argsort(distance2)[:count]]
                selected_boundary = np.zeros_like(boundary)
                selected_boundary[chosen[:, 0], chosen[:, 1]] = True
            u, v = original_uv[:, 0], original_uv[:, 1]
            candidates = np.flatnonzero(selected_boundary[v, u])
            if len(candidates):
                remove_rows = np.sort(
                    rng.choice(
                        candidates,
                        size=min(max_remove, len(candidates)),
                        replace=False,
                    )
                )
                retained[remove_rows] = False
                after_pixel_mask = shell_mask.copy()
                after_pixel_mask[v[remove_rows], u[remove_rows]] = False
                largest = largest_connected_component(after_pixel_mask)
                lcc_retained = largest[v, u]
                extra_removed = int((retained & ~lcc_retained).sum())
                if (
                    retained.sum() - extra_removed
                    >= int(self.config["min_points_after_augmentation"])
                    and (~lcc_retained).sum() <= max_remove
                ):
                    retained &= lcc_retained
                    remove_rows = np.flatnonzero(~retained)

        retained_points = original_points[retained]
        retained_targets = original_targets[retained]
        retained_uv = original_uv[retained]
        source_labels = np.zeros(len(retained_points), dtype=np.int64)
        added_points: list[np.ndarray] = []
        added_targets: list[np.ndarray] = []
        added_uv: list[np.ndarray] = []
        added_sources: list[str] = []
        rejected_points: list[np.ndarray] = []
        projection_distances: list[float] = []
        fracture_total = fracture_depth_rejected = fracture_template_rejected = 0
        max_add = int(
            np.floor(len(original_points) * float(self.config["max_added_fraction"]))
        )
        if mode in {"dilate", "mixed"} and max_add > 0:
            outer_ring = binary_dilation(shell_mask, radius) & ~shell_mask
            fracture_mask = (
                outer_ring
                & (instance_mask == int(instance_value))
                & (surface_mask == 2)
            )
            depth_ring_mask = outer_ring & (surface_mask != 1)
            candidate_rows: list[tuple[int, int, str]] = []
            if bool(self.config["include_fracture_candidates"]):
                candidate_rows.extend(
                    (int(v), int(u), "fracture")
                    for v, u in np.argwhere(fracture_mask)
                )
            if bool(self.config["include_depth_ring_candidates"]):
                candidate_rows.extend(
                    (int(v), int(u), "depth_ring")
                    for v, u in np.argwhere(depth_ring_mask & ~fracture_mask)
                )
            rng.shuffle(candidate_rows)
            seen_uv: set[tuple[int, int]] = set()
            window = int(self.config["local_depth_window_px"])
            max_depth_difference = float(
                self.config["max_local_depth_difference_m"]
            )
            accepted_candidates: list[tuple[np.ndarray, np.ndarray, str]] = []
            for v, u, source in candidate_rows:
                if len(accepted_candidates) >= max_add:
                    break
                if (u, v) in seen_uv:
                    continue
                seen_uv.add((u, v))
                if source == "fracture":
                    fracture_total += 1
                depth = float(depth_m[v, u])
                if not np.isfinite(depth) or depth <= 0:
                    if source == "fracture":
                        fracture_depth_rejected += 1
                    continue
                y0, y1 = max(0, v - window), min(depth_m.shape[0], v + window + 1)
                x0, x1 = max(0, u - window), min(depth_m.shape[1], u + window + 1)
                local_shell = shell_mask[y0:y1, x0:x1]
                local_depth = depth_m[y0:y1, x0:x1][local_shell]
                local_depth = local_depth[
                    np.isfinite(local_depth) & (local_depth > 0)
                ]
                if (
                    not len(local_depth)
                    or float(np.min(np.abs(local_depth - depth)))
                    > max_depth_difference
                ):
                    if source == "fracture":
                        fracture_depth_rejected += 1
                    continue
                point_C = _backproject(
                    np.asarray([u]),
                    np.asarray([v]),
                    depth_m,
                    intrinsics,
                )[0]
                if not np.isfinite(point_C).all():
                    if source == "fracture":
                        fracture_depth_rejected += 1
                    continue
                raw_O = _transform_inverse(
                    point_C[None], np.asarray(T_C_from_O, dtype=np.float32)
                )
                projection = closest_points_on_triangle_mesh(
                    torch.from_numpy(raw_O),
                    torch.as_tensor(template_vertices_O, dtype=torch.float32),
                    torch.as_tensor(template_faces, dtype=torch.long),
                    point_chunk_size=1,
                )
                distance = float(projection["distances"][0])
                if distance > float(
                    self.config["max_pseudo_target_distance_m"]
                ):
                    rejected_points.append(point_C)
                    if source == "fracture":
                        fracture_template_rejected += 1
                    continue
                accepted_candidates.append(
                    (
                        point_C,
                        projection["points"][0].cpu().numpy(),
                        source,
                    )
                )
                projection_distances.append(distance)
                added_uv.append(np.asarray((u, v), dtype=np.int64))
            for point_C, target_O, source in accepted_candidates:
                added_points.append(point_C.astype(np.float32))
                added_targets.append(target_O.astype(np.float32))
                added_sources.append(source)

        if added_points:
            points = np.concatenate(
                (retained_points, np.stack(added_points)), axis=0
            )
            targets = np.concatenate(
                (retained_targets, np.stack(added_targets)), axis=0
            )
            uv = np.concatenate((retained_uv, np.stack(added_uv)), axis=0)
            source_labels = np.concatenate(
                (
                    source_labels,
                    np.asarray(
                        [1 if source == "fracture" else 2 for source in added_sources],
                        dtype=np.int64,
                    ),
                )
            )
        else:
            points, targets, uv = retained_points, retained_targets, retained_uv
        if len(points) < int(self.config["min_points_after_augmentation"]):
            points, targets, uv = original_points, original_targets, original_uv
            source_labels = np.zeros(len(points), dtype=np.int64)
            retained = np.ones(len(original_points), dtype=bool)
            remove_rows = np.empty(0, dtype=np.int64)
            added_points, added_sources, projection_distances = [], [], []
            mode = "none"
        removed = int((~retained).sum())
        added = len(added_points)
        after_mask = shell_mask.copy()
        if len(remove_rows):
            after_mask[
                original_uv[remove_rows, 1], original_uv[remove_rows, 0]
            ] = False
        if len(added_uv):
            added_array = np.stack(added_uv)
            after_mask[added_array[:, 1], added_array[:, 0]] = True
        distances_mm = np.asarray(projection_distances) * 1000.0
        metadata = {
            "augmentation_applied": mode != "none",
            "augmentation_mode": mode,
            "radius_px": radius,
            "original_shell_count": len(original_points),
            "removed_shell_count": removed,
            "added_total_count": added,
            "added_fracture_count": added_sources.count("fracture"),
            "added_depth_ring_count": added_sources.count("depth_ring"),
            "final_point_count": len(points),
            "removed_fraction": removed / max(len(original_points), 1),
            "added_fraction": added / max(len(original_points), 1),
            "max_template_projection_distance_mm": (
                float(distances_mm.max()) if len(distances_mm) else 0.0
            ),
            "mean_template_projection_distance_mm": (
                float(distances_mm.mean()) if len(distances_mm) else 0.0
            ),
            "random_seed": int(seed),
            "epoch": int(epoch),
            "fracture_candidates_total": fracture_total,
            "fracture_candidates_accepted": added_sources.count("fracture"),
            "fracture_candidates_rejected_by_template_distance": (
                fracture_template_rejected
            ),
            "fracture_candidates_rejected_by_depth": fracture_depth_rejected,
            "gt_pose_usage": "train_target_construction_only",
        }
        return {
            "points_C": points.astype(np.float32, copy=False),
            "target_points_O": targets.astype(np.float32, copy=False),
            "pixel_uv": uv.astype(np.int64, copy=False),
            "source_labels": source_labels,
            "metadata": metadata,
            "debug": {
                "before_mask": shell_mask,
                "after_mask": after_mask,
                "boundary": boundary,
                "retained_points_C": original_points[retained],
                "removed_points_C": original_points[~retained],
                "added_points_C": (
                    np.stack(added_points)
                    if added_points
                    else np.empty((0, 3), dtype=np.float32)
                ),
                "added_sources": list(added_sources),
                "rejected_points_C": (
                    np.stack(rejected_points)
                    if rejected_points
                    else np.empty((0, 3), dtype=np.float32)
                ),
            },
        }


__all__ = [
    "BoundaryMaskAugmentation",
    "DEFAULT_BOUNDARY_AUGMENTATION",
    "binary_dilation",
    "binary_erosion",
    "largest_connected_component",
]
