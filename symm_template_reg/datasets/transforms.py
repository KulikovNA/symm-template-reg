"""Point selection policies that preserve all row-wise correspondences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OBSERVED_POLICIES = {
    "all_points",
    "voxel_downsample",
    "random_up_to_max",
    "farthest_point_up_to_max",
    "precomputed_dataset_points",
}


def farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic O(N*K) farthest-point sampling without optional packages."""

    points = np.asarray(points, dtype=np.float32)
    count = min(max(int(count), 0), len(points))
    if count == len(points):
        return np.arange(len(points), dtype=np.int64)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    center = points.mean(axis=0, dtype=np.float64)
    first = int(np.argmax(np.sum((points - center) ** 2, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    min_distance = np.full(len(points), np.inf, dtype=np.float32)
    current = first
    for index in range(count):
        selected[index] = current
        distance = np.sum((points - points[current]) ** 2, axis=1)
        np.minimum(min_distance, distance, out=min_distance)
        min_distance[selected[: index + 1]] = -1.0
        current = int(np.argmax(min_distance))
    return selected


def voxel_downsample_indices(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Keep one point per occupied voxel in deterministic lexicographic order."""

    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    if not len(points):
        return np.empty(0, dtype=np.int64)
    origin = np.min(points, axis=0)
    keys = np.floor((points - origin) / float(voxel_size_m)).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return np.sort(indices.astype(np.int64, copy=False))


def geometric_cap_indices(points: np.ndarray, max_points: int) -> np.ndarray:
    """Cap a cloud geometrically; never use raster-order first/linspace selection."""

    if len(points) <= max_points:
        return np.arange(len(points), dtype=np.int64)
    return np.sort(farthest_point_indices(points, max_points))


@dataclass(frozen=True)
class ObservedPointSelector:
    policy: str = "precomputed_dataset_points"
    min_points: int = 128
    max_points: int | None = 4096
    voxel_size_m: float = 0.002
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.policy not in OBSERVED_POLICIES:
            raise ValueError(
                f"unknown observed policy {self.policy!r}; expected one of {sorted(OBSERVED_POLICIES)}"
            )
        if self.min_points < 0:
            raise ValueError("min_points cannot be negative")
        if self.max_points is not None and self.max_points <= 0:
            raise ValueError("max_points must be positive or None")
        if self.max_points is not None and self.min_points > self.max_points:
            raise ValueError("min_points cannot exceed max_points")

    def indices(self, points: np.ndarray, *, sample_seed: int = 0) -> np.ndarray:
        points = np.asarray(points)
        size = len(points)
        if size < self.min_points:
            raise ValueError(
                f"sample has {size} observed points, below min_points={self.min_points}"
            )
        if self.policy == "all_points":
            return np.arange(size, dtype=np.int64)
        if self.policy == "voxel_downsample":
            indices = voxel_downsample_indices(points, self.voxel_size_m)
            if self.max_points is not None and len(indices) > self.max_points:
                local = farthest_point_indices(points[indices], self.max_points)
                indices = indices[local]
            if len(indices) < self.min_points:
                target = size if self.max_points is None else min(self.max_points, size)
                indices = farthest_point_indices(points, target)
            return np.sort(indices.astype(np.int64, copy=False))
        if self.max_points is None or size <= self.max_points:
            return np.arange(size, dtype=np.int64)
        if self.policy == "random_up_to_max":
            rng = np.random.default_rng(self.random_seed + int(sample_seed))
            return np.sort(rng.choice(size, size=self.max_points, replace=False).astype(np.int64))
        if self.policy == "farthest_point_up_to_max":
            return np.sort(farthest_point_indices(points, self.max_points))
        # The NPZ rows are raster ordered.  The precomputed policy preserves them
        # below the cap, but deliberately uses a geometric cap above it.
        return geometric_cap_indices(points, self.max_points)


__all__ = [
    "OBSERVED_POLICIES",
    "ObservedPointSelector",
    "farthest_point_indices",
    "geometric_cap_indices",
    "voxel_downsample_indices",
]
