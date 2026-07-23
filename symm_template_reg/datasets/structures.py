"""Variable-length point batch structures used by datasets and models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


def _move_value(value: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(value, Tensor):
        return value.to(*args, **kwargs)
    if isinstance(value, dict):
        return {key: _move_value(item, *args, **kwargs) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value(item, *args, **kwargs) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, *args, **kwargs) for item in value)
    return value


@dataclass
class PackedPointBatch:
    """A compact batch of point sets with optional point-aligned attributes.

    ``offsets`` follows the common prefix-sum convention and therefore has
    ``batch_size + 1`` entries: ``[0, N_0, N_0 + N_1, ...]``.
    """

    points: Tensor
    batch_indices: Tensor
    offsets: Tensor
    lengths: Tensor
    features: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.lengths.numel())

    @property
    def device(self) -> torch.device:
        return self.points.device

    @property
    def dtype(self) -> torch.dtype:
        return self.points.dtype

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, key: str) -> Tensor:
        """Allow concise access to point-aligned features."""

        if key == "points":
            return self.points
        return self.features[key]

    def get(self, key: str, default: Any = None) -> Any:
        if key == "points":
            return self.points
        return self.features.get(key, default)

    def to(self, *args: Any, **kwargs: Any) -> "PackedPointBatch":
        """Return a copy moved with the usual :meth:`torch.Tensor.to` rules."""

        moved_points = self.points.to(*args, **kwargs)
        moved_features: dict[str, Tensor] = {}
        for key, value in self.features.items():
            if value.is_floating_point() or value.is_complex():
                moved_features[key] = value.to(*args, **kwargs)
            else:
                moved_features[key] = value.to(device=moved_points.device)
        return PackedPointBatch(
            points=moved_points,
            batch_indices=self.batch_indices.to(device=moved_points.device),
            offsets=self.offsets.to(device=moved_points.device),
            lengths=self.lengths.to(device=moved_points.device),
            features=moved_features,
        )

    def validate(self) -> "PackedPointBatch":
        """Validate all prefix sums and point-aligned tensor shapes."""

        if not isinstance(self.points, Tensor) or self.points.ndim < 2:
            raise ValueError("points must be a tensor shaped [sum(N_i), C...]")
        total = int(self.points.shape[0])
        for name, value in (
            ("batch_indices", self.batch_indices),
            ("offsets", self.offsets),
            ("lengths", self.lengths),
        ):
            if not isinstance(value, Tensor) or value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional tensor")
            if value.dtype != torch.long:
                raise TypeError(f"{name} must have dtype torch.long")
        if int(self.batch_indices.numel()) != total:
            raise ValueError("batch_indices length must equal number of points")
        if int(self.offsets.numel()) != int(self.lengths.numel()) + 1:
            raise ValueError("offsets must have batch_size + 1 entries")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != total:
            raise ValueError("offsets must start at zero and end at total points")
        expected_offsets = torch.cat(
            [self.lengths.new_zeros(1), self.lengths.cumsum(dim=0)]
        )
        if not torch.equal(self.offsets, expected_offsets):
            raise ValueError("offsets are inconsistent with lengths")
        expected_batch = torch.repeat_interleave(
            torch.arange(self.batch_size, device=self.lengths.device), self.lengths
        )
        if not torch.equal(self.batch_indices, expected_batch):
            raise ValueError("batch_indices are inconsistent with lengths")
        if torch.any(self.lengths < 0):
            raise ValueError("lengths cannot be negative")
        for name, value in self.features.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"feature {name!r} must be a tensor")
            if value.ndim == 0 or int(value.shape[0]) != total:
                raise ValueError(
                    f"feature {name!r} must be point-aligned with first dimension {total}"
                )
        point_valid = self.features.get("valid_mask")
        if point_valid is not None and point_valid.dtype != torch.bool:
            raise TypeError("feature 'valid_mask' must have dtype torch.bool")
        return self

    def split(self, *, with_features: bool = False) -> list[Any]:
        """Split into individual point tensors (or dictionaries with features)."""

        result: list[Any] = []
        for start, end in zip(self.offsets[:-1].tolist(), self.offsets[1:].tolist()):
            if with_features:
                result.append(
                    {
                        "points": self.points[start:end],
                        "features": {
                            key: value[start:end] for key, value in self.features.items()
                        },
                    }
                )
            else:
                result.append(self.points[start:end])
        return result

    def to_padded(
        self,
        *,
        pad_value: float = 0.0,
        feature_pad_values: Mapping[str, float | int | bool] | None = None,
    ) -> dict[str, Any]:
        """Convert to dense tensors and an explicit boolean validity mask."""

        max_length = int(self.lengths.max()) if self.batch_size else 0
        shape = (self.batch_size, max_length, *self.points.shape[1:])
        padded = self.points.new_full(shape, pad_value)
        valid_mask = torch.zeros(
            (self.batch_size, max_length), dtype=torch.bool, device=self.points.device
        )
        padded_features: dict[str, Tensor] = {}
        feature_pad_values = dict(feature_pad_values or {})
        for name, value in self.features.items():
            fill = feature_pad_values.get(name, False if value.dtype == torch.bool else 0)
            padded_features[name] = value.new_full(
                (self.batch_size, max_length, *value.shape[1:]), fill
            )
        for batch_id, (start, end) in enumerate(
            zip(self.offsets[:-1].tolist(), self.offsets[1:].tolist())
        ):
            length = end - start
            padded[batch_id, :length] = self.points[start:end]
            valid_mask[batch_id, :length] = True
            for name, value in self.features.items():
                padded_features[name][batch_id, :length] = value[start:end]
            point_valid = self.features.get("valid_mask")
            if point_valid is not None:
                valid_mask[batch_id, :length] &= point_valid[start:end]
        return {
            "points": padded,
            "valid_mask": valid_mask,
            "lengths": self.lengths.clone(),
            "features": padded_features,
        }

    @classmethod
    def from_padded(
        cls,
        points: Tensor | Mapping[str, Any],
        valid_mask: Tensor | None = None,
        *,
        features: Mapping[str, Tensor] | None = None,
    ) -> "PackedPointBatch":
        """Pack dense points using ``valid_mask``; arbitrary masks are supported."""

        if isinstance(points, Mapping):
            payload = points
            points = payload["points"]
            valid_mask = payload["valid_mask"]
            features = payload.get("features", features)
        if not isinstance(points, Tensor) or points.ndim < 3:
            raise ValueError("padded points must be shaped [B, Nmax, C...]")
        if valid_mask is None or valid_mask.shape != points.shape[:2]:
            raise ValueError("valid_mask must be shaped [B, Nmax]")
        valid_mask = valid_mask.to(dtype=torch.bool, device=points.device)
        lengths = valid_mask.sum(dim=1, dtype=torch.long)
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(dim=0)])
        batch_indices = torch.repeat_interleave(
            torch.arange(points.shape[0], device=points.device), lengths
        )
        packed_features: dict[str, Tensor] = {}
        for name, value in dict(features or {}).items():
            if value.shape[:2] != points.shape[:2]:
                raise ValueError(f"padded feature {name!r} is not aligned with points")
            packed_features[name] = value[valid_mask]
        return cls(
            points=points[valid_mask],
            batch_indices=batch_indices,
            offsets=offsets,
            lengths=lengths,
            features=packed_features,
        )

    @classmethod
    def from_list(
        cls,
        point_sets: Sequence[Tensor],
        *,
        features: Sequence[Mapping[str, Tensor]] | None = None,
    ) -> "PackedPointBatch":
        """Pack a sequence of point tensors and optional aligned attributes."""

        if not point_sets:
            raise ValueError("cannot infer point shape/dtype from an empty sequence")
        lengths = torch.tensor(
            [int(points.shape[0]) for points in point_sets],
            dtype=torch.long,
            device=point_sets[0].device,
        )
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(dim=0)])
        batch_indices = torch.repeat_interleave(
            torch.arange(len(point_sets), device=lengths.device), lengths
        )
        packed_features: dict[str, Tensor] = {}
        if features is not None:
            if len(features) != len(point_sets):
                raise ValueError("features must have one mapping per point set")
            common = set(features[0])
            for feature_set in features[1:]:
                common.intersection_update(feature_set)
            for name in sorted(common):
                values = [feature_set[name] for feature_set in features]
                if all(int(value.shape[0]) == int(points.shape[0]) for value, points in zip(values, point_sets)):
                    packed_features[name] = torch.cat(values, dim=0)
        return cls(
            points=torch.cat(list(point_sets), dim=0),
            batch_indices=batch_indices,
            offsets=offsets,
            lengths=lengths,
            features=packed_features,
        )


@dataclass(frozen=True)
class DatasetSampleRecord:
    """Lightweight index entry; point payloads remain lazy on disk."""

    sample_id: str
    scene_id: str
    frame_id: int
    fragment_id: int
    fragment_key: str
    object_model_id: str
    visible_points_path: Path
    num_observed_points: int
    gt_fragment: Mapping[str, Any]
    scene_meta: Mapping[str, Any]
    fragment_mesh_metadata: Any


__all__ = ["DatasetSampleRecord", "PackedPointBatch"]
