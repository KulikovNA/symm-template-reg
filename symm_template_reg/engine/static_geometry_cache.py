"""Exact cache for coordinate-only geometry used by fixed overfit datasets.

The cache deliberately contains only integer neighbourhood/selection indices,
masks, lengths and coordinate-derived diagnostic distances.  Learned features
and outputs are never admitted.  Runtime reuse is only enabled for a fixed,
non-augmented batch whose identity is established by a content SHA256 key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from symm_template_reg.models.geometry.point_ops import (
    batched_gather,
    farthest_point_indices,
    knn_indices,
)


CACHE_SCHEMA_VERSION = "static-geometry-v1"


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def static_geometry_cache_key(
    *,
    manifest_sha256: str,
    observed_points: Tensor,
    observed_mask: Tensor,
    template_points: Tensor,
    template_mask: Tensor,
    template_mesh_sha256: str,
    geometry_config: Mapping[str, Any],
    point_selection_policy: str,
) -> str:
    """Return a content key covering every structure-defining input."""

    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "manifest_sha256": str(manifest_sha256),
        "observed_coordinates_and_order_sha256": _tensor_sha256(observed_points),
        "observed_valid_mask_sha256": _tensor_sha256(observed_mask),
        "template_coordinates_and_order_sha256": _tensor_sha256(template_points),
        "template_valid_mask_sha256": _tensor_sha256(template_mask),
        "template_mesh_sha256": str(template_mesh_sha256),
        "geometry_config": dict(geometry_config),
        "point_selection_policy": str(point_selection_policy),
        "dtype": str(observed_points.dtype),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_static_cache_configuration(
    *, enabled: bool, augmentations: Mapping[str, Any] | None,
    frozen_feature_cache: Mapping[str, Any] | None = None,
) -> None:
    """Reject unsafe cache combinations before training starts."""

    if not enabled:
        return
    if augmentations and bool(augmentations.get("enabled", False)):
        raise ValueError("static_geometry_cache requires augmentations.enabled=false")
    if frozen_feature_cache and bool(frozen_feature_cache.get("enabled", False)):
        raise ValueError(
            "scratch optimization forbids caching learned/frozen encoder features"
        )


def _neighbor_distances(points: Tensor, indices: Tensor) -> Tensor:
    neighbors = batched_gather(points, indices[..., 1:])
    return torch.linalg.vector_norm(neighbors - points.unsqueeze(-2), dim=-1)


@torch.no_grad()
def build_static_geometry(
    observed_points: Tensor,
    observed_mask: Tensor,
    template_points: Tensor,
    template_mask: Tensor,
    *,
    observed_encoder_neighbors: int = 12,
    template_encoder_neighbors: int = 12,
    observed_tokens: int = 256,
    template_tokens: int = 512,
    geometry_neighbors: int = 8,
    fine_neighbors: int = 32,
) -> dict[str, Tensor]:
    """Build exact online structures, using the same public primitives as modules."""

    observed_encoder_knn = knn_indices(
        observed_points, observed_points, observed_mask,
        observed_encoder_neighbors + 1,
    )
    template_encoder_knn = knn_indices(
        template_points, template_points, template_mask,
        template_encoder_neighbors + 1,
    )
    observed_fps, observed_token_mask = farthest_point_indices(
        observed_points, observed_mask, observed_tokens
    )
    template_fps, template_token_mask = farthest_point_indices(
        template_points, template_mask, template_tokens
    )
    observed_token_points = batched_gather(observed_points, observed_fps)
    template_token_points = batched_gather(template_points, template_fps)
    observed_geometry_knn = knn_indices(
        observed_token_points, observed_token_points, observed_token_mask,
        geometry_neighbors + 1,
    )
    template_geometry_knn = knn_indices(
        template_token_points, template_token_points, template_token_mask,
        geometry_neighbors + 1,
    )
    dense_to_observed_token = knn_indices(
        observed_points, observed_token_points, observed_token_mask, 1
    ).squeeze(-1)
    fine_knn = knn_indices(
        observed_points, observed_points, observed_mask, fine_neighbors + 1
    )
    return {
        "observed_encoder_knn": observed_encoder_knn,
        "template_encoder_knn": template_encoder_knn,
        "observed_fps_indices": observed_fps,
        "observed_token_mask": observed_token_mask,
        "template_fps_indices": template_fps,
        "template_token_mask": template_token_mask,
        "dense_to_observed_token_indices": dense_to_observed_token,
        "observed_geometry_knn": observed_geometry_knn,
        "template_geometry_knn": template_geometry_knn,
        "fine_adapter_knn": fine_knn,
        "observed_encoder_neighbor_distances": _neighbor_distances(
            observed_points, observed_encoder_knn
        ),
        "template_encoder_neighbor_distances": _neighbor_distances(
            template_points, template_encoder_knn
        ),
        "observed_valid_mask": observed_mask,
        "template_valid_mask": template_mask,
        "observed_lengths": observed_mask.sum(-1),
        "template_lengths": template_mask.sum(-1),
    }


@dataclass
class StaticGeometryCache:
    """Single fixed-batch in-memory cache used by an overfit model instance."""

    config: Mapping[str, Any]
    manifest_sha256: str = "runtime-fixed-batch"
    template_mesh_sha256: str = "runtime-template"
    point_selection_policy: str = "shell_only"
    entries: dict[str, dict[str, Tensor]] | None = None
    keys: dict[str, str] | None = None
    hits: int = 0
    misses: int = 0

    def get_or_build(
        self,
        observed_points: Tensor,
        observed_mask: Tensor,
        template_points: Tensor,
        template_mask: Tensor,
        batch_identity: str = "fixed-batch",
    ) -> dict[str, Tensor]:
        if self.entries is None:
            self.entries = {}
            self.keys = {}
        identity = str(batch_identity)
        if identity in self.entries:
            entry = self.entries[identity]
            expected = (
                entry["observed_valid_mask"].shape,
                entry["template_valid_mask"].shape,
                entry["observed_valid_mask"].device,
            )
            actual = (observed_mask.shape, template_mask.shape, observed_mask.device)
            if actual != expected:
                raise ValueError(
                    "fixed-batch static geometry cache received a different batch shape/device"
                )
            self.hits += 1
            return entry
        assert self.keys is not None
        self.keys[identity] = static_geometry_cache_key(
            manifest_sha256=self.manifest_sha256,
            observed_points=observed_points,
            observed_mask=observed_mask,
            template_points=template_points,
            template_mask=template_mask,
            template_mesh_sha256=self.template_mesh_sha256,
            geometry_config=self.config,
            point_selection_policy=self.point_selection_policy,
        )
        kwargs = {
            name: int(self.config[name])
            for name in (
                "observed_encoder_neighbors", "template_encoder_neighbors",
                "observed_tokens", "template_tokens", "geometry_neighbors",
                "fine_neighbors",
            ) if name in self.config
        }
        entry = build_static_geometry(
            observed_points, observed_mask, template_points, template_mask, **kwargs
        )
        self.entries[identity] = entry
        self.misses += 1
        return entry

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_key_sha256_by_batch": dict(self.keys or {}),
            "hits": self.hits,
            "misses": self.misses,
            "contains_learned_features": False,
            "cached_fields": sorted(next(iter(self.entries.values()))) if self.entries else [],
        }


def save_static_geometry_cache(
    output_dir: str | Path, entry: Mapping[str, Tensor], metadata: Mapping[str, Any]
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    tensor_path = output / "static_geometry_cache.pt"
    metadata_path = output / "static_geometry_cache_manifest.json"
    torch.save({name: value.detach().cpu() for name, value in entry.items()}, tensor_path)
    metadata_path.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n")
    return tensor_path, metadata_path


__all__ = [
    "CACHE_SCHEMA_VERSION", "StaticGeometryCache", "build_static_geometry",
    "save_static_geometry_cache", "static_geometry_cache_key",
    "validate_static_cache_configuration",
]
