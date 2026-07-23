"""Content-addressed cache contracts for frozen fine-coordinate inputs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from symm_template_reg.models.losses.joint_surface_correspondence_pose_loss_v3 import (
    coordinate_mean_and_tail_loss,
)
from symm_template_reg.models.pose.pose_representation import invert_transform, transform_points
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance
from symm_template_reg.models.pose.weighted_procrustes import WeightedProcrustes
from symm_template_reg.models.symmetry.groups import parse_rotation_group
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms


FINE_ONLY_PREFIXES = (
    "correspondence_head.fine_feature_adapter",
    "correspondence_head.fine_coordinate_auxiliary_head",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_module_state_sha256(
    model: nn.Module, trainable_prefixes: Sequence[str] = FINE_ONLY_PREFIXES
) -> str:
    prefixes = tuple(map(str, trainable_prefixes))
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def point_order_sha256(points: Tensor, valid_mask: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (points.detach().cpu().contiguous(), valid_mask.detach().cpu().contiguous()):
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cache_eligibility(
    model: nn.Module,
    *,
    trainable_prefixes: Sequence[str],
    augmentations_enabled: bool,
    deterministic_point_sampling: bool,
) -> dict[str, Any]:
    configured = tuple(map(str, trainable_prefixes))
    upstream_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in FINE_ONLY_PREFIXES
        )
    ]
    active_dropout = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Dropout) and module.p > 0
    ]
    checks = {
        "fine_only_trainable_prefixes": set(configured) == set(FINE_ONLY_PREFIXES),
        "upstream_modules_frozen": not upstream_trainable,
        "augmentations_disabled": not bool(augmentations_enabled),
        "dropout_disabled": not active_dropout,
        "point_sampling_deterministic": bool(deterministic_point_sampling),
    }
    return {
        "cache_allowed_by_policy": all(checks.values()),
        "checks": checks,
        "upstream_trainable_parameters": upstream_trainable,
        "active_dropout_modules": active_dropout,
    }


def build_frozen_feature_cache_key(
    *,
    frozen_module_state_sha256_value: str,
    initialization_checkpoint: str | Path,
    manifest: str | Path,
    template_sha256: str,
    sidecar_sha256: str,
    point_selection_policy: Mapping[str, Any] | str,
    model_config: Mapping[str, Any],
    dtype: str,
    tensor_shapes: Mapping[str, Sequence[int]],
    point_order_sha256_value: str,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema": "frozen_feature_cache_v1",
        "frozen_module_state_sha256": str(frozen_module_state_sha256_value),
        "initialization_checkpoint_sha256": _sha256_file(initialization_checkpoint),
        "manifest_file_sha256": _sha256_file(manifest),
        "template_sha256": str(template_sha256),
        "sidecar_sha256": str(sidecar_sha256),
        "point_selection_policy": point_selection_policy,
        "model_config": model_config,
        "dtype": str(dtype),
        "tensor_shapes": {
            str(name): list(map(int, shape)) for name, shape in sorted(tensor_shapes.items())
        },
        "point_order_sha256": str(point_order_sha256_value),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def detach_cache_tensors(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, Mapping):
        return {str(key): detach_cache_tensors(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_cache_tensors(item) for item in value)
    if isinstance(value, list):
        return [detach_cache_tensors(item) for item in value]
    return value


class FrozenFeatureCache:
    """Atomic storage which refuses a content-key mismatch."""

    def __init__(self, root: str | Path, key: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.key = str(key)
        self.path = self.root / f"{self.key}.pt"

    def store(self, payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema": "frozen_feature_cache_v1",
                "key": self.key,
                "metadata": dict(metadata),
                "payload": detach_cache_tensors(payload),
            },
            temporary,
        )
        temporary.replace(self.path)
        return self.path

    def load(self, device: torch.device | str = "cpu") -> dict[str, Any]:
        value = torch.load(self.path, map_location=device, weights_only=False)
        if value.get("schema") != "frozen_feature_cache_v1" or value.get("key") != self.key:
            raise ValueError("frozen feature cache key/schema mismatch")
        return value


def capture_fine_adapter_inputs(model: nn.Module, batch: Mapping[str, Any]):
    """Run online once and capture the exact positional adapter inputs."""

    adapter = model.correspondence_head.fine_feature_adapter
    captured: dict[str, Any] = {}

    def hook(_module, args):
        names = (
            "dense_observed_features",
            "template_conditioned_observed_features",
            "observed_points_C",
            "observed_valid_mask",
            "observed_normals_C",
        )
        captured.update(
            {name: value for name, value in zip(names, args) if value is not None}
        )

    handle = adapter.register_forward_pre_hook(hook)
    try:
        prediction = model(batch)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("fine adapter was not invoked")
    return prediction, captured


def cached_fine_coordinate_forward(
    adapter: nn.Module,
    coordinate_head: nn.Module,
    payload: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], Tensor]:
    output = adapter(
        payload["dense_observed_features"],
        payload["template_conditioned_observed_features"],
        payload["observed_points_C"],
        payload["observed_valid_mask"],
        payload.get("observed_normals_C"),
    )
    coordinates = coordinate_head(output["fine_point_features"])
    return output, coordinates


def _padded(value: Any) -> Tensor:
    return value.to_padded()["points"] if hasattr(value, "to_padded") else value


def _smooth_zero(value: Tensor, scale: float) -> Tensor:
    normalized = value / max(float(scale), 1e-12)
    return torch.nn.functional.smooth_l1_loss(
        normalized, torch.zeros_like(normalized), reduction="none"
    )


def fine_coordinate_active_loss(
    predicted_normalized: Tensor,
    batch: Mapping[str, Any],
    valid_mask: Tensor,
    loss_config: Mapping[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Exact five-term objective used by cached four-view fine-only training."""

    cfg = dict(loss_config["joint_surface_correspondence_pose_v3"])
    target_points = _padded(batch["gt"]["points_O_corresponding"])
    observed_points = _padded(batch["observed"])
    if isinstance(batch["observed"], Mapping):
        observed_points = batch["observed"]["points_C"]
    procrustes = WeightedProcrustes().to(predicted_normalized.device)
    sample_totals = []
    selected_coordinate = []
    selected_tail = []
    selected_rotation = []
    selected_translation = []
    selected_alignment = []
    selected_elements = []
    for index in range(len(predicted_normalized)):
        mask = valid_mask[index]
        vertices = batch["template_mesh_vertices_O"][index].to(predicted_normalized)
        extent = (vertices.amax(0) - vertices.amin(0)).clamp_min(1e-8)
        bbox_min = vertices.amin(0)
        qn = predicted_normalized[index, mask]
        q = 0.5 * (qn + 1.0) * extent + bbox_min
        raw_target = target_points[index]
        metadata = batch["template_symmetry_metadata"][index]
        group = parse_rotation_group(batch["gt"]["effective_symmetry_group"][index])
        symmetries = symmetry_transforms(
            group, metadata.axis.direction, metadata.axis.origin,
            so2_num_samples=36 if group.type == "SO2" else None,
            dtype=q.dtype, device=q.device,
        )
        targets = transform_points(invert_transform(symmetries), raw_target[None])[:, mask]
        target_normalized = 2.0 * (targets - bbox_min) / extent - 1.0
        coordinate, tail = coordinate_mean_and_tail_loss(qn, target_normalized, .10)
        observed = observed_points[index, mask]
        solution = procrustes.solve(
            q[None].float(), observed[None].float(),
            q.new_ones((1, len(q))).float(),
            torch.ones((1, len(q)), dtype=torch.bool, device=q.device),
        )
        pose = solution["transform"][0].to(q)
        equivalent = batch["gt"]["T_C_from_O"][index][None] @ symmetries
        rotation_raw = rotation_geodesic_distance(
            pose[:3, :3][None], equivalent[:, :3, :3]
        )
        translation_raw = torch.linalg.vector_norm(
            pose[:3, 3][None] - equivalent[:, :3, 3], dim=-1
        )
        reconstructed = transform_points(pose[None], q[None])[0]
        alignment_raw = torch.linalg.vector_norm(
            reconstructed - observed, dim=-1
        ).mean().expand(len(symmetries))
        rotation = _smooth_zero(
            rotation_raw, math.radians(float(cfg.get("raw_pose_rotation_scale_deg", 1.0)))
        )
        translation = _smooth_zero(
            translation_raw, float(cfg.get("raw_pose_translation_scale_m", .001))
        )
        alignment = _smooth_zero(
            alignment_raw, float(cfg.get("raw_alignment_scale_m", .001))
        )
        totals = (
            float(cfg.get("fine_coordinate_aux_weight", 0.0)) * coordinate
            + float(cfg.get("fine_coordinate_tail_weight", 0.0)) * tail
            + float(cfg.get("raw_pose_rotation_weight", 0.0)) * rotation
            + float(cfg.get("raw_pose_translation_weight", 0.0)) * translation
            + float(cfg.get("raw_alignment_weight", 0.0)) * alignment
        )
        selected = int(totals.detach().argmin())
        sample_totals.append(totals[selected])
        selected_coordinate.append(coordinate[selected])
        selected_tail.append(tail[selected])
        selected_rotation.append(rotation[selected])
        selected_translation.append(translation[selected])
        selected_alignment.append(alignment[selected])
        selected_elements.append(selected)
    mean = lambda rows: torch.stack(rows).mean()
    losses = {
        "loss_total": mean(sample_totals),
        "loss_fine_coordinate_aux_normalized": mean(selected_coordinate),
        "loss_fine_coordinate_aux_tail_normalized": mean(selected_tail),
        "loss_raw_aux_rotation_normalized": mean(selected_rotation),
        "loss_raw_aux_translation_normalized": mean(selected_translation),
        "loss_raw_aux_alignment_normalized": mean(selected_alignment),
        "selected_shared_symmetry_element_mean": predicted_normalized.new_tensor(
            selected_elements, dtype=predicted_normalized.dtype
        ).mean(),
    }
    return losses["loss_total"], losses


__all__ = [
    "FINE_ONLY_PREFIXES",
    "FrozenFeatureCache",
    "build_frozen_feature_cache_key",
    "cache_eligibility",
    "cached_fine_coordinate_forward",
    "capture_fine_adapter_inputs",
    "detach_cache_tensors",
    "frozen_module_state_sha256",
    "fine_coordinate_active_loss",
    "point_order_sha256",
]
