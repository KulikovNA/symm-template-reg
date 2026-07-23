"""Packed-first and padded collators for registration samples."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .structures import PackedPointBatch

try:
    from symm_template_reg.registry import COLLATE_FUNCTIONS
except (ImportError, AttributeError):
    COLLATE_FUNCTIONS = None


def _register(name: str):
    def decorator(function):
        if COLLATE_FUNCTIONS is not None and COLLATE_FUNCTIONS.get(name) is None:
            COLLATE_FUNCTIONS.register_module(function, name=name)
        return function
    return decorator


def _common_aligned_features(
    point_sets: Sequence[Tensor], payloads: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Tensor]]:
    """Preserve partially available fields and expose per-point availability masks."""

    result: list[dict[str, Tensor]] = []
    specifications: dict[str, Tensor] = {}
    for key in keys:
        for points, payload in zip(point_sets, payloads):
            candidate = payload.get(key)
            if isinstance(candidate, Tensor) and (
                candidate.ndim == 0 or len(candidate) != len(points)
            ):
                raise ValueError(f"feature {key!r} is not aligned with its point set")
        specifications[key] = next(
            (
                value
                for payload in payloads
                if isinstance((value := payload.get(key)), Tensor)
            ),
            None,
        )
    specifications = {key: value for key, value in specifications.items() if value is not None}
    complete = {
        key: all(isinstance(payload.get(key), Tensor) for payload in payloads)
        for key in specifications
    }
    for points, payload in zip(point_sets, payloads):
        feature_set: dict[str, Tensor] = {}
        for key, specification in specifications.items():
            value = payload.get(key)
            if isinstance(value, Tensor) and value.ndim >= 1 and len(value) == len(points):
                feature_set[key] = value
                available = True
            else:
                feature_set[key] = torch.zeros(
                    (len(points), *specification.shape[1:]),
                    dtype=specification.dtype,
                    device=points.device,
                )
                available = False
            if not complete[key] and key != "valid_mask":
                feature_set[f"{key}_valid_mask"] = torch.full(
                    (len(points),), available, dtype=torch.bool, device=points.device
                )
        result.append(feature_set)
    return result


def _ids(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [str(sample["sample_id"]) for sample in samples],
        "scene_id": [str(sample["scene_id"]) for sample in samples],
        "frame_id": torch.tensor([int(sample["frame_id"]) for sample in samples], dtype=torch.long),
        "fragment_id": torch.tensor([int(sample["fragment_id"]) for sample in samples], dtype=torch.long),
        "object_model_id": [str(sample["object_model_id"]) for sample in samples],
    }


def _stack_active_regions(values: Sequence[Any]) -> tuple[Tensor | None, Tensor | None]:
    if not values or not any(isinstance(value, Tensor) for value in values):
        return None, None
    width = max(int(value.numel()) for value in values if isinstance(value, Tensor))
    device = next(value.device for value in values if isinstance(value, Tensor))
    output = torch.zeros((len(values), width), dtype=torch.bool, device=device)
    valid = torch.zeros_like(output)
    for index, value in enumerate(values):
        if isinstance(value, Tensor):
            output[index, : value.numel()] = value.to(dtype=torch.bool)
            valid[index, : value.numel()] = True
    return output, valid


def _stack_point_region_indices(
    values: Sequence[Any],
) -> tuple[Tensor | None, Tensor | None]:
    """Pad row-aligned point-region targets with the ignore index ``-1``."""

    if not values or not any(isinstance(value, Tensor) for value in values):
        return None, None
    width = max(int(value.numel()) for value in values if isinstance(value, Tensor))
    device = next(value.device for value in values if isinstance(value, Tensor))
    output = torch.full((len(values), width), -1, dtype=torch.long, device=device)
    valid = torch.zeros((len(values), width), dtype=torch.bool, device=device)
    for index, value in enumerate(values):
        if isinstance(value, Tensor):
            target = value.to(device=device, dtype=torch.long).reshape(-1)
            output[index, : target.numel()] = target
            valid[index, : target.numel()] = target.ge(0)
    return output, valid


def _gt_common(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gt_items = [sample["gt"] for sample in samples]
    transforms = torch.stack([item["T_C_from_O"] for item in gt_items])
    active, active_valid = _stack_active_regions(
        [item.get("active_symmetry_regions") for item in gt_items]
    )
    point_regions, point_regions_valid = _stack_point_region_indices(
        [item.get("point_symmetry_region_indices") for item in gt_items]
    )
    world_from_camera = [item.get("T_W_from_C") for item in gt_items]
    return {
        "T_C_from_O": transforms,
        "T_W_from_C": (
            torch.stack(world_from_camera)
            if all(isinstance(value, Tensor) for value in world_from_camera)
            else None
        ),
        "pose_parameters_normalized": torch.stack(
            [item["pose_parameters_normalized"] for item in gt_items]
        ),
        "observed_centroid_C": torch.stack(
            [item["observed_centroid_C"] for item in gt_items]
        ),
        "observed_scale": torch.stack([item["observed_scale"] for item in gt_items]),
        "active_symmetry_regions": active,
        "active_symmetry_regions_valid_mask": active_valid,
        "point_symmetry_region_indices": point_regions,
        "point_symmetry_region_valid_mask": point_regions_valid,
        "symmetry_supervision_mask": torch.tensor(
            [isinstance(item.get("active_symmetry_regions"), Tensor) for item in gt_items],
            dtype=torch.bool,
            device=transforms.device,
        ),
        "effective_symmetry_group": [
            item.get("effective_symmetry_group") for item in gt_items
        ],
        "equivalent_T_C_from_O": [
            item.get("equivalent_T_C_from_O") for item in gt_items
        ],
        "symmetry_training_target_type": [
            item.get("symmetry_training_target_type") for item in gt_items
        ],
        "symmetry_pose_set_exhaustive": [
            item.get("symmetry_pose_set_exhaustive") for item in gt_items
        ],
        "symmetry_region_point_counts": [
            item.get("symmetry_region_point_counts") for item in gt_items
        ],
    }


@_register("packed")
@_register("packed_collate")
def packed_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate point sets as ``[sum(N_i), C]`` plus offsets and lengths."""

    if not samples:
        raise ValueError("cannot collate an empty sample sequence")
    observed_payloads = [sample["observed"] for sample in samples]
    observed_points = [payload["points_C"] for payload in observed_payloads]
    observed = PackedPointBatch.from_list(
        observed_points,
        features=_common_aligned_features(
            observed_points,
            observed_payloads,
            ("normals_C", "surface_labels", "valid_mask"),
        ),
    )
    template_payloads = [sample["template"] for sample in samples]
    template_points = [payload.get("fine_points_O", payload["points_O"]) for payload in template_payloads]
    template_feature_payloads = [
        {
            "normals_O": payload.get("fine_normals_O", payload.get("normals_O"))
        }
        for payload in template_payloads
    ]
    template = PackedPointBatch.from_list(
        template_points,
        features=_common_aligned_features(
            template_points, template_feature_payloads, ("normals_O",)
        ),
    )
    gt_payloads = [sample["gt"] for sample in samples]
    correspondence_points = [payload.get("points_O_corresponding") for payload in gt_payloads]
    gt = _gt_common(samples)
    if any(isinstance(points, Tensor) for points in correspondence_points):
        normalized_correspondences = [
            points
            if isinstance(points, Tensor)
            else torch.zeros_like(observed_points[index])
            for index, points in enumerate(correspondence_points)
        ]
        correspondence_valid = [
            torch.full(
                (len(observed_points[index]),),
                isinstance(points, Tensor),
                dtype=torch.bool,
                device=observed_points[index].device,
            )
            for index, points in enumerate(correspondence_points)
        ]
        gt["points_O_corresponding"] = PackedPointBatch.from_list(
            normalized_correspondences,
            features=[{"valid_mask": value} for value in correspondence_valid],
        )
        gt["points_O_corresponding_valid_mask"] = torch.cat(correspondence_valid)
    else:
        gt["points_O_corresponding"] = None
        gt["points_O_corresponding_valid_mask"] = None
    overlap_values = [payload.get("overlap_labels") for payload in gt_payloads]
    if any(isinstance(value, Tensor) for value in overlap_values):
        normalized_overlap = [
            value
            if isinstance(value, Tensor)
            else torch.zeros(
                len(observed_points[index]),
                dtype=torch.bool,
                device=observed_points[index].device,
            )
            for index, value in enumerate(overlap_values)
        ]
        overlap_valid = [
            torch.full(
                (len(observed_points[index]),),
                isinstance(value, Tensor),
                dtype=torch.bool,
                device=observed_points[index].device,
            )
            for index, value in enumerate(overlap_values)
        ]
        gt["overlap_labels"] = torch.cat(normalized_overlap)
        gt["overlap_labels_valid_mask"] = torch.cat(overlap_valid)
    else:
        gt["overlap_labels"] = None
        gt["overlap_labels_valid_mask"] = None
    template_faces = []
    template_meshes = []
    for payload, selected_points in zip(template_payloads, template_points):
        full_points = payload["points_O"]
        faces = payload.get("faces")
        fine_indices = payload.get("fine_indices")
        topology_compatible = (
            len(selected_points) == len(full_points)
            and isinstance(fine_indices, Tensor)
            and torch.equal(
                fine_indices,
                torch.arange(
                    len(full_points),
                    dtype=fine_indices.dtype,
                    device=fine_indices.device,
                ),
            )
            and torch.equal(selected_points, full_points)
        )
        template_faces.append(faces if topology_compatible else None)
        template_meshes.append({"points_O": full_points, "faces": faces})
    return {
        **_ids(samples),
        "collate_mode": "packed",
        "observed": observed,
        "template": template,
        "template_mesh_vertices_O": [payload["points_O"] for payload in template_payloads],
        "template_mesh_faces": [payload.get("faces") for payload in template_payloads],
        # Faces are only paired with packed points when no template subsampling occurred.
        "template_faces": template_faces,
        "template_meshes": template_meshes,
        "template_coarse_points_O": [payload.get("coarse_points_O") for payload in template_payloads],
        "template_metadata": [payload.get("metadata") for payload in template_payloads],
        "template_symmetry_metadata": [
            payload.get("symmetry_metadata") for payload in template_payloads
        ],
        # Stable short alias for consumers which do not need other template metadata.
        "symmetry_metadata": [
            payload.get("symmetry_metadata") for payload in template_payloads
        ],
        "gt": gt,
        "meta": [dict(sample["meta"]) for sample in samples],
    }


def _named_padded(
    packed: PackedPointBatch,
    point_name: str,
    feature_names: Sequence[str],
    *,
    feature_pad_values: Mapping[str, float | int | bool] | None = None,
) -> dict[str, Any]:
    dense = packed.to_padded(feature_pad_values=feature_pad_values)
    result: dict[str, Any] = {
        point_name: dense["points"],
        "points": dense["points"],
        "valid_mask": dense["valid_mask"],
        "lengths": dense["valid_mask"].sum(dim=1, dtype=torch.long),
    }
    for name in feature_names:
        result[name] = dense["features"].get(name)
    return result


@_register("padded")
@_register("padded_collate")
def padded_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate to dense ``[B, Nmax, C]`` tensors with explicit masks."""

    packed = packed_collate(samples)
    observed = _named_padded(
        packed["observed"],
        "points_C",
        (
            "normals_C",
            "normals_C_valid_mask",
            "surface_labels",
            "surface_labels_valid_mask",
        ),
        feature_pad_values={"surface_labels": 255, "valid_mask": False},
    )
    template = _named_padded(
        packed["template"], "points_O", ("normals_O", "normals_O_valid_mask")
    )
    gt = dict(packed["gt"])
    corresponding = gt["points_O_corresponding"]
    if corresponding is not None:
        correspondence_dense = corresponding.to_padded()
        gt["points_O_corresponding"] = correspondence_dense["points"]
        gt["points_O_corresponding_valid_mask"] = correspondence_dense["valid_mask"]
    else:
        gt["points_O_corresponding"] = None
        gt["points_O_corresponding_valid_mask"] = None
    overlap = gt["overlap_labels"]
    if overlap is not None:
        lengths = packed["observed"].lengths
        max_length = observed["points_C"].shape[1]
        padded_overlap = torch.zeros((len(lengths), max_length), dtype=torch.bool)
        start = 0
        for batch_id, length in enumerate(lengths.tolist()):
            padded_overlap[batch_id, :length] = overlap[start : start + length]
            start += length
        gt["overlap_labels"] = padded_overlap
        overlap_valid = gt.get("overlap_labels_valid_mask")
        if overlap_valid is not None:
            padded_overlap_valid = torch.zeros_like(padded_overlap)
            start = 0
            for batch_id, length in enumerate(lengths.tolist()):
                padded_overlap_valid[batch_id, :length] = overlap_valid[start : start + length]
                start += length
            gt["overlap_labels_valid_mask"] = padded_overlap_valid
    return {
        key: value
        for key, value in {
            **packed,
            "collate_mode": "padded",
            "observed": observed,
            "template": template,
            "gt": gt,
        }.items()
    }


def fragment_template_collate(
    samples: Sequence[Mapping[str, Any]], *, mode: str = "packed"
) -> dict[str, Any]:
    if mode == "packed":
        return packed_collate(samples)
    if mode == "padded":
        return padded_collate(samples)
    raise ValueError("collate mode must be 'packed' or 'padded'")


def build_collate_fn(mode: str = "packed"):
    return partial(fragment_template_collate, mode=mode)


@_register("FragmentTemplateCollator")
class FragmentTemplateCollator:
    """Config-buildable callable while retaining direct function aliases."""

    def __init__(self, mode: str = "packed") -> None:
        if mode not in {"packed", "padded"}:
            raise ValueError("collate mode must be 'packed' or 'padded'")
        self.mode = mode

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return fragment_template_collate(samples, mode=self.mode)


__all__ = [
    "build_collate_fn",
    "FragmentTemplateCollator",
    "fragment_template_collate",
    "packed_collate",
    "padded_collate",
]
