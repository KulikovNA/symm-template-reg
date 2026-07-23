"""Contracts and diagnostics for controlled single-fragment overfit stages."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from symm_template_reg.models.pose.metrics import symmetry_aware_pose_errors


def canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    value = dict(payload)
    value.pop("manifest_sha256", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_content_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(payload)).hexdigest()


@dataclass
class TrainingCounters:
    """Independent counters; ``optimizer_step`` is the legacy global-step alias."""

    batch_step: int = 0
    optimizer_step: int = 0
    samples_seen: int = 0

    def record_batch(self, batch_size: int) -> None:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        self.batch_step += 1
        self.samples_seen += int(batch_size)

    def record_optimizer_step(self) -> None:
        self.optimizer_step += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "batch_step": self.batch_step,
            "optimizer_step": self.optimizer_step,
            "samples_seen": self.samples_seen,
            "global_step": self.optimizer_step,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> "TrainingCounters":
        optimizer_step = int(
            payload.get("optimizer_step", payload.get("global_step", 0))
        )
        return cls(
            batch_step=int(payload.get("batch_step", optimizer_step)),
            optimizer_step=optimizer_step,
            samples_seen=int(payload.get("samples_seen", 0)),
        )


def apply_trainable_prefixes(
    model: torch.nn.Module,
    trainable_prefixes: Sequence[str] | None,
) -> dict[str, Any]:
    """Freeze every parameter except explicit prefixes; ``None`` means all."""

    prefixes = None if trainable_prefixes is None else tuple(map(str, trainable_prefixes))
    if prefixes is not None and (not prefixes or len(prefixes) != len(set(prefixes))):
        raise ValueError("stage.trainable_module_prefixes must be unique and non-empty")
    matched = {prefix: 0 for prefix in prefixes or ()}
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = prefixes is None or any(
            name == prefix or name.startswith(prefix + ".") for prefix in prefixes
        )
        parameter.requires_grad_(trainable)
        (trainable_names if trainable else frozen_names).append(name)
        for prefix in prefixes or ():
            if name == prefix or name.startswith(prefix + "."):
                matched[prefix] += parameter.numel()
    missing = [prefix for prefix, count in matched.items() if count == 0]
    if missing:
        raise ValueError(f"trainable module prefixes matched no parameters: {missing}")
    if not trainable_names:
        raise ValueError("stage freezing left no trainable parameters")
    return {
        "trainable_module_prefixes": list(prefixes) if prefixes is not None else ["<all>"],
        "frozen_module_prefixes": ["<all except explicit trainable prefixes>"]
        if prefixes is not None
        else [],
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


def build_selective_optimizer_parameter_groups(
    model: torch.nn.Module,
    *,
    default_lr: float,
    prefix_learning_rates: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Build disjoint AdamW groups for selective fine-stage unfreezing."""

    rules = tuple((str(prefix), float(lr)) for prefix, lr in (prefix_learning_rates or {}).items())
    matched = {prefix: 0 for prefix, _ in rules}
    grouped: dict[tuple[str, float], list[torch.nn.Parameter]] = {}
    names: dict[tuple[str, float], list[str]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [(prefix, lr) for prefix, lr in rules if name == prefix or name.startswith(prefix + ".")]
        if len(matches) > 1:
            raise ValueError(f"optimizer LR prefixes overlap for {name}: {matches}")
        label, lr = matches[0] if matches else ("<default>", float(default_lr))
        if matches:
            matched[label] += parameter.numel()
        key = (label, lr)
        grouped.setdefault(key, []).append(parameter)
        names.setdefault(key, []).append(name)
    missing = [prefix for prefix, count in matched.items() if count == 0]
    if missing:
        raise ValueError(f"optimizer LR prefixes matched no trainable parameters: {missing}")
    if not grouped:
        raise ValueError("optimizer has no trainable parameter groups")
    return [
        {"params": parameters, "lr": lr, "group_name": label, "parameter_names": names[(label, lr)]}
        for (label, lr), parameters in grouped.items()
    ]


def load_model_initialization(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    module_prefixes: Sequence[str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load model-only weights, optionally restricted to strict module prefixes."""

    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint has no model state mapping: {path}")
    model_state = model.state_dict()
    prefixes = tuple(map(str, module_prefixes or ()))
    if prefixes:
        selected = {
            key: value
            for key, value in state.items()
            if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)
        }
        expected = {
            key for key in model_state if any(key == p or key.startswith(p + ".") for p in prefixes)
        }
        if not selected:
            raise ValueError(f"init module prefixes matched no checkpoint keys: {prefixes}")
        missing_selected = sorted(expected - set(selected))
        unexpected_selected = sorted(set(selected) - set(model_state))
        incompatible = sorted(
            key
            for key in expected & set(selected)
            if tuple(model_state[key].shape) != tuple(selected[key].shape)
        )
        if strict and (missing_selected or unexpected_selected or incompatible):
            raise RuntimeError(
                "strict module initialization failed: "
                f"missing={missing_selected[:10]}, unexpected={unexpected_selected[:10]}, "
                f"shape_mismatch={incompatible[:10]}"
            )
        compatible = {
            key: value
            for key, value in selected.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        result = model.load_state_dict(compatible, strict=False)
        loaded_prefixes = list(prefixes)
    else:
        result = model.load_state_dict(state, strict=strict)
        compatible = dict(state)
        loaded_prefixes = ["<all>"]
    return {
        "mode": "init_checkpoint_model_only",
        "checkpoint_path": str(path),
        "strict": bool(strict),
        "loaded_module_prefixes": loaded_prefixes,
        "loaded_key_count": len(compatible),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "optimizer_loaded": False,
        "scheduler_loaded": False,
        "scaler_loaded": False,
        "counters_loaded": False,
    }


def validate_single_fragment_manifest_payload(
    payload: Mapping[str, Any],
    *,
    expected_samples: int = 10,
    min_num_faces: int = 840,
) -> dict[str, Any]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != int(expected_samples):
        raise ValueError(
            f"single-fragment manifest must contain exactly {expected_samples} samples"
        )
    sample_ids = [str(sample.get("sample_id")) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("single-fragment manifest contains duplicate sample IDs")
    scenes = {str(sample.get("scene_id")) for sample in samples}
    fragments = {int(sample.get("fragment_id", -1)) for sample in samples}
    frames = {int(sample.get("frame_id", -1)) for sample in samples}
    meshes = {str(sample.get("fragment_mesh_sha256")) for sample in samples}
    if len(scenes) != 1:
        raise ValueError("single-fragment manifest samples must belong to one scene")
    if len(fragments) != 1:
        raise ValueError("single-fragment manifest samples must belong to one fragment_id")
    if len(frames) != len(samples):
        raise ValueError("single-fragment manifest must contain different frame IDs")
    if len(meshes) != 1 or "None" in meshes:
        raise ValueError("single-fragment manifest must reference one physical mesh SHA256")
    if any(int(sample.get("fragment_num_faces", -1)) < min_num_faces for sample in samples):
        raise ValueError(f"single fragment does not satisfy min_num_faces={min_num_faces}")
    if any(not bool(sample.get("T_W_from_C_available")) for sample in samples):
        raise ValueError("single-fragment manifest requires T_W_from_C for every sample")
    if payload.get("train_sample_ids") != sample_ids:
        raise ValueError("train_sample_ids differ from manifest samples")
    if payload.get("validation_sample_ids") != sample_ids:
        raise ValueError("validation must use the same single-fragment samples")
    return {
        "scene_id": next(iter(scenes)),
        "fragment_id": next(iter(fragments)),
        "fragment_mesh_sha256": next(iter(meshes)),
        "sample_count": len(samples),
        "frame_ids": sorted(frames),
    }


def validate_single_fragment_config(config: Mapping[str, Any]) -> None:
    data = config.get("data", {})
    train = config.get("train", {})
    if not bool(data.get("single_fragment_contract", False)):
        return
    manifest = str(data.get("train_manifest", ""))
    if not manifest or "<" in manifest or "REQUIRED" in manifest:
        raise ValueError(
            "single-fragment config requires data.train_manifest=<built manifest path>"
        )
    if str(data.get("validation_manifest")) != "same_as_train":
        raise ValueError("single-fragment train and validation must use one manifest")
    if bool(config.get("augmentation", {}).get("enabled", False)):
        raise ValueError("single-fragment overfit requires augmentation.enabled=False")
    scratch = str(config.get("initialization_mode", "")) == "scratch"
    allowed_schedulers = {"constant", "linear_warmup_constant"} if scratch else {"constant"}
    if str(train.get("scheduler", {}).get("type")) not in allowed_schedulers:
        raise ValueError(
            f"single-fragment overfit requires scheduler.type in {sorted(allowed_schedulers)}"
        )
    accumulation = int(train.get("gradient_accumulation_steps", -1))
    if scratch:
        effective = int(data.get("train_batch_size", -1)) * accumulation
        if effective != int(data.get("effective_views_per_optimizer_step", 10)):
            raise ValueError("scratch batch/accumulation must cover every view once")
    elif accumulation != 1:
        raise ValueError("single-fragment overfit requires gradient_accumulation_steps=1")
    budget_mode = str(config.get("train_budget", {}).get("mode", "optimizer_steps"))
    if budget_mode == "optimizer_steps" and int(train.get("max_optimizer_steps", 0)) < 1:
        raise ValueError("single-fragment overfit requires positive max_optimizer_steps")
    stage_name = str(config.get("stage", {}).get("name", ""))
    if stage_name.startswith("view_ladder_"):
        if int(data.get("train_batch_size", -1)) != 1 or int(
            data.get("validation_batch_size", -1)
        ) != 1:
            raise ValueError("view-ladder diagnostics require batch_size=1")
        if bool(train.get("amp", True)):
            raise ValueError("view-ladder diagnostics require train.amp=False")
        if int(train.get("early_stopping_patience_evals", -1)) != 0:
            raise ValueError("view-ladder diagnostics require early stopping disabled")
        if int(config.get("dataset", {}).get("random_seed", -1)) != 0:
            raise ValueError("view-ladder point selection requires dataset.random_seed=0")


def inverse_sqrt_frequency_weights(
    frequencies: Tensor | Sequence[int], *, max_class_weight: float = 5.0
) -> Tensor:
    counts = torch.as_tensor(frequencies, dtype=torch.float64)
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional sequence")
    present = counts > 0
    if float(max_class_weight) < 1.0:
        raise ValueError("max_class_weight must be at least 1")
    weights = torch.zeros_like(counts)
    if bool(present.any()):
        base = counts[present].rsqrt()
        low, high = 0.0, 1.0
        while float((base * high).clamp_max(float(max_class_weight)).mean()) < 1.0:
            high *= 2.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            mean = float(
                (base * middle).clamp_max(float(max_class_weight)).mean()
            )
            if mean < 1.0:
                low = middle
            else:
                high = middle
        weights[present] = (base * high).clamp_max(float(max_class_weight))
    return weights.to(torch.float32)


def region_class_distribution(
    samples: Iterable[Mapping[str, Any]], *, max_class_weight: float = 5.0
) -> dict[str, Any]:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("cannot audit region distribution of an empty sample set")
    metadata = sample_list[0]["template"]["symmetry_metadata"]
    region_ids = list(metadata.region_ids)
    point_counts = [0 for _ in region_ids]
    active_positive = [0 for _ in region_ids]
    active_valid = [0 for _ in region_ids]
    per_sample: list[dict[str, Any]] = []
    for sample in sample_list:
        indices = sample["gt"].get("point_symmetry_region_indices")
        active = sample["gt"].get("active_symmetry_regions")
        counts = [0 for _ in region_ids]
        if isinstance(indices, Tensor):
            for index in range(len(region_ids)):
                counts[index] = int(indices.eq(index).sum())
                point_counts[index] += counts[index]
        active_values: list[bool] = []
        for index in range(len(region_ids)):
            valid = isinstance(active, Tensor) and index < active.numel()
            value = bool(active[index]) if valid else False
            active_values.append(value)
            active_valid[index] += int(valid)
            active_positive[index] += int(valid and value)
        per_sample.append(
            {
                "sample_id": sample["sample_id"],
                "frame_id": int(sample["frame_id"]),
                "point_counts": dict(zip(region_ids, counts)),
                "active_regions": dict(zip(region_ids, active_values)),
            }
        )
    weights = inverse_sqrt_frequency_weights(
        point_counts, max_class_weight=float(max_class_weight)
    )
    pos_weight = [
        max((valid - positive) / max(positive, 1), 1.0)
        for positive, valid in zip(active_positive, active_valid)
    ]
    return {
        "region_ids": region_ids,
        "num_samples": len(sample_list),
        "max_class_weight": float(max_class_weight),
        "point_frequency": dict(zip(region_ids, point_counts)),
        "inverse_sqrt_frequency_weights": dict(zip(region_ids, weights.tolist())),
        "active_positive_samples": dict(zip(region_ids, active_positive)),
        "active_valid_samples": dict(zip(region_ids, active_valid)),
        "active_pos_weight": dict(zip(region_ids, pos_weight)),
        "absent_point_classes": [
            region for region, count in zip(region_ids, point_counts) if count == 0
        ],
        "absent_active_positive_classes": [
            region for region, count in zip(region_ids, active_positive) if count == 0
        ],
        "absent_active_negative_classes": [
            region
            for region, positive, valid in zip(
                region_ids, active_positive, active_valid
            )
            if positive == valid
        ],
        "samples": per_sample,
    }


def _axis_spread_deg(transforms: Tensor, axis_O: Tensor) -> float:
    axes = transforms[:, :3, :3] @ axis_O
    axes = torch.nn.functional.normalize(axes, dim=-1)
    dots = torch.abs(axes @ axes.transpose(0, 1)).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(dots)).max())


def world_pose_consistency(
    transforms: Tensor,
    symmetry_metadata: Any,
    effective_group: Any,
) -> dict[str, float]:
    """Cross-view spread for poses of one static object expressed in world."""

    poses = torch.as_tensor(transforms)
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4) or len(poses) < 1:
        raise ValueError("world transforms must have shape [V,4,4]")
    centers = poses[:, :3, 3]
    centered = centers - centers.mean(dim=0)
    std_mm = torch.sqrt(centered.square().sum(dim=-1).mean()) * 1000.0
    range_mm = torch.cdist(centers, centers).max() * 1000.0
    axis = torch.as_tensor(
        symmetry_metadata.axis.direction, dtype=poses.dtype, device=poses.device
    )
    pairwise_rotation: list[Tensor] = []
    for index in range(len(poses)):
        errors = symmetry_aware_pose_errors(
            poses,
            poses[index],
            symmetry_metadata,
            effective_group=effective_group,
        )
        pairwise_rotation.append(errors["rotation_deg"])
    rotation_spread = torch.stack(pairwise_rotation).max()
    return {
        "world_translation_center_std_mm": float(std_mm),
        "world_translation_range_mm": float(range_mm),
        "world_axis_spread_deg": _axis_spread_deg(poses, axis),
        "world_rotation_symmetry_aware_spread_deg": float(rotation_spread),
    }


def ranking_diagnostics(
    pose_logits: Tensor,
    pose_costs: Tensor,
    target_distribution: Tensor,
) -> dict[str, Tensor]:
    probabilities = torch.softmax(pose_logits, dim=-1)
    eps = torch.finfo(probabilities.dtype).eps
    entropy = lambda value: -(value * value.clamp_min(eps).log()).sum(dim=-1)
    return {
        "pose_cost_min": pose_costs.min(dim=-1).values,
        "pose_cost_max": pose_costs.max(dim=-1).values,
        "pose_cost_mean": pose_costs.mean(dim=-1),
        "pose_cost_std": pose_costs.std(dim=-1, unbiased=False),
        "ranking_target_entropy": entropy(target_distribution),
        "ranking_predicted_entropy": entropy(probabilities),
        "ranking_target_max_probability": target_distribution.max(dim=-1).values,
        "ranking_predicted_max_probability": probabilities.max(dim=-1).values,
    }


__all__ = [
    "TrainingCounters",
    "apply_trainable_prefixes",
    "build_selective_optimizer_parameter_groups",
    "canonical_manifest_bytes",
    "inverse_sqrt_frequency_weights",
    "load_model_initialization",
    "manifest_content_sha256",
    "ranking_diagnostics",
    "region_class_distribution",
    "validate_single_fragment_config",
    "validate_single_fragment_manifest_payload",
    "world_pose_consistency",
]
