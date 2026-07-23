"""Python-config loading with optional lightweight ``_base_`` inheritance."""

from __future__ import annotations

import runpy
import ast
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


_POINT_POLICY_ALIASES = {
    "deterministic_all_or_geometric_cap": "farthest_point_up_to_max",
}


def canonical_point_policy(value: Any) -> str:
    """Return the runtime name for a point-selection policy."""

    policy = str(value)
    return _POINT_POLICY_ALIASES.get(policy, policy)


def validate_data_policy(config: Mapping[str, Any]) -> None:
    """Reject conflicting legacy and authoritative observed-point policies."""

    data = config.get("data")
    dataset = config.get("dataset")
    if not isinstance(data, Mapping) or not isinstance(dataset, Mapping):
        return
    observed_filter = data.get("observed_filter")
    if not isinstance(observed_filter, Mapping) or "point_policy" not in observed_filter:
        return
    legacy = dataset.get("observed_policy")
    if legacy is None:
        return
    authoritative = canonical_point_policy(observed_filter["point_policy"])
    legacy_canonical = canonical_point_policy(legacy)
    if authoritative != legacy_canonical:
        raise ValueError(
            "conflicting observed point policies: "
            f"data.observed_filter.point_policy={observed_filter['point_policy']!r}, "
            f"dataset.observed_policy={legacy!r}; data.observed_filter is authoritative"
        )


def validate_primary_joint_losses(config: Mapping[str, Any]) -> None:
    """Exactly one primary joint objective may own a training step."""

    loss = config.get("loss", {})
    if not isinstance(loss, Mapping):
        return
    names = ("joint_correspondence_pose", "joint_surface_correspondence_pose_v3")
    enabled = [
        name for name in names
        if isinstance(loss.get(name), Mapping) and bool(loss[name].get("enabled", False))
    ]
    if len(enabled) > 1:
        raise ValueError(
            "primary joint losses are mutually exclusive; enabled=" + ", ".join(enabled)
        )


def _merge(base: dict[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in child.items():
        if key.startswith("__") or key == "_base_":
            continue
        if isinstance(value, Mapping) and bool(value.get("_delete_", False)):
            result[key] = {
                child_key: deepcopy(child_value)
                for child_key, child_value in value.items()
                if child_key != "_delete_"
            }
        elif isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    namespace = runpy.run_path(str(config_path))
    values = {key: value for key, value in namespace.items() if not key.startswith("__")}
    bases = values.get("_base_")
    if not bases:
        validate_data_policy(values)
        validate_primary_joint_losses(values)
        return values
    if isinstance(bases, (str, Path)):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        merged = _merge(merged, load_config(base_path))
    result = _merge(merged, values)
    validate_data_policy(result)
    validate_primary_joint_losses(result)
    return result


def apply_overrides(
    config: Mapping[str, Any], overrides: list[str] | None
) -> dict[str, Any]:
    """Apply ``dotted.key=value`` CLI overrides to a copied config."""

    result = deepcopy(dict(config))
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(f"config override must be key=value: {expression!r}")
        dotted_key, raw_value = expression.split("=", 1)
        try:
            value = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            lowered = raw_value.lower()
            if lowered == "true":
                value = True
            elif lowered == "false":
                value = False
            elif lowered in {"none", "null"}:
                value = None
            else:
                value = raw_value
        cursor: dict[str, Any] = result
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                child = {}
                cursor[key] = child
            cursor = child
        cursor[keys[-1]] = value
    validate_data_policy(result)
    validate_primary_joint_losses(result)
    return result


__all__ = [
    "apply_overrides",
    "canonical_point_policy",
    "load_config",
    "validate_data_policy",
    "validate_primary_joint_losses",
]
