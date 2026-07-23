"""Small MMDetection-style registries used by every configurable component."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, TypeVar


T = TypeVar("T")


class Registry:
    """Map stable config names to Python classes or factory functions."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._modules: dict[str, Callable[..., Any]] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def __len__(self) -> int:
        return len(self._modules)

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, items={sorted(self._modules)})"

    @property
    def module_dict(self) -> dict[str, Callable[..., Any]]:
        return dict(self._modules)

    def get(self, key: str) -> Callable[..., Any] | None:
        return self._modules.get(key)

    def register_module(
        self,
        module: Callable[..., T] | None = None,
        *,
        name: str | None = None,
        force: bool = False,
    ) -> Callable[[Callable[..., T]], Callable[..., T]] | Callable[..., T]:
        def decorator(candidate: Callable[..., T]) -> Callable[..., T]:
            module_name = name or candidate.__name__
            if not force and module_name in self._modules:
                raise KeyError(f"{module_name!r} is already registered in {self.name}")
            self._modules[module_name] = candidate
            return candidate

        if module is None:
            return decorator
        return decorator(module)


def build_from_cfg(
    cfg: Mapping[str, Any],
    registry: Registry,
    default_args: Mapping[str, Any] | None = None,
) -> Any:
    """Instantiate ``cfg['type']`` from ``registry`` without mutating the config."""

    if not isinstance(cfg, Mapping):
        raise TypeError(f"cfg must be a mapping, got {type(cfg).__name__}")
    args = deepcopy(dict(cfg))
    if default_args:
        for key, value in default_args.items():
            args.setdefault(key, value)
    if "type" not in args:
        raise KeyError(f"config for registry {registry.name!r} has no 'type'")
    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        factory = registry.get(obj_type)
        if factory is None:
            raise KeyError(
                f"{obj_type!r} is not registered in {registry.name}; "
                f"available={sorted(registry.module_dict)}"
            )
    elif callable(obj_type):
        factory = obj_type
    else:
        raise TypeError("cfg['type'] must be a registered name or callable")
    try:
        return factory(**args)
    except Exception as exc:
        raise type(exc)(f"while building {factory.__name__} from {registry.name}: {exc}") from exc


MODELS = Registry("models")
BACKBONES = Registry("backbones")
ATTENTION = Registry("attention")
GEOMETRY_MODULES = Registry("geometry_modules")
MATCHERS = Registry("matchers")
HEADS = Registry("heads")
LOSSES = Registry("losses")
POSE_MODULES = Registry("pose_modules")
SYMMETRY_MODULES = Registry("symmetry_modules")
DATASETS = Registry("datasets")
COLLATE_FUNCTIONS = Registry("collate_functions")

