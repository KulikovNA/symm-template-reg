"""Builders that first import local modules so registration is deterministic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from symm_template_reg.registry import LOSSES, MODELS, build_from_cfg


def register_all_modules() -> None:
    # Local imports only.  Neighbouring research repositories are never runtime dependencies.
    from symm_template_reg import datasets as _datasets  # noqa: F401
    from . import attention as _attention  # noqa: F401
    from . import backbones as _backbones  # noqa: F401
    from . import geometry as _geometry  # noqa: F401
    from . import pose as _pose  # noqa: F401
    from . import symmetry as _symmetry  # noqa: F401
    from . import heads as _heads  # noqa: F401
    from . import losses as _losses  # noqa: F401
    from . import matching as _matching  # noqa: F401
    from .detectors import symm_template_reg as _detectors  # noqa: F401
    from .detectors import conditioned_symm_template_reg as _conditioned_detectors  # noqa: F401
    from .detectors import uniform_correspondence_procrustes as _uniform_detectors  # noqa: F401
    from .detectors import coordinate_guided_surface_registration_v3 as _clean_v3_detectors  # noqa: F401


def build_model(cfg: Mapping[str, Any]):
    register_all_modules()
    return build_from_cfg(cfg, MODELS)


def build_loss(cfg: Mapping[str, Any]):
    register_all_modules()
    return build_from_cfg(cfg, LOSSES)
