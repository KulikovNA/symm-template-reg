"""Builders that first import local modules so registration is deterministic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from symm_template_reg.registry import LOSSES, MODELS, build_from_cfg


def register_all_modules() -> None:
    # Import only the production graph. Neighbouring repositories and legacy
    # research models are never runtime dependencies or registry entries.
    from symm_template_reg import datasets as _datasets  # noqa: F401
    from .attention.regtr import interaction as _attention  # noqa: F401
    from .backbones import simple_point_encoder as _backbones  # noqa: F401
    from .geometry import dual_stream as _dual_stream  # noqa: F401
    from .geometry import fine_local_correspondence_features as _fine  # noqa: F401
    from .geometry import geometric_embedding as _geometry  # noqa: F401
    from .geometry import ppf as _ppf  # noqa: F401
    from .heads import fine_coordinate_auxiliary_head as _heads  # noqa: F401
    from .losses import clean_coordinate_pose_loss_v3 as _losses  # noqa: F401
    from .pose import weighted_procrustes as _pose  # noqa: F401
    from .detectors import coordinate_guided_surface_registration_v3 as _clean_v3_detectors  # noqa: F401


def build_model(cfg: Mapping[str, Any]):
    register_all_modules()
    return build_from_cfg(cfg, MODELS)


def build_loss(cfg: Mapping[str, Any]):
    register_all_modules()
    return build_from_cfg(cfg, LOSSES)
