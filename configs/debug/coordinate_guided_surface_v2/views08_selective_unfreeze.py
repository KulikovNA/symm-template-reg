"""Prepared eight-view fallback; do not run without external fine-only review."""

_base_ = ["views08.py"]

experiment = dict(name="coordinate_guided_surface_v2_views08_selective_unfreeze")
stage = dict(
    name="EIGHT_VIEW_coordinate_guided_surface_v2_selective_unfreeze",
    initialization="model_only_from_failed_eight_view_fine_only_after_review",
    strict_initialization=True,
    trainable_module_prefixes=(
        "correspondence_head.fine_feature_adapter",
        "correspondence_head.fine_coordinate_auxiliary_head",
        "interaction_transformer.layers.3",
        "dense_observed_fine_projection",
        "fine_template_projection",
    ),
    prefix_learning_rates={
        "correspondence_head.fine_feature_adapter": 1e-4,
        "correspondence_head.fine_coordinate_auxiliary_head": 1e-4,
        "interaction_transformer.layers.3": 1e-5,
        "dense_observed_fine_projection": 1e-5,
        "fine_template_projection": 1e-5,
    },
)
frozen_feature_cache = dict(
    enabled=False,
    disabled_reason="upstream_modules_are_trainable",
    require_passing_audit=True,
    fallback_to_online=True,
)
