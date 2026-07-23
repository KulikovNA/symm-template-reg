"""Fallback only after explicit review of a failed fine-only two-view run."""

_base_ = ["views02.py"]
experiment = dict(name="coordinate_guided_surface_v2_views02_selective_unfreeze")
stage = dict(
    name="TWO_VIEW_coordinate_guided_surface_v2_selective_unfreeze",
    trainable_module_prefixes=(
        "correspondence_head.fine_feature_adapter",
        "correspondence_head.fine_coordinate_auxiliary_head",
        "interaction_transformer.layers.2",
        "dense_observed_fine_projection",
        "fine_template_projection",
    ),
    prefix_learning_rates={
        "correspondence_head.fine_feature_adapter": 1e-4,
        "correspondence_head.fine_coordinate_auxiliary_head": 1e-4,
        "interaction_transformer.layers.2": 1e-5,
        "dense_observed_fine_projection": 1e-5,
        "fine_template_projection": 1e-5,
    },
)
selective_pretrained_learning_rate_options = (1e-5, 3e-5)
