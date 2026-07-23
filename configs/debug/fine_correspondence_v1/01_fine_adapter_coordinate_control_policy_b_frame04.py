"""F1 retry only after adapter-only F1 fails its coordinate gate."""

_base_ = ["01_fine_adapter_coordinate_control_frame04.py"]
experiment = dict(name="fine_adapter_coordinate_control_policy_b_frame04")
stage = dict(
    name="F1_fine_adapter_plus_last_interaction_frame04",
    policy="fine_adapter_plus_last_interaction",
    trainable_module_prefixes=(
        "correspondence_head.fine_feature_adapter",
        "correspondence_head.fine_coordinate_auxiliary_head",
        "interaction_transformer.layers.3",
        "dense_observed_fine_projection",
        "fine_template_projection",
    ),
    prefix_learning_rates={
        "correspondence_head.fine_feature_adapter": 3e-4,
        "correspondence_head.fine_coordinate_auxiliary_head": 3e-4,
        "interaction_transformer.layers.3": 3e-5,
        "dense_observed_fine_projection": 3e-5,
        "fine_template_projection": 3e-5,
    },
)
